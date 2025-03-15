# When we make a new one, we should inherit the Finetune class.
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn import functional as F
from utils.train_utils import select_model, select_scheduler
from torch.utils.data import DataLoader
from utils.data_loader import ImageDataset
import pandas as pd
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics

from methods.cl_manager import MemoryBase

from methods.cl_manager import CLManagerBase

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class MGI_DVC(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.num_classes = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200, 'imagenet': 1000}
        self.input_size = {'cifar10': 32, 'cifar100': 32, 'tinyimagenet': 64, 'imagenet': 224}

        self.num_classes = self.num_classes[self.dataset]
        self.input_size = self.input_size[self.dataset]

        self.model = DVCNet(select_model(self.model_name, self.dataset, self.num_classes), 128, self.num_classes, self.num_classes).to(self.device)
        self.optimizer = self.select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.contrast_transform = nn.Sequential(
            RandomResizedCrop(size=(self.input_size, self.input_size), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1),
            RandomGrayscale(p=0.2)
        )
        self.dl_weight = 2.0

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = MGIMemory(self.memory_size)

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()

    def select_optimizer(self, opt_name, lr, model):
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        return optimizer
    
    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            stream_x = x[:self.temp_batch_size]
            stream_y = y[:self.temp_batch_size]
            self.before_model_update()

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(stream_x, stream_y)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.total_flops += (len(y) * self.backward_flops)

            memory_batch = self.memory.retrieval_live(8)
            print("MEMORY BATCH", memory_batch)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data
    
    def model_forward(self, x, y):
        x_aug = self.contrast_transform(x).to(self.device)

        with torch.cuda.amp.autocast(self.use_amp):
            pred_results = self.model(x, x_aug)
            before_proj_logits_x, before_proj_logits_x_aug, logit, _ = pred_results
            ce = self.cross_entropy_loss(before_proj_logits_x, before_proj_logits_x_aug, y, label_smoothing=0)
            agreement_loss, dl = self.agmax_loss(pred_results, y, dl_weight=self.dl_weight)
            loss  = ce + agreement_loss + dl

        return logit, loss
    
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                _, _, logit, _ = self.model(x, x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def cross_entropy_loss(self, z, zt, ytrue, label_smoothing=0):
        zz = torch.cat((z, zt))
        yy = torch.cat((ytrue, ytrue))
        if label_smoothing > 0:
            ce = LabelSmoothingCrossEntropy(label_smoothing)(zz, yy)
        else:
            ce = nn.CrossEntropyLoss()(zz, yy)
        return ce


    def cross_entropy(self, z, zt):
        # eps = np.finfo(float).eps
        Pz = F.softmax(z, dim=1)
        Pzt = F.softmax(zt, dim=1)
        # make sure no zero for log
        # Pz  [(Pz   < eps).data] = eps
        # Pzt [(Pzt  < eps).data] = eps
        return -(Pz * torch.log(Pzt)).mean()


    def agmax_loss(self, y, ytrue, dl_weight=1.0):
        z, zt, zzt,_ = y
        Pz = F.softmax(z, dim=1)
        Pzt = F.softmax(zt, dim=1)
        Pzzt = F.softmax(zzt, dim=1)

        dl_loss = nn.L1Loss()
        yy = torch.cat((Pz, Pzt))
        zz = torch.cat((Pzzt, Pzzt))
        dl = dl_loss(zz, yy)
        dl *= dl_weight

        # -1/3*(H(z) + H(zt) + H(z, zt)), H(x) = -E[log(x)]
        entropy = self.entropy_loss(Pz, Pzt, Pzzt)
        return entropy, dl

    def clamp_to_eps(self, Pz, Pzt, Pzzt):
        eps = np.finfo(float).eps
        # make sure no zero for log
        Pz[(Pz < eps).data] = eps
        Pzt[(Pzt < eps).data] = eps
        Pzzt[(Pzzt < eps).data] = eps

        return Pz, Pzt, Pzzt


    def batch_probability(self, Pz, Pzt, Pzzt):
        Pz = Pz.sum(dim=0)
        Pzt = Pzt.sum(dim=0)
        Pzzt = Pzzt.sum(dim=0)

        Pz = Pz / Pz.sum()
        Pzt = Pzt / Pzt.sum()
        Pzzt = Pzzt / Pzzt.sum()

        # return Pz, Pzt, Pzzt
        return self.clamp_to_eps(Pz, Pzt, Pzzt)


    def entropy_loss(self, Pz, Pzt, Pzzt):
        # negative entropy loss
        Pz, Pzt, Pzzt = self.batch_probability(Pz, Pzt, Pzzt)
        entropy = (Pz * torch.log(Pz)).sum()
        entropy += (Pzt * torch.log(Pzt)).sum()
        entropy += (Pzzt * torch.log(Pzzt)).sum()
        entropy /= 3
        return entropy

class DVCNet(nn.Module):
    def __init__(self,
                 backbone,
                 n_units,
                 n_classes,
                 current_n_classes,
                 has_mi_qnet=True):
        super(DVCNet, self).__init__()

        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes, current_n_classes=current_n_classes)

    def forward(self, x, x_aug):
        batch_size = x.size(0)
        x_and_x_aug = torch.cat((x, x_aug))
        before_proj_logits, before_proj_feature = self.backbone(x_and_x_aug, get_feature=True) # logits, feature

        before_proj_logits_x = before_proj_logits[:batch_size]
        before_proj_logits_x_aug = before_proj_logits[batch_size:]

        before_proj_feature_x = before_proj_feature[:batch_size]
        before_proj_feature_x_aug = before_proj_feature[batch_size:]

        before_proj_logits_cat = torch.cat((before_proj_logits_x, before_proj_logits_x_aug), dim=1)
        after_proj_logits = self.qnet(before_proj_logits_cat)

        return before_proj_logits_x, before_proj_logits_x_aug, after_proj_logits, \
            [torch.sum(torch.abs(before_proj_feature_x), 1).reshape(-1, 1),torch.sum(torch.abs(before_proj_feature_x_aug), 1).reshape(-1, 1)]
    
    def init_weights(self, std=0.01):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()

class QNet(nn.Module):
    def __init__(self,
                 n_units,
                 n_classes,
                 current_n_classes):
        super(QNet, self).__init__()

        self.q_fc1 = nn.Linear(2 * n_classes, n_units)
        self.q_relu = nn.ReLU(True)
        self.head = nn.Linear(n_units, current_n_classes)

    def forward(self, before_proj_logits_cat):
        # zzt = self.model(zcat)
        out = self.q_fc1(before_proj_logits_cat)
        out = self.q_relu(out)
        out = self.head(out)

        return out
    
    def init_weights(self, std=0.01):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()

class MGIMemory(MemoryBase):
    def __init__(self, memory_size):
        super().__init__(memory_size)
        self.exposed_classes = []

    # not offering meta data like retrieval, but gives live images and labels
    def retrieval_live(self, size):
        metadata = self.retrieval(size)
        # {'file_name': 'train/automobile/2483.png', 'klass': 'automobile', 'label': 0, 'time': 0.0}
        dataset = ImageDataset(
            metadata,
            dataset="cifar10",
            transform=None,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )

        loader = DataLoader(
            dataset,
            batch_size=size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # just return first batch, because we only it's all batch
        for data in loader:
            return data