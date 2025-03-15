# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
from methods.er_new import ER
from utils.data_loader import CCLDCLoader
from utils.train_utils import kl_loss
from flops_counter.ptflops import get_model_complexity_info
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.augment import get_transform
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.block_utils import MODEL_BLOCK_DICT, get_blockwise_flops
from methods.cl_manager import MemoryBase
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ER_CCLDC(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.model2 = select_model(self.model_name, self.dataset, num_classes = 1, channel_constant=kwargs["channel_constant"], kwinner=kwargs["kwinner"]).to(self.device)
        self.optimizer2 = select_optimizer(self.opt_name, self.lr, self.model2)

    def initialize_future(self):
        if self.model_name == 'vit':
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes, self.transform_1, self.transform_2, self.transform_3, self.base_transform, self.normalize = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu, 224, ccldc=True)
        else:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes, self.transform_1, self.transform_2, self.transform_3, self.base_transform, self.normalize = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu, ccldc=True)
        
        self.data_stream = iter(self.train_datalist)
        self.dataloader = CCLDCLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, transform_1 = self.transform_1, transform_2 = self.transform_2, transform_3 = self.transform_3, base_transform = self.base_transform, normalize = self.normalize)
        self.memory = MemoryBase(self.memory_size)

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
        
        # ccldc hyperparameter
        self.kd_lambda = 1

        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def add_new_class(self, class_name):
        if hasattr(self.model, 'fc'):
            fc_name = 'fc'
        elif hasattr(self.model, 'head'):
            fc_name = 'head'
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        # model 1
        model_fc = getattr(self.model, fc_name)
        prev_weight = copy.deepcopy(model_fc.weight.data)
        prev_bias = copy.deepcopy(model_fc.bias.data)
        setattr(self.model, fc_name, nn.Linear(model_fc.in_features, self.num_learned_class).to(self.device))
        model_fc = getattr(self.model, fc_name)
        with torch.no_grad():
            if self.num_learned_class > 1:
                model_fc.weight[:self.num_learned_class - 1] = prev_weight
                model_fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': model_fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        
        # model 2
        model_fc = getattr(self.model2, fc_name)
        prev_weight = copy.deepcopy(model_fc.weight.data)
        prev_bias = copy.deepcopy(model_fc.bias.data)
        setattr(self.model2, fc_name, nn.Linear(model_fc.in_features, self.num_learned_class).to(self.device))
        model_fc = getattr(self.model2, fc_name)
        with torch.no_grad():
            if self.num_learned_class > 1:
                model_fc.weight[:self.num_learned_class - 1] = prev_weight
                model_fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer2.param_groups[1]['params']:
            if param in self.optimizer2.state.keys():
                del self.optimizer2.state[param]
        del self.optimizer2.param_groups[1]
        self.optimizer2.add_param_group({'params': model_fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def model_forward(self, x, y, combined_x, combined_aug1, combined_aug2, combined_aug):
        
        logits1, feature = self.model(x, get_feature=True)
        logits2, feature2 = self.model2(x, get_feature=True)
        
        loss_ce = self.criterion(logits1, y)
        loss_ce2 = self.criterion(logits2, y)
        
        # Augment
        # combined_aug1 = self.transform_1(combined_x)
        # combined_aug2 = self.transform_2(combined_aug1)
        # combined_aug = self.transform_3(combined_aug2)

        logits1 = self.model(combined_aug)
        logits2 = self.model2(combined_aug)

        logits1_vanilla = self.model(combined_x)
        logits2_vanilla = self.model2(combined_x)

        logits1_step1 = self.model(combined_aug1)
        logits2_step1 = self.model2(combined_aug1)

        logits1_step2 = self.model(combined_aug2)
        logits2_step2 = self.model2(combined_aug2)
        
        # Cls Loss
        loss_ce += self.criterion(logits1, y.long()) + self.criterion(logits1_vanilla, y.long()) + self.criterion(logits1_step1, y.long()) + self.criterion(logits1_step2, y.long())
        loss_ce2 += self.criterion(logits2, y.long()) + self.criterion(logits2_vanilla, y.long()) + self.criterion(logits2_step1, y.long()) + self.criterion(logits2_step2, y.long())

        # Distillation Loss
        loss_dist = kl_loss(logits1, logits2.detach()) + kl_loss(logits1_vanilla, logits2_step1.detach()) + kl_loss(logits1_step1, logits2_step2.detach()) + kl_loss(logits1_step2, logits2.detach()) 
        loss_dist2 = kl_loss(logits2, logits1.detach()) + kl_loss(logits2_vanilla, logits1_step1.detach()) + kl_loss(logits2_step1, logits1_step2.detach()) + kl_loss(logits2_step2, logits1.detach())

        # Total Loss
        loss = 0.5 * loss_ce  + self.kd_lambda * loss_dist 
        loss2 = 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

        self.total_flops += ((len(y) * self.forward_flops) * 10)
        return logits1, logits2, loss, loss2

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            self.model2.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            transform_1_image = data["transform_1_image"].to(self.device)
            transform_2_image = data["transform_2_image"].to(self.device)
            transform_3_image = data["transform_3_image"].to(self.device)
            not_aug_image = data["not_aug_image"].to(self.device)
            self.before_model_update()

            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            logit, logit2, loss, loss2 = self.model_forward(x,y,not_aug_image,transform_1_image,transform_2_image,transform_3_image)

            _, preds = logit.topk(self.topk, 1, True, True)

            # model1 update
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            # model2 update
            if self.use_amp:
                self.scaler.scale(loss2).backward()
                self.scaler.step(self.optimizer2)
                self.scaler.update()
            else:
                loss2.backward()
                self.optimizer2.step()

            self.total_flops += (len(y) * self.backward_flops) * 2
            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        self.model2.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit1 = self.model(x)
                logit2 = self.model2(x)
                logit = (logit1 + logit2) / 2

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

