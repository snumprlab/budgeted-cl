
import copy
import types
import copy
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from methods.cl_manager import CLManagerBase, MemoryBase
from utils.train_utils import select_model, select_optimizer, select_scheduler

from collections import defaultdict
from utils.data_loader import cutmix_data, get_statistics
from utils.data_loader import MultiProcessLoader, XDERLoader

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class CO2L(CLManagerBase):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"] // 2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.model2 = copy.deepcopy(self.model)
        self.model2.eval()
        mean, std, num_class, inp_size, _ = get_statistics(dataset=self.dataset)
        self.tasks = 5
        self.cpt = int(num_class / self.tasks)
        self.train_transform = transforms.Compose([
            transforms.Resize(size=(inp_size, inp_size)),
            transforms.RandomResizedCrop(size=inp_size, scale=(0.1 if self.dataset=='tiny_imagenet' else 0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=inp_size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if inp_size>32 else 0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = XDERLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, self.train_transform)
        self.memory = MemoryBase(self.memory_size)
        self.class_opt = torch.optim.Adam(self.model.fc.parameters(), lr=self.lr)

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.cur_task = 0
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()


    def fit_linear_classifier(self, data):
        x, y = data["image"].to(self.device), data["label"].to(self.device)
        y = y[:len(x)]
        self.model.eval()
        self.model.fc.weight.requires_grad = True
        self.class_opt.zero_grad()
        loss = F.cross_entropy(self.model(x), y)
        loss.backward()
        self.class_opt.step()


    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
                
        for param in self.class_opt.param_groups[0]['params']:
            if param in self.class_opt.state.keys():
                del self.class_opt.state[param]
        del self.class_opt.param_groups[0]
        self.class_opt.add_param_group({'params': self.model.fc.parameters()})
        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
            
            
    def online_before_task(self, cur_iter):
        self.cur_task = cur_iter
        self.model2 = copy.deepcopy(self.model)
    
        
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            self.model.fc.weight.requires_grad = False
                
            data = self.get_batch()
            x = data["image"].to(self.device)
            # need to change name "not_aug_img" to "supcon_img" to express its meaning
            x2 = data["not_aug_img"].to(self.device)
            y = data["label"].to(self.device)
            self.before_model_update()

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(torch.cat([x, x2]), y)
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.total_flops += (len(y) * self.backward_flops)
            self.fit_linear_classifier(data)
            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds[:len(y)] == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data
        
    def model_forward(self, x, y):
        criterion = SupConLoss()
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit, features = self.model(x, get_feature=True)
                features = F.normalize(features, dim=1)
                
                if self.cur_task > 0:
                    features1_prev_task = features
                    features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), 0.2)
                    logits_mask = torch.scatter(
                        torch.ones_like(features1_sim),
                        1,
                        torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                        0
                    )
                    logits_max1, _ = torch.max(features1_sim*logits_mask, dim=1, keepdim=True)
                    features1_sim = features1_sim - logits_max1.detach()
                    row_size = features1_sim.size(0)
                    logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                
                f1, f2 = torch.split(features, [features.size(0)//2, features.size(0)//2], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features, y, target_labels=list(range(self.cur_task*self.cpt, (self.cur_task+1)*self.cpt)))
                
                if self.cur_task > 0:
                    with torch.no_grad():
                        _, features2_prev_task = self.model2(x, get_feature=True)
                        features2_prev_task = F.normalize(features2_prev_task, dim=1)
                        features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), 0.01)
                        logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                        features2_sim = features2_sim - logits_max2.detach()
                        logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
            
                    loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                    loss += loss_distill

        self.total_flops += (len(y) * self.forward_flops)
        return logit, loss

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



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
