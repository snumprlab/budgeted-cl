import copy
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase
from utils.data_loader import ImageDataset, cutmix_data
from utils.data_worker import load_data

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class SDP(CLManagerBase):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.memory_size = kwargs["memory_size"]

        self.sdp_mean = kwargs['sdp_mean']
        self.sdp_varcoeff = kwargs['sdp_var']
        assert 0.5 - 1 / self.sdp_mean < self.sdp_varcoeff < 1 - 1 / self.sdp_mean
        self.ema_ratio_1 = (1 - np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.ema_ratio_2 = (1 + np.sqrt(2 * self.sdp_varcoeff - 1 + 2 / self.sdp_mean)) / (
                self.sdp_mean - 1 - self.sdp_mean * self.sdp_varcoeff)
        self.cur_time = None
        self.sdp_model = copy.deepcopy(self.model)
        self.ema_model_1 = copy.deepcopy(self.model)
        self.ema_model_2 = copy.deepcopy(self.model)
        self.sdp_updates = 0
        self.fc_mode = kwargs['fc_train']
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.temp_ret = None
        self.cls_pred_length = 100
        self.cls_pred = []

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.update_memory(sample)
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if self.future_num_updates >= 1:
            self.temp_future_batch = []
            self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def update_memory(self, sample):
        self.balanced_replace_memory(sample)

    def balanced_replace_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.memory.cls_dict[sample['klass']]] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

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
        self.sdp_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        self.cls_pred.append([])
        if self.fc_mode != 'none':
            sdict = copy.deepcopy(self.optimizer.state_dict())
            fc_params = sdict['param_groups'][1]['params']
            if len(sdict['state']) > 0:
                fc_weight_state = sdict['state'][fc_params[0]]
                fc_bias_state = sdict['state'][fc_params[1]]
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
            if len(sdict['state']) > 0:
                if 'adam' in self.opt_name:
                    fc_weight = self.optimizer.param_groups[1]['params'][0]
                    fc_bias = self.optimizer.param_groups[1]['params'][1]
                    self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                    self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                            torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                                dim=1)]).to(self.device)], dim=0)
                    self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                               torch.zeros([1, fc_weight_state[
                                                                                   'exp_avg_sq'].size(dim=1)]).to(
                                                                                   self.device)], dim=0)
                    self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                    self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                          torch.tensor([0]).to(
                                                                              self.device)], dim=0)
                    self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                             torch.tensor([0]).to(
                                                                                 self.device)], dim=0)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.num_updates += self.online_iter
        self.sample_inference(sample)
        self.update_schedule()

        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))

            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)

    def sample_inference(self, sample):
        self.sdp_model.eval()
        x = load_data(sample, self.data_dir, self.test_transform).unsqueeze(0)
        y = self.cls_dict[sample['klass']]
        x = x.to(self.device)
        logit = self.sdp_model(x)

        self.total_flops += self.forward_flops

        prob = F.softmax(logit, dim=1)
        self.cls_pred[y].append(prob[0, y].item())
        if len(self.cls_pred[y]) > self.cls_pred_length:
            del self.cls_pred[y][0]
        self.cls_pred_mean = np.clip(np.mean([np.mean(cls_pred) for cls_pred in self.cls_pred]) - 1/self.num_learned_class, 0, 1) * self.num_learned_class/(self.num_learned_class + 1)

    def update_schedule(self, reset=False):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * (1 - self.cls_pred_mean)

    def after_model_update(self):
        self.update_sdp_model(num_updates=1.0)

    @torch.no_grad()
    def update_sdp_model(self, num_updates=1.0):
        ema_inv_ratio_1 = (1 - self.ema_ratio_1) ** num_updates
        ema_inv_ratio_2 = (1 - self.ema_ratio_2) ** num_updates
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.sdp_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        self.sdp_updates += 1
        for name, param in model_params.items():
            ema_params_1[name].sub_((1. - ema_inv_ratio_1) * (ema_params_1[name] - param))
            ema_params_2[name].sub_((1. - ema_inv_ratio_2) * (ema_params_2[name] - param))
            ema_params[name].copy_(
                self.ema_ratio_2 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_1[name] - self.ema_ratio_1 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_2[
                    name])
            # + ((1. - self.ema_ratio_2)*self.ema_ratio_1**self.ema_updates - (1. - self.ema_ratio_1)*self.ema_ratio_2**self.ema_updates) / (self.ema_ratio_1 - self.ema_ratio_2) * param)
        self.sdp_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.sdp_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

        self.total_flops += (9 * self.params)

    def model_forward(self, x, y, distill=True, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.sdp_model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        logit2, feature2 = self.sdp_model(x, get_feature=True)
                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    self.total_flops += ((feature.shape[0] * feature.shape[1] * 3) / 10e9)

                    sample_weight = self.cls_pred_mean
                    grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                            1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    self.total_flops += (grad.shape[0] * grad.shape[1] * 3 + len(distill_loss)) / 10e9
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    self.sdp_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.sdp_model(x, get_feature=True)
                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    self.total_flops += ((feature.shape[0] * feature.shape[1] * 3) / 10e9)

                    sample_weight = self.cls_pred_mean
                    grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    self.total_flops += (grad.shape[0] * grad.shape[1] * 3 + len(distill_loss)) / 10e9
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()

            self.total_flops += (len(y) * 2 * self.forward_flops)
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)

        front = (prob - oh_label).shape
        back = weight.shape
        self.total_flops += ((front[0] * back[1] * (2 * front[1] - 1)) / 10e9)

        return torch.matmul((prob - oh_label), weight)
