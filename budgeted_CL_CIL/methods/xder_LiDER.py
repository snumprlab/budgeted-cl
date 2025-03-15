import logging
import random
import copy
import math
import os
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import torch.multiprocessing as multiprocessing
from utils.train_utils import select_model, select_optimizer, select_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from methods.lider import LiDER

from utils.data_loader import XDERLoader, cutmix_data, get_statistics, MemoryDataset

from methods.xder import XDER

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class XDER_LiDER(XDER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.lider = LiDER(self.model, device, self.forward_flops, self.backward_flops)
        print("XDER LiDER")

    def online_before_task(self, cur_iter):
        self.cur_task = cur_iter
        self.task_id = cur_iter

        print("TASK CHANGE", self.task_id)

        if self.task_id == 1:
            print("reset lider")
            self.lider.reset_lip_values()

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations):
            self.model.train()
            data, scl_data = self.get_batch()
            x = data["image"].to(self.device)
            x = transforms.Normalize(self.mean, self.std)(x)
            y = data["label"].to(self.device)
            not_aug_img = data["not_aug_img"].to(self.device)
            
            buf_na_inputsscl = scl_data["image"].to(self.device) if scl_data is not None else None
            buf_labelsscl = scl_data["label"].to(self.device) if scl_data is not None else None
                
            # buf_idx = self.indices.to(self.device)
            buf_idx = torch.Tensor(self.waiting_indices.pop(0)).to(self.device)
            if len(self.logit_num_to_get[0]) > 0:
                y2, mask = self.memory.get_logit(self.logit_num_to_get[0], self.n_classes)
                y2, mask = y2.to(self.device), mask.to(self.device)
            else:
                y2, mask = [], []
            del self.logit_num_to_get[0]

            self.before_model_update()
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x, y, i, y2, mask)
            loss_cons = self.get_consistency_loss(loss, y, not_aug_img, buf_na_inputsscl, buf_labelsscl)
            loss_constr_past, loss_constr_futu = self.get_logit_constraint_loss(loss, logit[:self.temp_batch_size], logit[self.temp_batch_size:], y[self.temp_batch_size:])
            
            loss += loss_cons + loss_constr_futu + loss_constr_past
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

            if self.cur_task > 0:
                with torch.no_grad():
                    # update past logits
                    logit[:self.temp_batch_size] = self.update_memory_logits(y[:self.temp_batch_size], logit[:self.temp_batch_size], logit[:self.temp_batch_size], 0, n_tasks=self.cur_task)
                    
                    # update future past logits
                    chosen = (y[self.temp_batch_size:] // self.cpt) < self.cur_task
                    self.update_counter[buf_idx[chosen]] += 1
                    c = chosen.clone()
                    chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                    if chosen.any():
                        # to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.task, self.tasks - self.task)
                        to_transplant = self.update_memory_logits(y[self.temp_batch_size:], y2, logit[self.temp_batch_size:].detach(), self.cur_task, n_tasks = self.tasks - self.cur_task)
                        '''
                        self.memory.logits[buf_idx[chosen], :] = to_transplant
                        self.memory.task_ids[buf_idx[chosen]] = self.cur_task
                        '''
                        self.memory.update_logits(buf_idx[chosen], to_transplant[chosen])
                        self.memory.update_task_ids(buf_idx[chosen], self.cur_task)

            if len(y2) > 0:
                return_logit = logit[:-len(y2)].detach().cpu()
            else:
                return_logit = logit.detach().cpu()
            self.total_flops += self.lider.total_flops
            self.lider.total_flops = 0
        return total_loss / iterations, correct / num_data, return_logit

    def model_forward(self, x, y, i, y2=None, mask=None, alpha=0.5, beta=0.5):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        stream_x, stream_y = x[:self.temp_batch_size], y[:self.temp_batch_size]
        
        if i == 0 and self.task_id == 0: 
            self.lider.update_lip_values(stream_x, stream_y)

        distill_size = len(y2)//2
        if distill_size > 0:
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            mask = mask[-distill_size:]
            if do_cutmix:
                x[:-distill_size], labels_a, labels_b, lam = cutmix_data(x=x[:-distill_size], y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True, get_features_detach=False, detached=False, include_out_for_features=True)
                    cls_logit = logit[:-distill_size]
                    cls_loss = lam * criterion(cls_logit, labels_a) + (1 - lam) * criterion(cls_logit, labels_b)
                    self.total_flops += ((len(cls_logit) * 4) / 10e9)
                    
                    loss = cls_loss[:self.temp_batch_size].mean() + alpha * cls_loss[self.temp_batch_size:].mean()
                    self.total_flops += (len(cls_loss)  / 10e9)
                    
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                    self.total_flops += ((distill_size * 4) / 10e9)
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True, get_features_detach=False, detached=False, include_out_for_features=True)
                    cls_logit = logit[:-distill_size]
                    cls_loss = criterion(cls_logit, y)
                    self.total_flops += ((distill_size * 2) / 10e9)
                    loss = cls_loss[:self.temp_batch_size].mean() + alpha * cls_loss[self.temp_batch_size:].mean()
                    self.total_flops += (len(cls_loss) / 10e9)
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                    self.total_flops += ((distill_size * 4)/10e9)

            self.total_flops += (len(y) * self.forward_flops)
            
            if len(x) > self.temp_batch_size and self.task_id > 0:
                input = x[self.temp_batch_size:]
                for idx in range(len(features)):
                    features[idx] = features[idx][self.temp_batch_size:]
                
                features = [input] + features[:-1]

                # budget_lip_loss = self.lider.budget_lip_loss(features)
                # buffer_lip_loss = self.lider.buffer_lip_loss(features)
                lip_loss = self.lider.budget_loss(features)
                loss += lip_loss

                # print("lip_loss", lip_loss)

            features = [f.detach() for f in features]
            
            del features
            torch.cuda.empty_cache()

            return logit, loss
        else:
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.model(x)
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.model(x)
                    loss = self.criterion(logit, y)

            self.total_flops += (len(y) * self.forward_flops)
            return logit, loss