import logging
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from methods.lider import LiDER
from methods.der_new import DER

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import MultiProcessLoader, cutmix_data, get_statistics, DistillationMemory

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class DER_LiDER(DER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"] - 2 * kwargs["batchsize"] // 3
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.lider = LiDER(self.model, device, self.forward_flops, self.backward_flops)
        print("DER LiDER")

    def online_before_task(self, task_id):
        self.task_id = task_id
        if self.task_id == 1: 
            print("reset lider")
            self.lider.reset_lip_values()
        
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            if len(self.logit_num_to_get[0]) > 0:
                y2, mask = self.memory.get_logit(self.logit_num_to_get[0], self.num_learned_class)
                y2, mask = y2.to(self.device), mask.to(self.device)
            else:
                y2, mask = [], []
            del self.logit_num_to_get[0]

            self.before_model_update()
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x, y, i, y2, mask)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.total_flops += (len(y) * (self.backward_flops))

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            if len(y2) > 0:
                return_logit = logit[:-len(y2)].detach().cpu()
            else:
                return_logit = logit.detach().cpu()
        # add total flops of LiDER
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
                for idx in range(len(features)):
                    features[idx] = features[idx][self.temp_batch_size:]
                
                features = [x[self.temp_batch_size:]] + features[:-1]

                budget_lip_loss = self.lider.budget_lip_loss(features)
                buffer_lip_loss = self.lider.buffer_lip_loss(features)
                loss += budget_lip_loss
                loss += buffer_lip_loss

                print(f"ce loss: {loss.item():.4f}, buffer lip loss: {buffer_lip_loss.item():.4f}, budget lip loss: {budget_lip_loss.item():.4f}")

            features = [f.detach() for f in features]

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