# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from typing import List
import torch.nn.functional as F
from methods.lider import LiDER

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase
from methods.er_new import ER

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ER_LiDER(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2

        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.lider = LiDER(self.model, device, self.forward_flops, self.backward_flops)

    def online_before_task(self, task_id):
        self.task_id = task_id
        if self.task_id == 1:
            print("reset lider")
            self.lider.reset_lip_values()

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
            self.before_model_update()

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x,y,i)

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

        return total_loss / iterations, correct / num_data
    
    def model_forward(self,x,y,i):
        stream_x, stream_y = x[:self.temp_batch_size], y[:self.temp_batch_size]

        if i == 0 and self.task_id == 0: self.lider.update_lip_values(stream_x, stream_y)

        with torch.cuda.amp.autocast(self.use_amp):
            logit, features = self.model(x, get_features=True, get_features_detach=False, detached=False, include_out_for_features=True)
            loss = self.criterion(logit, y)
        
        if len(x) > self.temp_batch_size and self.task_id > 0:
            input = x[self.temp_batch_size:]
            for idx in range(len(features)):
                features[idx] = features[idx][self.temp_batch_size:]
                
            features = [input] + features[:-1]
                
            budget_lip_loss = self.lider.budget_lip_loss(features)
            buffer_lip_loss = self.lider.buffer_lip_loss(features)
            loss += budget_lip_loss
            loss += buffer_lip_loss

            print(f"ce loss: {loss.item():.4f}, buffer lip loss: {buffer_lip_loss.item():.4f}, budget lip loss: {budget_lip_loss.item():.4f}")

        features = [f.detach() for f in features]
        return logit, loss
