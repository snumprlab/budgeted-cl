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

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, DistillationMemory

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class DER(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        if kwargs["temp_batchsize"] is None:
            self.temp_batchsize = self.batch_size - 2 * self.batch_size//3
        self.memory = DistillationMemory(self.dataset, self.memory_size, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia)


    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            train_loss, train_acc, logits = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            self.report_training(sample_num, train_loss, train_acc)
            for i, stored_sample in enumerate(self.temp_batch):
                self.update_memory(stored_sample, logits[i])
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)


    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, alpha=0.5, beta=0.5):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            y2 = data['logit'].to(self.device)
            mask = data['logit_mask'].to(self.device)
            logit, loss = self.model_forward(x, y, y2, mask, memory_batch_size // 2, alpha, beta, use_cutmix=(i != iterations-1))

            _, preds = logit.topk(self.topk, 1, True, True)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            print("self.total_flops", self.total_flops)
            
        return total_loss / iterations, correct / num_data, logit[:stream_batch_size]

    def model_forward(self, x, y, y2=None, mask=None, distill_size=0, alpha=0.5, beta=0.5, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            mask = mask[-distill_size:]
            if do_cutmix:
                x[:-distill_size], labels_a, labels_b, lam = cutmix_data(x=x[:-distill_size], y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.model(x)
                    cls_logit = logit[:-distill_size]
                    cls_loss = lam * criterion(cls_logit, labels_a) + (1 - lam) * criterion(cls_logit, labels_b)
                    
                    self.total_flops += ((len(cls_logit) * 4) / 10e9)
                    
                    loss = cls_loss[:self.temp_batchsize].mean() + alpha * cls_loss[self.temp_batchsize:].mean()
                    
                    self.total_flops += (len(cls_loss)  / 10e9)
                    
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                    
                    self.total_flops += ((distill_size * 4) / 10e9)
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.model(x)
                    cls_logit = logit[:-distill_size]
                    cls_loss = criterion(cls_logit, y)
                    
                    self.total_flops += ((distill_size * 2) / 10e9)
                    
                    loss = cls_loss[:self.temp_batchsize].mean() + alpha * cls_loss[self.temp_batchsize:].mean()
                    
                    self.total_flops += (len(cls_loss) / 10e9)
                    
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                    
                    self.total_flops += ((distill_size * 4)/10e9)
            
            return logit, loss
        else:
            return super().model_forward(x, y)


    def update_memory(self, sample, logit=None):
        self.reservoir_memory(sample, logit)

    def reservoir_memory(self, sample, logit=None):
        self.seen += 1
        #
        if len(self.memory.images) + math.ceil(self.memory.logit_budget) >= self.memory_size:
        #if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
                self.memory.save_logit(logit, j)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_logit(logit)