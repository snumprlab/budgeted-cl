import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class GSS(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)

    def update_memory(self, sample):
        self.gss_memory(sample)

    def gss_memory(self, sample):
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        gpu_test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std)])
        gss_batch = 10
        gss_iter = 10
        if len(self.memory) > 0:
            self.model.eval()
            gss_data = [self.memory.get_batch(min(gss_batch, len(self.memory)), transform=gpu_test_transform) for i in range(gss_iter)]
            grads = []
            for data in gss_data:
                self.optimizer.zero_grad()
                x = data['image'].to(self.device)
                y = data['label'].to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                loss.backward()
                grad = []
                for p in self.model.parameters():
                    if p.requires_grad:
                        grad.append(copy.deepcopy(p.grad.view(-1)))
                grads.append(torch.cat(grad))
            sample_dataset = StreamDataset([sample], dataset=self.dataset, transform=gpu_test_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, transform_on_gpu=True)
            stream_data = sample_dataset.get_data()
            x = stream_data['image'].to(self.device)
            y = stream_data['label'].to(self.device)
            self.optimizer.zero_grad()
            logit = self.model(x)
            loss = self.criterion(logit, y)
            loss.backward()
            sample_grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    sample_grad.append(copy.deepcopy(p.grad.view(-1)))
            sample_grad = torch.cat(sample_grad)
            cos_sim = [F.cosine_similarity(grad, sample_grad, dim=0).item() for grad in grads]
            score = max(cos_sim) + 1
            if len(self.memory) >= self.memory_size:
                if score < 1:
                    mem_scores = np.array(self.memory.score)
                    idx = np.random.choice(range(len(self.memory)), p=(mem_scores+1e-8) / np.sum(mem_scores+1e-8))
                    if np.random.rand() < mem_scores[idx] / (mem_scores[idx] + score):
                        self.memory.replace_sample(sample, idx)
                        self.memory.update_gss_score(score, idx)
            else:
                self.memory.replace_sample(sample)
                self.memory.update_gss_score(score)
        else:
            self.memory.replace_sample(sample)
            self.memory.update_gss_score(2.0)