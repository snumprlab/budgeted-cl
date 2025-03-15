# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class BASELINE(ER):
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

        self.class_std_list = []
        self.sample_std_list = []

        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]

        self.device = device
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'const'
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.memory_size = kwargs["memory_size"]
        self.data_dir = kwargs["data_dir"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        self.memory_size -= self.temp_batchsize

        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        print("model")
        print(self.model)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia)
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []
        self.forgetting = []
        self.knowledge_gain = []
        self.total_knowledge = []
        self.retained_knowledge = []
        self.forgetting_time = []
        self.note = kwargs['note']
        self.rnd_seed = kwargs['rnd_seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_calculated = False
        self.total_flops = 0.0
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        self.total_samples = num_samples[self.dataset]
        
    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.update_memory(sample)
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()


    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=True)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])


            # std check 위해서
            class_std, sample_std = self.memory.get_std()
            self.class_std_list.append(class_std)
            self.sample_std_list.append(sample_std)

            x = torch.cat(x)
            y = torch.cat(y)

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)
            _, preds = logit.topk(self.topk, 1, True, True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            print("total_flops", self.total_flops)
        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(x)
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)
        return logit, loss







