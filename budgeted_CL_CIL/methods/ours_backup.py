# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from collections import Counter
from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, cutmix_data_two, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.focal_loss import FocalLoss
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Ours(ER):
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

        self.avg_prob = kwargs["avg_prob"]

        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]

        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.weight_ema_ratio = kwargs["weight_ema_ratio"]
        self.use_batch_cutmix = kwargs["use_batch_cutmix"]
        self.device = device
        self.klass_warmup = kwargs["klass_warmup"]
        self.loss_balancing_option = kwargs["loss_balancing_option"]
        self.use_class_balancing = kwargs["use_class_balancing"]
        print("self.use_class_balancing", self.use_class_balancing)
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

        self.klass_train_warmup = kwargs["klass_train_warmup"]
        self.memory_size = kwargs["memory_size"]
        self.data_dir = kwargs["data_dir"]
        #self.use_human_training = kwargs["use_human_training"]
        self.use_human_training = False

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        self.memory_size -= self.temp_batchsize

        self.recent_ratio = kwargs["recent_ratio"]
        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp = kwargs["use_amp"]
        self.cls_weight_decay = kwargs["cls_weight_decay"]

        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)

        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        
        '''
        criterion_kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        self.criterion = FocalLoss(**criterion_kwargs)
        '''
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia, 
                                    cls_weight_decay = self.cls_weight_decay, weight_option = self.weight_option, 
                                    weight_ema_ratio = self.weight_ema_ratio, use_human_training=self.use_human_training, 
                                    klass_warmup = self.klass_warmup, klass_train_warmup = self.klass_train_warmup)
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]
        self.total_count = 0
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
            self.add_new_class(sample['klass'], sample)

        self.total_count += 1
        self.update_memory(sample, self.total_count)
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def add_new_class(self, class_name, sample=None):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        self.memory.add_new_class(cls_list=self.exposed_classes, sample=sample)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

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
            x2 = []
            y2 = []
            
            #if len(self.memory) > 0:
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight = True, exp_weight=True, recent_ratio = self.recent_ratio)
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight=False, use_human_training=True)
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight="classwise", weight_method = self.weight_method, n_class = self.num_learned_class, avg_prob = self.avg_prob)
            memory_data = self.memory.get_batch(memory_batch_size, use_weight="classwise", weight_method = self.weight_method)
            
            # std check 위해서
            class_std, sample_std = self.memory.get_std()
            self.class_std_list.append(class_std)
            self.sample_std_list.append(sample_std)
            
            x.append(memory_data['image'])
            y.append(memory_data['label'])
            cls_weight = memory_data['cls_weight']
            counter = memory_data["counter"]

            if self.use_batch_cutmix:
                memory_data2 = self.memory.get_batch(memory_batch_size, prev_batch_index=memory_data['indices'])
                x2.append(memory_data2['image'])
                y2.append(memory_data2['label'])
                x2 = torch.cat(x2)
                y2 = torch.cat(y2)
                x2 = x2.to(self.device)
                y2 = y2.to(self.device)
                    
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.weight_option == "loss":
                if self.use_batch_cutmix:
                    logit, loss, cls_loss = self.model_forward(x, y, x2, y2, return_cls_loss=True, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                else:
                    logit, loss, cls_loss = self.model_forward(x, y, return_cls_loss=True, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                
                if cls_loss is not None:
                    self.memory.update_sample_loss(memory_data['indices'], cls_loss.detach().cpu())
                    self.memory.update_class_loss(memory_data['indices'], cls_loss.detach().cpu())
            else:
                if self.use_batch_cutmix:
                    logit, loss = self.model_forward(x, y, x2, y2, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                else:
                    logit, loss = self.model_forward(x, y, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                
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

    def model_forward(self, x, y, x2=None, y2=None, return_cls_loss=False, loss_balancing_option=None, loss_balancing_weight=None):
        '''
        self.criterion의 reduction
        1) none
        just element wise 곱셈
        
        2) mean
        element wise 곱셈 후 sum, 후 나눗셈

        3) sum
        just element wise 곱셈 후 sum
        '''
        if loss_balancing_option == "reverse_class_weight":    
            loss_balancing_weight = 1 / np.array(loss_balancing_weight)
            loss_balancing_weight /= np.sum(loss_balancing_weight)
        elif loss_balancing_option == "class_weight": # default
            pass
        else:
            pass
        
        mix_data = x2 is not None and y2 is not None

        #do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        
        do_cutmix = self.cutmix and np.random.rand(1) < self.memory.get_ratio(y)

        #TODO Cutmix를 0.5의 확률이 아닌, class별로 magnitude 기반으로 바꿔주기
        
        if do_cutmix or mix_data:
            if mix_data:
                x, labels_a, labels_b, lam = cutmix_data_two(x=x, y=y, x2=x2, y2=y2, alpha=1.0)    
            else:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(x)
                    if self.weight_option == "loss": # 이때는 self.criterion(reduction="none")
                        loss_a = self.criterion(logit, labels_a) # 1
                        loss_b = self.criterion(logit, labels_b) # 1
                        total_loss = lam * loss_a + (1 - lam) * (loss_b) # 3
                        
                        if loss_balancing_weight is None:
                            loss = torch.mean(total_loss) # 1
                        else:
                            for i, sample_loss in enumerate(total_loss):
                                if i==0:
                                    loss = loss_balancing_weight[i] * sample_loss
                                else:
                                    loss += loss_balancing_weight[i] * sample_loss
                            
                        self.total_flops += (len(logit) * 6) / 10e9
                    else: # 이때는 self.criterion(reduction="mean")
                        loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) # 4
                        self.total_flops += (len(logit) * 4) / 10e9
            else:
                logit = self.model(x)
                if self.weight_option == "loss":
                    loss_a = self.criterion(logit, labels_a)
                    loss_b = self.criterion(logit, labels_b)
                    total_loss = lam * loss_a + (1 - lam) * (loss_b)
                    loss = torch.mean(total_loss)
                    self.total_flops += (len(logit) * 6) / 10e9
                else:
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                    self.total_flops += (len(logit) * 4) / 10e9
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(x)
                    if self.weight_option == "loss":
                        total_loss = self.criterion(logit, y)
                        loss = torch.mean(total_loss)
                    else:
                        loss = self.criterion(logit, y)
                    self.total_flops += (len(logit) * 2) / 10e9
            else:
                logit = self.model(x)
                if self.weight_option == "loss":
                    total_loss = self.criterion(logit, y)

                    if loss_balancing_weight is None:
                        loss = torch.mean(total_loss) # 1
                    else:
                        for i, sample_loss in enumerate(total_loss):
                            if i==0:
                                loss = loss_balancing_weight[i] * sample_loss
                            else:
                                loss += loss_balancing_weight[i] * sample_loss
                else:
                    loss = self.criterion(logit, y)

                self.total_flops += (len(logit) * 2) / 10e9
        if return_cls_loss:
            if do_cutmix:
                return logit, loss, None
            else:
                return logit, loss, total_loss
        else:
            return logit, loss

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
                logit = self.model(x)

                if self.weight_option == "loss":
                    loss = torch.mean(criterion(logit, y))
                else:
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

    def save_std_pickle(self):
        '''
        class_std, sample_std = self.memory.get_std()
        self.class_std_list.append(class_std)
        self.sample_std_list.append(sample_std)
        '''
        
        with open('ours_cls_std_real.pickle', 'wb') as handle:
            pickle.dump(self.class_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('ours_sample_std_real.pickle', 'wb') as handle:
            pickle.dump(self.sample_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    '''
    def report_training(self, sample_num, train_loss, train_acc):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, online_acc):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        writer.add_scalar(f"test/online_acc", online_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | online_acc {online_acc:.4f} "
        )
    '''

    def update_memory(self, sample, count):
        #self.reservoir_memory(sample)
        #self.memory.replace_sample(sample, count=count)
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.memory.cls_dict(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            idx_to_replace = random.choice(cand_idx) 
            #score = self.memory.others_loss_decrease[cand_idx]
            #idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
