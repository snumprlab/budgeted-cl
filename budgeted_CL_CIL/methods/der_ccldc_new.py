import logging
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils.augment import get_transform
from utils.data_loader import CCLDCLoader
import pandas as pd
from utils.train_utils import kl_loss, select_model, select_optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import MultiProcessLoader, cutmix_data, get_statistics, DistillationMemory

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class DER_CCLDC(CLManagerBase):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"] - 2 * kwargs["batchsize"] // 3
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.model2 = select_model(self.model_name, self.dataset, num_classes = 1, channel_constant=kwargs["channel_constant"], kwinner=kwargs["kwinner"]).to(self.device)
        self.optimizer2 = select_optimizer(self.opt_name, self.lr, self.model2)

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

    def initialize_future(self):
        if self.model_name == 'vit':
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes, self.transform_1, self.transform_2, self.transform_3, self.base_transform, self.normalize = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu, 224, ccldc=True)
        else:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes, self.transform_1, self.transform_2, self.transform_3, self.base_transform, self.normalize = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu, ccldc=True)
        self.data_stream = iter(self.train_datalist)
        self.dataloader = CCLDCLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, transform_1 = self.transform_1, transform_2 = self.transform_2, transform_3 = self.transform_3, base_transform = self.base_transform, normalize = self.normalize)
        self.memory = DERMemory(self.memory_size)

        self.logit_num_to_get = []
        self.logit_num_to_save = []

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

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1

        if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
            self.exposed_domains.append(sample["time"])

        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(max(1, int(self.future_num_updates)))
            temp_batch_logit_num = []
            for stored_sample in self.temp_future_batch:
                logit_num = self.update_memory(stored_sample)
                temp_batch_logit_num.append(logit_num)
            self.logit_num_to_save.append(temp_batch_logit_num)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0


    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            memory_batch, logit_nums = self.memory.retrieval(self.memory_batch_size)
            self.waiting_batch.append(self.temp_future_batch + memory_batch)
            self.logit_num_to_get.append(logit_nums)


    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc, logits = self.online_train(iterations=int(self.num_updates))
                for i, num in enumerate(self.logit_num_to_save[0]):
                    if num is not None:
                        self.memory.save_logits(num, logits[i])
                del self.logit_num_to_save[0]
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            else:
                self.model.train()
                self.model2.train()
                data = self.get_batch()
                x = data["image"].to(self.device)
                logits = self.model(x).detach().cpu()
                for i, num in enumerate(self.logit_num_to_save[0]):
                    if num is not None:
                        self.memory.save_logits(num, logits[i])
                del self.logit_num_to_save[0]
            self.temp_batch = []


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
            if len(self.logit_num_to_get[0]) > 0:
                y2, mask = self.memory.get_logit(self.logit_num_to_get[0], self.num_learned_class)
                y2, mask = y2.to(self.device), mask.to(self.device)
            else:
                y2, mask = [], []
            del self.logit_num_to_get[0]

            self.before_model_update()
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            
            if len(y2)//2 > 0:
                #logit, loss = model_forward(self, x, y, y2, mask, combined_x, combined_aug1, combined_aug2, combined_aug):
                logit, logit2, loss, loss2 = self.model_forward(x, y, not_aug_image, transform_1_image, transform_2_image, transform_3_image, y2, mask)
            else:
                logit, loss = self.model_forward(x, y, not_aug_image, transform_1_image, transform_2_image, transform_3_image, y2, mask)
            
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
            if len(y2)//2 > 0:
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
            if len(y2) > 0:
                return_logit = logit[:-len(y2)].detach().cpu()
            else:
                return_logit = logit.detach().cpu()
        return total_loss / iterations, correct / num_data, return_logit
    
    #def model_forward(self, x, y, y2=None, mask=None, alpha=0.5, beta=0.5):
    def model_forward(self, x, y, combined_x, combined_aug1, combined_aug2, combined_aug, y2=None, mask=None, alpha=0.5, beta=0.5):
        criterion = nn.CrossEntropyLoss(reduction='none')
        distill_size = len(y2)//2
        if distill_size > 0:
            whole_y = y
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            mask = mask[-distill_size:]
        
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                logit2 = self.model2(x)
                
                cls_logit = logit[:-distill_size]
                cls_logit2 = logit2[:-distill_size]
                cls_loss = criterion(cls_logit, y)
                cls_loss2 = criterion(cls_logit2, y)

                logits1 = self.model(combined_aug)
                logits2 = self.model2(combined_aug)

                logits1_vanilla = self.model(combined_x)
                logits2_vanilla = self.model2(combined_x)

                logits1_step1 = self.model(combined_aug1)
                logits2_step1 = self.model2(combined_aug1)

                logits1_step2 = self.model(combined_aug2)
                logits2_step2 = self.model2(combined_aug2)
                
                self.total_flops += ((distill_size * 2) / 10e9)
                self.total_flops += (len(cls_loss) / 10e9)
                
                distill_logit = logit[-distill_size:]
                distill_logit2 = logit2[-distill_size:]
                
                ### DER losses
                loss = cls_loss[:self.temp_batch_size].mean() + alpha * cls_loss[self.temp_batch_size:].mean()
                loss2 = cls_loss2[:self.temp_batch_size].mean() + alpha * cls_loss2[self.temp_batch_size:].mean()
                
                loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                loss2 += beta * (mask * (y2 - distill_logit2) ** 2).mean()
                
                ### CCLDC losses ###
                # Cls Loss
                loss_ce = criterion(logits1, whole_y.long()) + criterion(logits1_vanilla, whole_y.long()) + criterion(logits1_step1, whole_y.long()) + criterion(logits1_step2, whole_y.long())
                loss_ce2 = criterion(logits2, whole_y.long()) + criterion(logits2_vanilla, whole_y.long()) + criterion(logits2_step1, whole_y.long()) + criterion(logits2_step2, whole_y.long())

                # Distillation Loss
                loss_dist = kl_loss(logits1, logits2.detach()) + kl_loss(logits1_vanilla, logits2_step1.detach()) + kl_loss(logits1_step1, logits2_step2.detach()) + kl_loss(logits1_step2, logits2.detach()) 
                loss_dist2 = kl_loss(logits2, logits1.detach()) + kl_loss(logits2_vanilla, logits1_step1.detach()) + kl_loss(logits2_step1, logits1_step2.detach()) + kl_loss(logits2_step2, logits1.detach())

                # Total Loss
                loss += 0.5 * loss_ce.mean() + self.kd_lambda * loss_dist
                loss2 += 0.5 * loss_ce2.mean() + self.kd_lambda * loss_dist2
                
                self.total_flops += ((distill_size * 4)/10e9)

            self.total_flops += (len(x) * self.forward_flops)
            return logit, logit2, loss, loss2
        else:
            return super().model_forward(x, y)


    def update_memory(self, sample):
        logit_num = self.reservoir_memory(sample)
        return logit_num


    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                logit_num = self.memory.replace_sample(sample, j)
            else:
                logit_num = None
        else:
            logit_num = self.memory.replace_sample(sample)
        return logit_num
    
    
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

class DERMemory(MemoryBase):
    def __init__(self, memory_size):
        super().__init__(memory_size)
        self.logits = []
        self.logit_num = []

    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        logit_num = len(self.logits)
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            self.logit_num.append(logit_num)
            self.logits.append(None)
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.logit_num[idx] = logit_num
            self.logits.append(None)
        return logit_num

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])

    def retrieval(self, size, stream_batch=None, return_indices = False):
        if stream_batch is not None:
            stream_file = [x['file_name'] for x in stream_batch]
            memory_file = [x['file_name'] for x in self.images]
            exclude_idx = [i for i, x in enumerate(memory_file) if x in stream_file]
            if len(exclude_idx) > 0:
                print(f"find!!!! {stream_file}")
        sample_size = min(size, len(self.images))
        memory_batch = []
        batch_logit_num = []
        if stream_batch is not None:
            indices = np.random.choice([i for i in range(len(self.images)) if i not in exclude_idx], size=sample_size, replace=False)
        else:
            indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
            batch_logit_num.append(self.logit_num[i])
        if return_indices:
            return memory_batch, batch_logit_num, torch.tensor(indices)
        else:
            return memory_batch, batch_logit_num

    def get_logit(self, logit_nums, num_classes):
        logits = []
        logit_masks = []
        for i in logit_nums:
            len_logit = len(self.logits[i])
            logits.append(torch.cat([self.logits[i], torch.zeros(num_classes-len_logit)]))
            logit_masks.append(torch.cat([torch.ones(len_logit), torch.zeros(num_classes-len_logit)]))
        return torch.stack(logits), torch.stack(logit_masks)

    def save_logits(self, logit_num, logit):
        self.logits[logit_num] = logit

    