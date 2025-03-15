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

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import MultiProcessLoader, cutmix_data, get_statistics, DistillationMemory

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
        
class CAMA(CLManagerBase):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"] - 2 * kwargs["batchsize"] // 3
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = DERMemory(self.memory_size)
        self.cls_pred = []
        self.cls_pred_length = 25
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

        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
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

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.cls_pred.append([])

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
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            if len(self.logit_num_to_get[0]) > 0:
                y2, mask = self.memory.get_logit(self.logit_num_to_get[0], self.num_learned_class)
                y2, mask = y2.to(self.device), mask.to(self.device)
            else:
                y2, mask = [], []

            self.before_model_update()
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(self.logit_num_to_get[0], x, y, y2, mask)
            del self.logit_num_to_get[0]
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
            if len(y2) > 0:
                return_logit = logit[:-len(y2)].detach().cpu()
            else:
                return_logit = logit.detach().cpu()
        return total_loss / iterations, correct / num_data, return_logit

    def model_forward(self, logit_indicies, x, y, y2=None, mask=None, alpha=0.5, beta=0.5):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        distill_size = len(y2)//2
        if distill_size > 0:
            whole_y = y
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            mask = mask[-distill_size:]

            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                
                ### recent confidence score update ###
                for idx_y, idx_logit in zip(whole_y, logit.detach()):
                    label_y = idx_y.item()
                    gt_confidence = F.softmax(idx_logit)[label_y].item()
                    self.cls_pred[label_y].append(gt_confidence)
                    if len(self.cls_pred[label_y]) > self.cls_pred_length:
                        del self.cls_pred[label_y][0]
                
                cls_logit = logit[:-distill_size]
                cls_loss = criterion(cls_logit, y)
                
                self.total_flops += ((distill_size * 2) / 10e9)
                loss = cls_loss[:self.temp_batch_size].mean() + alpha * cls_loss[self.temp_batch_size:].mean()
                
                self.total_flops += (len(cls_loss) / 10e9)
                distill_logit = logit[-distill_size:]
                loss += beta * (mask * (y2 - distill_logit) ** 2).mean()
                
                ### update logits ###
                for index, (logit_past, logit_index) in enumerate(zip(y2, logit_indicies)):
                    c = np.mean(self.cls_pred[y.detach()[distill_size + index].item()])
                    # print(y.detach()[distill_size + index].item(), len(self.cls_pred[y.detach()[distill_size + index].item()]), "confidence : ", c)
                    updated_logit = c * logit.detach()[distill_size + index] * (1-c) * logit_past
                    self.memory.logits[logit_index] = updated_logit.cpu()
                
                self.total_flops += ((distill_size * 4)/10e9)

            self.total_flops += (len(x) * self.forward_flops)
            return logit, loss
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
