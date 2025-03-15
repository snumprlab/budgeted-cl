import logging
import random
import copy
import time
import os
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch.multiprocessing as multiprocessing
from utils.train_utils import select_model, select_optimizer, select_scheduler, SupConLoss, strong_aug, random_grayscale, random_flip, normalize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_worker import worker_loop
from methods.cl_manager import CLManagerBase, MemoryBase
from methods.der_new import DER, DERMemory
from utils.data_loader import XDERLoader, cutmix_data, get_statistics, MemoryDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class XDER(DER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.mean, self.std, num_class, inp_size, _ = get_statistics(dataset=kwargs["dataset"])
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.simclr_lss = SupConLoss(temperature=5, base_temperature=5, reduction='sum')
        self.gpu_augmentation = strong_aug(inp_size, self.mean, self.std)
        self.simclr_temp = 5
        self.simclr_batch_size = 20000
        self.gamma = 0.85
        if 'clear' in self.dataset:
            self.tasks = 10
        else:
            self.tasks = 5
        self.m = 0.2
        self.eta = 0.01
        self.simclr_num_aug = 2
        self.lambd = 0.04
        self.cpt = int(num_class / self.tasks)
        self.update_counter = torch.zeros(kwargs["memory_size"]).to(self.device)
        self.x = time.time()
    

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.scl_dataloader = XDERLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, self.test_transform)
        self.dataloader = XDERLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, self.test_transform)
        self.memory = XDERMemory(self.memory_size, self.n_classes)
        self.model = select_model(self.model_name, self.dataset, self.n_classes).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        self.logit_num_to_get = []
        self.logit_num_to_save = []
        self.indices = []

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.cur_task = 0
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        self.waiting_indices = []
        self.waiting_scl_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)

    def get_batch(self):
        batch = self.dataloader.get_batch()
        scl_batch = self.scl_dataloader.get_batch()
        self.load_batch()
        return batch, scl_batch

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
                data, _ = self.get_batch()
                x = data["image"].to(self.device)
                x = transforms.Normalize(self.mean, self.std)(x)
                logits = self.model(x).detach().cpu()
                for i, num in enumerate(self.logit_num_to_save[0]):
                    if num is not None:
                        self.memory.save_logits(num, logits[i])
                del self.logit_num_to_save[0]
                self.waiting_indices.pop(0)
            self.temp_batch = []

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            memory_batch, logit_nums, indices = self.memory.retrieval(self.memory_batch_size, self.temp_future_batch, return_indices=True)
            scl_memory_batch, _, _ = self.memory.retrieval(min(len(self.temp_future_batch + memory_batch), len(self.memory)), return_indices=True)
            self.waiting_batch.append(self.temp_future_batch + memory_batch)
            self.waiting_indices.append(indices)
            self.waiting_scl_batch.append(scl_memory_batch)
            self.logit_num_to_get.append(logit_nums)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
            
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.scl_dataloader.add_new_class(self.memory.cls_dict)

        if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
            self.exposed_domains.append(sample["time"])
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

    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            self.scl_dataloader.load_batch(self.waiting_scl_batch[0])
            del self.waiting_batch[0]
            del self.waiting_scl_batch[0]

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def online_before_task(self, cur_iter):
        self.cur_task = cur_iter
        if self.dataset == 'clear100' and self.cur_task == 10:
            self.cur_task -= 1

    def online_after_task(self, batchsize=512):
        # Update future past logits
        if len(self.memory) > 0:
            self.model.eval()
            with torch.no_grad():
                exclude_index = self.flatten(self.logit_num_to_save)
                memory_dataset = XDERMemoryDataset(self.memory, exclude_index, self.test_transform, self.data_dir)
                memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=batchsize)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        for images, labels, past_logits, indices in memory_loader:
                            logits = self.model(images).cpu()
                            self.total_flops += len(images) * self.forward_flops
                            chosen = (labels // self.cpt) < self.cur_task
                            if chosen.any():
                                to_transplant = self.update_memory_logits(labels[chosen], past_logits[chosen], logits[chosen], self.cur_task, self.tasks - self.cur_task)
                                self.memory.update_logits(indices[chosen], to_transplant)
                                self.memory.update_task_ids(indices[chosen], self.cur_task)
                
                '''
                include_index = np.array(sorted(list(set(range(len(self.memory.images))) - set(exclude_index))))
                include_index_tensor = torch.Tensor(include_index).long()
                x = torch.stack(x)[include_index_tensor.to(self.device)]
                y = torch.LongTensor(self.memory.labels)[[include_index]].to(self.device)
                logits = torch.stack(np.array(self.memory.logits)[include_index].tolist())
                buf_idxs = torch.arange(len(self.memory.images))[include_index].to(self.device)
                logits = logits[include_index].to(self.device)                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        for i in range(-(-len(x) // batchsize)):
                            past_logit = logits[i * batchsize:min((i + 1) * batchsize, len(x))]
                            buf_idx = buf_idxs[i * batchsize:min((i + 1) * batchsize, len(x))]
                            logit = self.model(x[i * batchsize:min((i + 1) * batchsize, len(x))].to(self.device))
                            self.total_flops += len(x[i * batchsize:min((i + 1) * batchsize, len(x))]) * self.forward_flops
                            chosen = (y[i * batchsize:min((i + 1) * batchsize, len(x))] // self.cpt) < self.cur_task
                            if chosen.any():
                                to_transplant = self.update_memory_logits(y[i * batchsize:min((i + 1) * batchsize, len(x))][chosen], past_logit[chosen], logit[chosen], self.cur_task, self.tasks - self.cur_task)
                                self.memory.update_logits(buf_idx[chosen], to_transplant)
                                self.memory.update_task_ids(buf_idx[chosen], self.cur_task)
                '''
        self.update_counter = torch.zeros(self.memory_size).to(self.device)

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
            logit, loss = self.model_forward(x, y, y2, mask)
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
        return total_loss / iterations, correct / num_data, return_logit

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

    def reservoir_memory(self, sample):
        # self.seen += 1
        # try:
        #     img_name = sample['file_name']
        # except KeyError:
        #     img_name = sample['filepath']
        # if self.data_dir is None:
        #     img_path = os.path.join("dataset", self.dataset, img_name)
        # else:
        #     img_path = os.path.join(self.data_dir, img_name)
        # img = PIL.Image.open(img_path).convert('RGB')
        # img = self.train_transform(img).to(self.device)
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                logit_num = self.memory.replace_sample(sample, j)
                self.memory.save_task_id(self.cur_task, j)
            else:
                logit_num = None
        else:
            logit_num = self.memory.replace_sample(sample)
            self.memory.save_task_id(self.cur_task)
        return logit_num

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
                x = transforms.Normalize(self.mean, self.std)(x)
                logit = self.model(x)

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


    def update_memory_logits(self, gt, old, new, cur_task, n_tasks = 1):
        #transplant = new[task_mask][torch.arange(new[task_mask]), self.cur_task]
        transplant = new[:, cur_task * self.cpt : (cur_task + n_tasks) * self.cpt]
        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, cur_task * self.cpt:(cur_task + n_tasks) * self.cpt] = transplant
        return old
    
    def get_consistency_loss(self, loss, y, not_aug_inputs, buf_na_inputsscl, buf_labelsscl):
        # Consistency Loss (future heads)
        loss_cons = torch.tensor(0.)
        loss_cons = loss_cons.type(loss.dtype)
        if self.cur_task < self.tasks - 1:
            scl_labels = y[:min(self.simclr_batch_size, len(y))]
            scl_na_inputs = not_aug_inputs[:min(self.simclr_batch_size, len(y))]
            
            #buf_idxscl, buf_na_inputsscl, buf_labelsscl, buf_logitsscl, _ = self.get_batch(min(self.simclr_batch_size, len(self.memory.images)))
            # data = self.get_memory_batch(min(self.simclr_batch_size, len(self.memory.images)))
            # buf_na_inputsscl = data["image"].to(self.device)
            # buf_labelsscl = data["label"].to(self.device)
            if buf_na_inputsscl is not None:
                scl_na_inputs = torch.cat([buf_na_inputsscl, scl_na_inputs])
                scl_labels = torch.cat([buf_labelsscl, scl_labels])
            with torch.no_grad():
                scl_inputs = self.gpu_augmentation(scl_na_inputs.repeat_interleave(self.simclr_num_aug, 0)).to(self.device)
            #with bn_track_stats(self, False):
            scl_outputs = self.model(scl_inputs).float()
            self.total_flops += len(scl_inputs) * self.forward_flops

            scl_featuresFull = scl_outputs.reshape(-1, self.simclr_num_aug, scl_outputs.shape[-1])  # [N, n_aug, 100]

            scl_features = scl_featuresFull[:, :, (self.cur_task + 1) * self.cpt:]  # [N, n_aug, 70]
            scl_n_heads = self.tasks - self.cur_task - 1

            scl_features = torch.stack(scl_features.split(self.cpt, 2), 1)  # [N, 7, n_aug, 10]
            loss_cons = torch.stack([self.simclr_lss(features=F.normalize(scl_features[:, h], dim=2), labels=scl_labels) for h in range(scl_n_heads)]).sum()
            loss_cons /= scl_n_heads * scl_features.shape[0]
            loss_cons *= self.lambd
        return loss_cons


    def get_logit_constraint_loss(self, loss_stream, outputs, buf_outputs, buf_labels):
        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.cur_task > 0:
            chead = F.softmax(outputs[:, :(self.cur_task + 1) * self.cpt], 1)

            good_head = chead[:, self.cur_task * self.cpt:(self.cur_task + 1) * self.cpt]
            bad_head = chead[:, :self.cpt * self.cur_task]

            loss_constr = bad_head.max(1)[0].detach() + self.m - good_head.max(1)[0]

            mask = loss_constr > 0

            if (mask).any():
                loss_constr_past = self.eta * loss_constr[mask].mean()

        # Future Logits Constraint
        loss_constr_futu = torch.tensor(0.)
        if self.cur_task < self.tasks - 1:
            bad_head = outputs[:, (self.cur_task + 1) * self.cpt:]
            good_head = outputs[:, self.cur_task * self.cpt:(self.cur_task + 1) * self.cpt]

            if len(self.memory.images) > 0:
                buf_tlgt = buf_labels // self.cpt
                bad_head = torch.cat([bad_head, buf_outputs[:, (self.cur_task + 1) * self.cpt:]])
                good_head = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

            loss_constr = bad_head.max(1)[0] + self.m - good_head.max(1)[0]

            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = self.eta * loss_constr[mask].mean()
        
        return loss_constr_past, loss_constr_futu
        

class XDERMemory(DERMemory):
    def __init__(self, memory_size, n_classes):
        super().__init__(memory_size)  
        self.task_ids = []  
        self.n_classes = n_classes
        
    def update_logits(self, indices, new_logits):
        for new_logit, indice in zip(new_logits, indices) :
            self.logits[indice] = new_logit.cpu()
            
    def update_task_ids(self, indices, new_task_id):
        for indice in indices:
            self.task_ids[indice] = new_task_id
            
    def save_task_id(self, task_id, idx=None):
        if idx is None:
            self.task_ids.append(task_id)
        else:
            self.task_ids[idx] = task_id

    def replace_sample(self, sample, idx=None, logit=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        logit_num = len(self.logits)
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            self.logit_num.append(logit_num)
            self.logits.append(logit)
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.logit_num[idx] = logit_num
            self.logits.append(logit)
        return logit_num


class XDERMemoryDataset(Dataset):
    def __init__(self, memory, exclude_idx, transform, data_dir=None, device="cuda"):
        self.memory = memory
        self.transform = transform
        self.data_dir = data_dir
        self.device = device
        
        self.image_path = [self.memory.images[i] for i in range(len(self.memory)) if i not in exclude_idx]
        self.logit = [self.memory.logits[i] for i in range(len(self.memory)) if i not in exclude_idx]
        self.indices = [i for i in range(len(self.memory)) if i not in exclude_idx]
    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]['file_name']
        if self.data_dir is None:
            img_path = os.path.join("dataset", self.dataset, image_path)
        else:
            img_path = os.path.join(self.data_dir, image_path)
        img = PIL.Image.open(img_path).convert('RGB')
        img = self.transform(img).to(self.device)  
        label = self.image_path[idx]['label']
        return img, label, self.logit[idx], self.indices[idx]
