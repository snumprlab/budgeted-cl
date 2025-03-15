import copy
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_ind
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from methods.er_baseline import ER
from models.layers import StatTrack
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, \
    DistillationMemory, cutmix_feature, get_test_datalist
from utils.train_utils import select_optimizer

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class Ours(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = kwargs["memory_size"]

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

    def update_memory(self, sample):
        self.balanced_replace_memory(sample)

    def balanced_replace_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        sdict = copy.deepcopy(self.optimizer.state_dict())
        fc_params = sdict['param_groups'][1]['params']
        if len(sdict['state']) > 0:
            fc_weight_state = sdict['state'][fc_params[0]]
            fc_bias_state = sdict['state'][fc_params[1]]
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        if len(sdict['state']) > 0:
            if 'adam' in self.opt_name:
                fc_weight = self.optimizer.param_groups[1]['params'][0]
                fc_bias = self.optimizer.param_groups[1]['params'][1]
                self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                        torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                            dim=1)]).to(self.device)], dim=0)
                self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                           torch.zeros([1, fc_weight_state[
                                                                               'exp_avg_sq'].size(dim=1)]).to(
                                                                               self.device)], dim=0)
                self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                      torch.tensor([0]).to(
                                                                          self.device)], dim=0)
                self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                         torch.tensor([0]).to(
                                                                             self.device)], dim=0)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)


class Ours_v2(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)

        self.class_ema = torch.ones(n_classes) / n_classes

        self.time_records = [[] for i in range(n_classes)]
        self.corrected_avg_accs = {}
        self.corrected_online_accs = {}

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        label = self.memory.cls_dict[sample['klass']]
        one_hot = torch.zeros(self.n_classes)
        one_hot[label] = 1
        ema_ratio = 0.99
        self.class_ema = ema_ratio * self.class_ema + (1 - ema_ratio) * one_hot
        self.time_records[label].append(sample['time'])

        self.update_memory(sample)
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def calculate_online_acc(self, cls_acc, data_time, cls_dict):
        mean = (np.arange(self.n_classes * self.repeat) / self.n_classes).reshape(-1, 10)
        cls_weight = np.exp(-0.5 * ((data_time - mean) / (self.sigma / 100)) ** 2) / (
                self.sigma / 100 * np.sqrt(2 * np.pi))
        cls_weight = cls_weight.mean(axis=0)
        cls_order = [cls_dict[cls] for cls in self.exposed_classes]
        for i in range(self.n_classes):
            if i not in cls_order:
                cls_order.append(i)
        cls_weight = cls_weight[cls_order] / np.sum(cls_weight)
        online_acc = np.sum(np.array(cls_acc) * cls_weight)

        return online_acc

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion, data_time, cls_dict)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])

        additional = eval_dict['additional']

        for info_dict in additional:
            if info_dict['name'] in self.corrected_avg_accs.keys():
                self.corrected_avg_accs[info_dict['name']].append(info_dict['avg_acc'])
            else:
                self.corrected_avg_accs[info_dict['name']] = [info_dict['avg_acc']]
            online_acc = self.calculate_online_acc(info_dict['cls_acc'], data_time, cls_dict)
            if info_dict['name'] in self.corrected_online_accs.keys():
                self.corrected_online_accs[info_dict['name']].append(online_acc)
            else:
                self.corrected_online_accs[info_dict['name']] = [online_acc]
            self.report_test(sample_num, 0, info_dict['avg_acc'], online_acc, message=info_dict[
                                                                                          'name'] + f'(KL_Div:{F.kl_div(torch.log(additional[0]["prob"]), info_dict["prob"], reduction="batchmean"):.5f})')
        return eval_dict

    def evaluation(self, test_loader, criterion, data_time=None, cls_dict=None):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        mean = (np.arange(self.n_classes * self.repeat) / self.n_classes).reshape(-1, 10)
        cls_weight = np.exp(-0.5 * ((data_time - mean) / (self.sigma / 100)) ** 2) / (
                self.sigma / 100 * np.sqrt(2 * np.pi))
        cls_weight = cls_weight.mean(axis=0)
        cls_order = [cls_dict[cls] for cls in self.exposed_classes]
        for i in range(self.n_classes):
            if i not in cls_order:
                cls_order.append(i)
        cls_weight = cls_weight[cls_order] / np.sum(cls_weight)
        cls_weight = torch.Tensor(cls_weight)

        additional = []
        additional.append(
            {'name': 'Oracle', 'prob': cls_weight, 'correct': 0.0, 'correct_l': torch.zeros(self.n_classes)})
        additional.append(
            {'name': 'EMA', 'prob': self.class_ema, 'correct': 0.0, 'correct_l': torch.zeros(self.n_classes)})

        kd_models = [
            (0.1, 'gaussian'),
            (0.1, 'tophat'),
            (0.1, 'epanechnikov'),
            (0.1, 'exponential')
        ]

        for j, (bw, ker) in enumerate(kd_models):
            weights = torch.zeros(self.n_classes)
            for ii in range(self.n_classes):
                if len(self.time_records[ii]) > 0:
                    kd_model = KernelDensity(bandwidth=bw, kernel=ker)
                    kd_model.fit(np.array(self.time_records[ii]).reshape(-1, 1))
                    score = kd_model.score(np.array(data_time).reshape(1, 1))
                    prob = np.exp(score)
                    weights[ii] = prob * len(self.time_records[ii])
                else:
                    weights[ii] = 0
            additional.append({'name': f'KD_{ker}', 'prob': weights / weights.sum(), 'correct': 0.0,
                               'correct_l': torch.zeros(self.n_classes)})

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)

                y = y.to(self.device)
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

                logit = logit.cpu()
                y = y.cpu()

                for info_dict in additional:
                    new_logit = info_dict['prob'][:logit.size(1)] * logit
                    new_pred = torch.argmax(new_logit, dim=-1)
                    info_dict['correct'] += torch.sum(new_pred == y).item()
                    _, new_cxcnt = self._interpret_pred(y, new_pred)
                    info_dict['correct_l'] += new_cxcnt.detach().cpu()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        for info_dict in additional:
            info_dict['avg_acc'] = info_dict['correct'] / total_num_data
            info_dict['cls_acc'] = (info_dict['correct_l'] / (num_data_l + 1e-5)).numpy().tolist()

        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "additional": additional}

        return ret

    def report_test(self, sample_num, avg_loss, avg_acc, online_acc, message='Base'):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        writer.add_scalar(f"test/online_acc", online_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | {message} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | online_acc {online_acc:.4f} "
        )


class Ours_v3(Ours_v2):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.valid_list = []
        self.valid_size = round(self.memory_size * 0.1)
        self.memory_size = self.memory_size - self.valid_size
        self.val_per_cls = self.valid_size
        self.val_full = False
        self.abc_layer = None

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        label = self.memory.cls_dict[sample['klass']]
        one_hot = torch.zeros(self.n_classes)
        one_hot[label] = 1
        ema_ratio = 0.99
        self.class_ema = ema_ratio * self.class_ema + (1 - ema_ratio) * one_hot
        self.time_records[label].append(sample['time'])

        use_sample = self.online_valid_update(sample)
        self.num_updates += self.online_iter
        if use_sample:
            self.update_memory(sample)

        if self.num_updates >= 1 and len(self.memory) > 0:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        if self.num_learned_class > 1:
            self.online_reduce_valid(self.num_learned_class)

    def online_reduce_valid(self, num_learned_class):
        self.val_per_cls = self.valid_size // num_learned_class
        val_df = pd.DataFrame(self.valid_list)
        valid_list = []
        for klass in val_df["klass"].unique():
            class_val = val_df[val_df.klass == klass]
            if len(class_val) > self.val_per_cls:
                new_class_val = class_val.sample(n=self.val_per_cls)
            else:
                new_class_val = class_val
            valid_list += new_class_val.to_dict(orient="records")
        self.valid_list = valid_list
        self.val_full = False

    def online_valid_update(self, sample):
        val_df = pd.DataFrame(self.valid_list, columns=['klass', 'file_name', 'label'])
        if not self.val_full:
            if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls:
                self.valid_list.append(sample)
                if len(self.valid_list) == self.val_per_cls * self.num_learned_class:
                    self.val_full = True
                use_sample = False
            else:
                use_sample = True
        else:
            use_sample = True
        return use_sample

    def evaluation(self, test_loader, criterion, data_time=None, cls_dict=None):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        mean = (np.arange(self.n_classes * self.repeat) / self.n_classes).reshape(-1, 10)
        cls_weight = np.exp(-0.5 * ((data_time - mean) / (self.sigma / 100)) ** 2) / (
                self.sigma / 100 * np.sqrt(2 * np.pi))
        cls_weight = cls_weight.mean(axis=0)
        cls_order = [cls_dict[cls] for cls in self.exposed_classes]
        for i in range(self.n_classes):
            if i not in cls_order:
                cls_order.append(i)
        cls_weight = cls_weight[cls_order] / np.sum(cls_weight)
        cls_weight = torch.Tensor(cls_weight)

        additional = []
        additional.append({'name': 'Oracle', 'prob': cls_weight, 'weight': self.abc(cls_weight), 'correct': 0.0,
                           'correct_l': torch.zeros(self.n_classes)})
        additional.append({'name': 'EMA', 'prob': self.class_ema, 'weight': self.abc(self.class_ema), 'correct': 0.0,
                           'correct_l': torch.zeros(self.n_classes)})

        kd_models = [
            (0.1, 'gaussian'),
            (0.1, 'tophat'),
            (0.1, 'epanechnikov'),
            (0.1, 'exponential')
        ]

        for j, (bw, ker) in enumerate(kd_models):
            weights = torch.zeros(self.n_classes)
            for ii in range(self.n_classes):
                if len(self.time_records[ii]) > 0:
                    kd_model = KernelDensity(bandwidth=bw, kernel=ker)
                    kd_model.fit(np.array(self.time_records[ii]).reshape(-1, 1))
                    score = kd_model.score(np.array(data_time).reshape(1, 1))
                    prob = np.exp(score)
                    weights[ii] = prob * len(self.time_records[ii])
                else:
                    weights[ii] = 0
            additional.append(
                {'name': f'KD_{ker}', 'prob': weights / weights.sum(), 'weight': self.abc(weights / weights.sum()),
                 'correct': 0.0, 'correct_l': torch.zeros(self.n_classes)})

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)

                y = y.to(self.device)
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

                logit = logit.cpu()
                y = y.cpu()

                for info_dict in additional:
                    new_logit = info_dict['weight'][:logit.size(1)] * logit
                    new_pred = torch.argmax(new_logit, dim=-1)
                    info_dict['correct'] += torch.sum(new_pred == y).item()
                    _, new_cxcnt = self._interpret_pred(y, new_pred)
                    info_dict['correct_l'] += new_cxcnt.detach().cpu()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        for info_dict in additional:
            info_dict['avg_acc'] = info_dict['correct'] / total_num_data
            info_dict['cls_acc'] = (info_dict['correct_l'] / (num_data_l + 1e-5)).numpy().tolist()

        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "additional": additional}

        return ret

    def abc(self, weight, n_iter=256, batch_size=100):
        abc_layer = torch.ones(self.num_learned_class, requires_grad=True, device=self.device)
        if self.val_full:
            val_df = pd.DataFrame(self.valid_list)
            val_dataset = ImageDataset(val_df, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, preload=True)
            bias_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
            criterion = nn.CrossEntropyLoss(reduction='none')
            optimizer = torch.optim.Adam(params=[abc_layer], lr=0.001)
            self.model.eval()
            model_out = []
            xlabels = []
            for i, data in enumerate(bias_loader):
                x = data["image"]
                xlabel = data["label"]
                x = x.to(self.device)
                xlabel = xlabel.to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                model_out.append(out.detach().cpu())
                xlabels.append(xlabel.detach().cpu())
            for iteration in range(n_iter):
                total_loss = 0.0
                for i, out in enumerate(model_out):
                    logit = abc_layer * out.to(self.device)
                    xlabel = xlabels[i]
                    losses = criterion(logit, xlabel.to(self.device))
                    losses *= weight[xlabel].cuda()
                    loss = losses.sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            print(abc_layer)
        return abc_layer.cpu()


class Ours_v4(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory = DistillationMemory(self.dataset, self.train_transform, self.exposed_classes,
                                         test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                         transform_on_gpu=self.gpu_transform)

    def update_memory(self, sample):
        self.model.eval()
        sample_dataset = StreamDataset([sample], dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform)
        data = sample_dataset.get_data()
        x = data['image'].to(self.device)
        logit = self.model(x)[0]
        self.balanced_replace_memory(sample, logit)

    def balanced_replace_memory(self, sample, logit=None):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
            self.memory.save_logit(logit, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_logit(logit)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=1 / 3):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            y2 = None
            mask = None
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
                y2 = memory_data['logit'].to(self.device)
                mask = memory_data['logit_mask'].to(self.device)
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            if i == iterations - 1:
                logit, loss = self.model_forward(x, y, y2, mask, memory_batch_size // 2, beta, use_cutmix=False)
            else:
                logit, loss = self.model_forward(x, y, y2, mask, memory_batch_size // 2, beta, use_cutmix=True)

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

        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y, y2=None, mask=None, distill_size=0, beta=1 / 3, use_cutmix=True):
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            mask = mask[-distill_size:]
            if do_cutmix:
                x[:-distill_size], labels_a, labels_b, lam = cutmix_data(x=x[:-distill_size], y=y, alpha=1.0)
                loss = 0
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit = self.model(x)
                        cls_logit = logit[:-distill_size]
                        loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit,
                                                                                                       labels_b)
                        distill_logit = logit[-distill_size:]
                        loss += beta * (mask * (y2 - distill_logit) ** 2).sum(dim=1).mean()
                else:
                    logit = self.model(x)
                    cls_logit = logit[:-distill_size]
                    loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit, labels_b)
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).sum(dim=1).mean()
            else:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit = self.model(x)
                        cls_logit = logit[:-distill_size]
                        loss = self.criterion(cls_logit, y)
                        distill_logit = logit[-distill_size:]
                        loss += beta * (mask * (y2 - distill_logit) ** 2).sum(dim=1).mean()
                else:
                    logit = self.model(x)
                    cls_logit = logit[:-distill_size]
                    loss = self.criterion(cls_logit, y)
                    distill_logit = logit[-distill_size:]
                    loss += beta * (mask * (y2 - distill_logit) ** 2).sum(dim=1).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)


class Ours_v5(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory = DistillationMemory(self.dataset, self.train_transform, self.exposed_classes,
                                         test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                         transform_on_gpu=self.gpu_transform, use_feature=True)

    def update_memory(self, sample):
        self.model.eval()
        sample_dataset = StreamDataset([sample], dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform)
        data = sample_dataset.get_data()
        x = data['image'].to(self.device)
        with torch.no_grad():
            logit, feature = self.model(x, get_feature=True)
            self.balanced_replace_memory(sample, copy.deepcopy(feature.detach()[0]))

    def balanced_replace_memory(self, sample, logit=None):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
            self.memory.save_logit(logit, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_logit(logit)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=0.1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            y2 = None
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
                y2 = memory_data['logit'].to(self.device)
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, y2, memory_batch_size // 2, beta, use_cutmix=True)

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

        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y, y2=None, distill_size=0, beta=0.1, use_cutmix=True):
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            if do_cutmix:
                x[:-distill_size], labels_a, labels_b, lam = cutmix_data(x=x[:-distill_size], y=y, alpha=1.0)
                loss = 0
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit, feature = self.model(x, get_feature=True)
                        cls_logit = logit[:-distill_size]
                        loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit,
                                                                                                       labels_b)
                        distill_feature = feature[-distill_size:]
                        loss += beta * ((y2 - distill_feature) ** 2).sum(dim=1).mean()
                else:
                    logit, feature = self.model(x, get_feature=True)
                    cls_logit = logit[:-distill_size]
                    loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit, labels_b)
                    distill_feature = feature[-distill_size:]
                    loss += beta * ((y2 - distill_feature) ** 2).sum(dim=1).mean()
            else:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit, feature = self.model(x, get_feature=True)
                        cls_logit = logit[:-distill_size]
                        loss = self.criterion(cls_logit, y)
                        distill_feature = feature[-distill_size:]
                        loss += beta * ((y2 - distill_feature) ** 2).sum(dim=1).mean()
                else:
                    logit, feature = self.model(x, get_feature=True)
                    cls_logit = logit[:-distill_size]
                    loss = self.criterion(cls_logit, y)
                    distill_feature = feature[-distill_size:]
                    loss += beta * ((y2 - distill_feature) ** 2).sum(dim=1).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)


class Ours_v6(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory = DistillationMemory(self.dataset, self.train_transform, self.exposed_classes,
                                         test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                         transform_on_gpu=self.gpu_transform, use_feature=True, use_logit=True)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.trans_feature = kwargs['trans_feature']
        self.encounter_dict = {}

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        # torch.nn.init.xavier_normal(self.model.fc.weight)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if len(self.memory) > 0:
            features = torch.stack(self.memory.features)
            logits = self.model.fc(features)
            for i, logit in enumerate(logits):
                self.memory.logits[i] = logit
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def update_memory(self, sample):
        self.model.eval()
        if self.trans_feature:
            transform = self.train_transform
        else:
            transform = self.test_transform
        sample_dataset = StreamDataset([sample], dataset=self.dataset, transform=transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform and self.trans_feature)
        data = sample_dataset.get_data()
        x = data['image'].to(self.device)
        with torch.no_grad():
            logit, feature = self.model(x, get_feature=True)
            self.balanced_replace_memory(sample, copy.deepcopy(feature.detach()[0]), logit.detach()[0])

    def balanced_replace_memory(self, sample, feature=None, logit=None):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
            self.memory.save_feature(feature, idx_to_replace)
            self.memory.save_logit(logit, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_feature(feature)
            self.memory.save_logit(logit)

    def balanced_uniform_memory(self, sample, feature=None, logit=None):
        if sample['klass'] in self.encounter_dict:
            self.encounter_dict[sample['klass']] += 1
        else:
            self.encounter_dict[sample['klass']] = 1
        if len(self.memory.images) >= self.memory_size:
            if np.random.rand() < self.memory_size // self.num_learned_class / self.encounter_dict[sample['klass']]:
                label_frequency = copy.deepcopy(self.memory.cls_count)
                label_frequency[self.exposed_classes.index(sample['klass'])] += 1
                cls_to_replace = np.argmax(label_frequency)
                idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
                self.memory.replace_sample(sample, idx_to_replace)
                self.memory.save_feature(feature, idx_to_replace)
                self.memory.save_logit(logit, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.memory.save_feature(feature)
            self.memory.save_logit(logit)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            y2 = None
            y3 = None
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
                y2 = memory_data['feature'].to(self.device)
                y3 = memory_data['logit'].to(self.device)
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, y2, y3, memory_batch_size // 2, self.beta, use_cutmix=True)

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

        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y, y2=None, y3=None, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if self.weighted:
                distill_weight = self.get_class_weight()
                weight = distill_weight[y].to(self.device)
            else:
                weight = torch.ones([y.size(0), y2.size(1)]).to(self.device) / y2.size(1)
            if self.pred_based:
                prob = torch.softmax(y3, dim=-1)[torch.arange(y.size(0)), y]
            else:
                prob = 0.5 * torch.ones(y.size(0)).to(self.device)
            if do_cutmix:
                x, labels_a, labels_b, feature_a, feature_b, prob_a, prob_b, weight_a, weight_b, lam = cutmix_feature(
                    x=x, y=y, feature=y2, prob=prob, weight=weight, alpha=1.0)
                loss = 0
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    loss += lam * (criterion(logit, labels_a) * (1 - prob_a)).mean() + (1 - lam) * (
                            criterion(logit, labels_b) * (1 - prob_b)).mean()
                    loss += beta * (
                            lam * (prob_a * (weight_a * (feature_a - feature) ** 2).sum(dim=1)).mean() + (
                            1 - lam) * (prob_b * (weight_b * (feature_b - feature) ** 2).sum(
                        dim=1)).mean())
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    loss = (criterion(logit, y) * (1 - prob)).mean()
                    loss += beta * (prob * (weight * (y2 - feature) ** 2).sum(dim=1)).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def model_forward_v2(self, x, y, y2=None, y3=None, distill_size=0, beta=10.0, use_cutmix=True):
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            y = y[:-distill_size]
            y2 = y2[-distill_size:]
            distill_cls = y[-distill_size:]
            if self.weighted:
                distill_weight = self.get_class_weight()
                weight = distill_weight[distill_cls].to(self.device)
            else:
                weight = 1 / y2.size(1)
            if do_cutmix:
                x[:-distill_size], labels_a, labels_b, lam = cutmix_data(x=x[:-distill_size], y=y, alpha=1.0)
                loss = 0
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit, feature = self.model(x, get_feature=True)
                        cls_logit = logit[:-distill_size]
                        loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit,
                                                                                                       labels_b)
                        distill_feature = feature[-distill_size:]
                        loss += beta * (weight * (y2 - distill_feature) ** 2).sum(dim=1).mean()
                else:
                    logit, feature = self.model(x, get_feature=True)
                    cls_logit = logit[:-distill_size]
                    loss += lam * self.criterion(cls_logit, labels_a) + (1 - lam) * self.criterion(cls_logit, labels_b)
                    distill_feature = feature[-distill_size:]
                    loss += beta * (weight * (y2 - distill_feature) ** 2).sum(dim=1).mean()
            else:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit, feature = self.model(x, get_feature=True)
                        cls_logit = logit[:-distill_size]
                        loss = self.criterion(cls_logit, y)
                        distill_feature = feature[-distill_size:]
                        loss += beta * (weight * (y2 - distill_feature) ** 2).sum(dim=1).mean()
                else:
                    logit, feature = self.model(x, get_feature=True)
                    cls_logit = logit[:-distill_size]
                    loss = self.criterion(cls_logit, y)
                    distill_feature = feature[-distill_size:]
                    loss += beta * (weight * (y2 - distill_feature) ** 2).sum(dim=1).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight


class Ours_v7(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.stat_ema = 0.9
        self.distill_mean = torch.zeros(1).to(self.device)
        self.distill_var = torch.ones(1).to(self.device)
        self.cls_mean = torch.zeros(1).to(self.device)
        self.cls_var = torch.ones(1).to(self.device)
        self.ema_model = copy.deepcopy(self.model)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True)

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
            self.update_ema_model()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        # print(self.cls_mean.item(), self.cls_var.item(), self.distill_mean.item(), self.distill_var.item())
        if distill_size > 0:
            if self.weighted:
                distill_weight = self.get_class_weight().to(self.device)
            else:
                distill_weight = torch.ones([self.num_learned_class, self.model.fc.in_features]).to(
                    self.device) / self.model.fc.in_features
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = (lam * full_prob[torch.arange(y.size(0)), labels_a] + (1 - lam) * full_prob[
                                torch.arange(y.size(0)), labels_b]) / (1 - 2 * lam * (1 - lam))
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                            # samelabel = labels_a == labels_b
                            # prob = lam * criterion(logit2, labels_a) + (1 - lam) * criterion(logit2, labels_b)
                            # max = lam * np.log(lam) + (1 - lam) * np.log(1 - lam)
                            # if self.num_learned_class > 1:
                            #     randloss = np.log(self.num_learned_class)
                            #     scale = torch.ones_like(prob) * randloss
                            #     scale[samelabel] -= max
                            #     prob = F.relu((randloss - prob) / scale)
                            # else:
                            #     prob = 0
                        else:
                            prob = 0.5
                        weight = lam * distill_weight[labels_a] + (1 - lam) * distill_weight[labels_b]
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)

                    self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                    self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                    if y.size(0) > 1:
                        self.cls_var += (1 - self.stat_ema) * (
                                cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                        self.distill_var += (1 - self.stat_ema) * (
                                distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                        distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                            self.distill_var + 1e-8) + self.cls_mean

                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                        if self.pred_based:
                            # full_prob = torch.softmax(logit2, dim=-1)
                            # prob = full_prob[torch.arange(y.size(0)), y]
                            # if self.num_learned_class > 1:
                            #     prob = F.relu((prob * self.num_learned_class - 1)/(self.num_learned_class - 1))
                            # else:
                            #     prob = 0
                            prob = criterion(logit2, y)
                            if self.num_learned_class > 1:
                                randloss = np.log(self.num_learned_class)
                                prob = F.relu((randloss - prob) / randloss)
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                        weight = distill_weight[y]
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                    self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                    if y.size(0) > 1:
                        self.cls_var += (1 - self.stat_ema) * (
                                cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                        self.distill_var += (1 - self.stat_ema) * (
                                distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                        distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                            self.distill_var + 1e-8) + self.cls_mean
                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight


class Ours_v8(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.label_only = True
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.cls_dim = kwargs['cls_dim']
        self.loss_ema = kwargs['loss_ema']
        self.ema_model = copy.deepcopy(self.model)
        # self.cls_distill_weight = torch.Tensor([]).to(self.device)

    def cls_out(self, out):
        out = out.reshape(out.size(0), -1, self.num_learned_class)
        return out.sum(dim=2)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features,
                                  self.num_learned_class * (self.num_learned_class + 1) // 2).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:(self.num_learned_class - 1) * self.num_learned_class // 2] = prev_weight
        sdict = copy.deepcopy(self.optimizer.state_dict())
        fc_params = sdict['param_groups'][1]['params']
        if len(sdict['state']) > 0:
            fc_weight_state = sdict['state'][fc_params[0]]
            fc_bias_state = sdict['state'][fc_params[1]]
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        if len(sdict['state']) > 0:
            if 'adam' in self.opt_name:
                fc_weight = self.optimizer.param_groups[1]['params'][0]
                fc_bias = self.optimizer.param_groups[1]['params'][1]
                self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                        torch.zeros([self.num_learned_class,
                                                                                     fc_weight_state['exp_avg'].size(
                                                                                         dim=1)]).to(self.device)],
                                                                       dim=0)
                self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                           torch.zeros(
                                                                               [self.num_learned_class, fc_weight_state[
                                                                                   'exp_avg_sq'].size(dim=1)]).to(
                                                                               self.device)], dim=0)
                self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                      torch.zeros(self.num_learned_class).to(
                                                                          self.device)], dim=0)
                self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                         torch.zeros(self.num_learned_class).to(
                                                                             self.device)], dim=0)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        ema_prev_weight = copy.deepcopy(self.ema_model.fc.weight.data)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.ema_model.fc.weight[:(self.num_learned_class - 1) * self.num_learned_class // 2] = ema_prev_weight
                self.ema_model.fc.weight[(self.num_learned_class - 1) * self.num_learned_class // 2:] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        # self.cls_distill_weight = torch.cat([self.cls_distill_weight, torch.Tensor([0]).to(self.device)])

    def get_matrix_output(self, out):
        mat_out = torch.zeros(out.size(0), self.num_learned_class, self.num_learned_class, dtype=out.dtype).to(
            self.device)
        i, j = torch.tril_indices(self.num_learned_class, self.num_learned_class)
        mat_out[:, i, j] = out
        mat_out[:, j, i] = out
        return mat_out

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True)

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
            self.update_ema_model()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.get_matrix_output(self.model(x))
                    cls_loss = lam * criterion(self.cls_out(logit), labels_a) + (1 - lam) * criterion(
                        self.cls_out(logit), labels_b)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2 = self.get_matrix_output(self.ema_model(x))
                        # cls_loss2 = lam * criterion(self.cls_out(logit2), labels_a) + (1 - lam) * criterion(self.cls_out(logit2), labels_b)
                        # for i in range(y.size(0)):
                        #     self.cls_distill_weight[labels_a[i]] += (1 - self.loss_ema) * lam * (1 - cls_loss2[i] - self.cls_distill_weight[labels_a[i]])
                        #     self.cls_distill_weight[labels_b[i]] += (1 - self.loss_ema) * (1 - lam) * (1 - cls_loss2[i] - self.cls_distill_weight[labels_b[i]])
                        # if self.pred_based:
                        #     full_prob = torch.softmax(self.cls_out(logit2), dim=-1)
                        #     prob = lam * full_prob[torch.arange(y.size(0)), labels_a] + (1 - lam) * full_prob[torch.arange(y.size(0)), labels_b]
                        # else:
                        #     prob = 0.5
                    distill_loss = self.classwise_distillation_loss(logit, logit2, y)
                    loss = (cls_loss + self.beta * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit = self.get_matrix_output(self.model(x))
                    cls_loss = criterion(self.cls_out(logit), y)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2 = self.get_matrix_output(self.ema_model(x))
                        cls_loss2 = criterion(self.cls_out(logit2), y)
                        # for i in range(y.size(0)):
                        #     self.cls_distill_weight[y[i]] += (1 - self.loss_ema) * (1 - cls_loss2[i] - self.cls_distill_weight[y[i]])
                        # if self.pred_based:
                        #     full_prob = torch.softmax(self.cls_out(logit2), dim=-1)
                        #     prob = full_prob[torch.arange(y.size(0)), y]
                        # else:
                        #     prob = 0.5
                    distill_loss = self.classwise_distillation_loss(logit, logit2, y)
                    loss = (cls_loss + self.beta * distill_loss).mean()
            return self.cls_out(logit), loss
        else:
            return super().model_forward(x, y)

    def classwise_distillation_loss(self, old_logit, new_logit, y):
        T = 2
        old_softmax = torch.softmax(old_logit.reshape(old_logit.size(0), -1, self.num_learned_class) / T, dim=-1)
        new_log_softmax = torch.log_softmax(new_logit.reshape(new_logit.size(0), -1, self.num_learned_class) / T,
                                            dim=-1)
        if self.label_only:
            old_softmax = old_softmax[torch.arange(old_softmax.size(0)), y, :].unsqueeze(1)
            new_log_softmax = new_log_softmax[torch.arange(new_log_softmax.size(0)), y, :].unsqueeze(1)
            cls_weight = 1
        else:
            cls_weight = F.relu(self.cls_distill_weight)
        # print(cls_weight)
        distillation_loss = -(((old_softmax * new_log_softmax).sum(dim=2)) * cls_weight).mean(dim=1)
        return distillation_loss

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
                logit = self.cls_out(self.get_matrix_output(self.model(x)))

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


class Ours_v9(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.ema_model = copy.deepcopy(self.model)
        self.feature_only = kwargs['feature_only']
        self.stat_ema = 0.9
        self.distill_mean = torch.zeros(1).to(self.device)
        self.distill_var = torch.ones(1).to(self.device)
        self.cls_mean = torch.zeros(1).to(self.device)
        self.cls_var = torch.ones(1).to(self.device)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, sample_num, n_worker):
        super().online_step(sample, sample_num, n_worker)
        self.update_ema_model()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True)

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

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

        # model_modules = OrderedDict(self.model.named_modules())
        # ema_modules = OrderedDict(self.ema_model.named_modules())
        # for name, layer in model_modules.items():
        #     if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
        #         # ema_modules[name].weight = copy.deepcopy(model_modules[name].weight)
        #         # ema_modules[name].bias = copy.deepcopy(model_modules[name].bias)
        #         ema_modules[name].running_mean = copy.deepcopy(model_modules[name].running_mean)
        #         ema_modules[name].running_var = copy.deepcopy(model_modules[name].running_var)

    def ema_bn_set(self):
        model_modules = OrderedDict(self.model.named_modules())
        ema_modules = OrderedDict(self.ema_model.named_modules())
        for name, layer in model_modules.items():
            if isinstance(layer, StatTrack):
                # ema_modules[name].weight = copy.deepcopy(model_modules[name].weight)
                # ema_modules[name].bias = copy.deepcopy(model_modules[name].bias)
                batch_mean = model_modules[name].batch_mean.detach().float()
                batch_var = model_modules[name].batch_var.detach().float()
            if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
                ema_modules[name].running_mean = batch_mean
                ema_modules[name].running_var = batch_var

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if np.random.rand(1) < 0.5:
            self.ema_model.train()
        else:
            self.ema_model.eval()
        # self.ema_model.eval()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    # self.ema_model.train()
                    # self.ema_bn_set()
                    self.ema_model.zero_grad()
                    logit2, features2 = self.ema_model(x, get_features=True)
                    if self.weighted:
                        cls_loss_all = (lam * criterion(logit2, labels_a) + (1 - lam) * criterion(logit2, labels_b))
                        cls_loss_2 = cls_loss_all.mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        if self.feature_only:
                            feature = features[-1].flatten(start_dim=1)
                            feature2 = features2[-1].flatten(start_dim=1).detach()
                            weight = torch.abs(torch.flatten(grads[-1], start_dim=1))
                            # weight = torch.exp(weight*(1-cls_loss_all.detach().unsqueeze(1)))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                        else:
                            feature = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features[:-1]], dim=1)
                            feature2 = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features2[:-1]],
                                                 dim=1).detach()

                            weight = torch.abs(
                                torch.cat([torch.flatten(tensor, start_dim=1) for tensor in grads[:-1]], dim=1))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                    else:
                        feature = features[-1].flatten(start_dim=1)
                        feature2 = features2[-1].flatten(start_dim=1).detach()
                        weight = torch.ones(feature2.shape).to(self.device) / self.model.fc.in_features

                    with torch.no_grad():
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = lam * full_prob[torch.arange(y.size(0)), labels_a] + (1 - lam) * full_prob[
                                torch.arange(y.size(0)), labels_b]
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    if self.pred_based:
                        self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                        self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                        if y.size(0) > 1:
                            self.cls_var += (1 - self.stat_ema) * (
                                    cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                            self.distill_var += (1 - self.stat_ema) * (
                                    distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                            distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                                self.distill_var + 1e-8) + self.cls_mean

                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True)
                    cls_loss = criterion(logit, y)
                    # self.ema_model.eval()
                    # self.ema_bn_set()
                    self.ema_model.zero_grad()
                    logit2, features2 = self.ema_model(x, get_features=True)
                    if self.weighted:
                        cls_loss_all = criterion(logit2, y)
                        cls_loss_2 = cls_loss_all.mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        if self.feature_only:
                            feature = features[-1].flatten(start_dim=1)
                            feature2 = features2[-1].flatten(start_dim=1).detach()
                            weight = torch.abs(torch.flatten(grads[-1], start_dim=1))
                            # weight = torch.exp(weight * (1 - cls_loss_all.detach().unsqueeze(1)))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                        else:
                            feature = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features[:-1]], dim=1)
                            feature2 = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features2[:-1]],
                                                 dim=1).detach()
                            weight = torch.abs(
                                torch.cat([torch.flatten(tensor, start_dim=1) for tensor in grads[:-1]], dim=1))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                    else:
                        feature = features[-1].flatten(start_dim=1)
                        feature2 = features2[-1].flatten(start_dim=1).detach()
                        weight = torch.ones(feature2.shape).to(self.device) / self.model.fc.in_features
                    with torch.no_grad():
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = full_prob[torch.arange(y.size(0)), y]
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    if self.pred_based:
                        self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                        self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                        if y.size(0) > 1:
                            self.cls_var += (1 - self.stat_ema) * (
                                    cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                            self.distill_var += (1 - self.stat_ema) * (
                                    distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                            distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                                self.distill_var + 1e-8) + self.cls_mean
                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight


class Ours_v10(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.val_set = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                     test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                     transform_on_gpu=self.gpu_transform, save_test='gpu')
        self.val_size = round(self.memory_size * 0.1)
        self.memory_size = self.memory_size - self.val_size

        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, keep_history=True)
        self.imp_update_period = kwargs['imp_update_period']
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'

        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.update_memory(sample)
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            if len(self.memory) > 0:
                train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                          iterations=int(self.num_updates), stream_batch_size=0)
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
                self.update_schedule()

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.val_set.add_new_class(self.exposed_classes)

    def update_memory(self, sample):
        mem_sample = self.update_valset(sample)
        if mem_sample is not None:
            self.samplewise_importance_memory(mem_sample)

    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

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
            self.samplewise_loss_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def update_valset(self, sample):
        label_frequency = copy.deepcopy(self.val_set.cls_count)
        memory_label_frequency = copy.deepcopy(self.memory.cls_count)
        if label_frequency[self.exposed_classes.index(sample['klass'])] > memory_label_frequency[
            self.exposed_classes.index(sample['klass'])]:
            return sample
        else:
            if len(self.val_set.images) >= self.val_size:
                label_frequency[self.exposed_classes.index(sample['klass'])] += 1
                cls_to_replace = np.random.choice(
                    np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
                idx_to_replace = np.random.choice(self.val_set.cls_idx[cls_to_replace])
                replaced_sample = copy.deepcopy(self.val_set.datalist[idx_to_replace])
                self.val_set.replace_sample(sample, idx_to_replace)
                self.dropped_idx.append(idx_to_replace)
                self.memory_dropped_idx.append(idx_to_replace)
                return replaced_sample
            else:
                self.val_set.replace_sample(sample)
                self.memory_dropped_idx.append(len(self.val_set) - 1)
                self.dropped_idx.append(len(self.val_set) - 1)
                return None

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.val_set) > 0:
                self.model.eval()
                with torch.no_grad():
                    x = self.val_set.device_img
                    y = torch.LongTensor(self.val_set.labels)
                    y = y.to(self.device)
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit = torch.cat(
                                [self.model(
                                    torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                                    for i in range(-(-len(x) // batchsize))], dim=0)

                    else:
                        logit = torch.cat(
                            [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                             for i in range(-(-len(x) // batchsize))], dim=0)

                    loss = F.cross_entropy(logit, y, reduction='none').cpu().numpy()
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio,
                                                dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss

    def samplewise_importance_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            score = self.memory.others_loss_decrease[cand_idx]
            idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            mask = np.ones(len(self.loss), bool)
            mask[self.dropped_idx] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(
                            np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(
                            np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.low_lr_loss) > min_iter:
                            del self.low_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = True
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.high_lr
                        param_group["initial_lr"] = self.high_lr
                self.dropped_idx = []
                if len(self.high_lr_loss) == len(self.low_lr_loss) and len(self.high_lr_loss) >= min_iter:
                    stat, pvalue = ttest_ind(self.low_lr_loss, self.high_lr_loss, equal_var=False,
                                             alternative='greater')
                    print(pvalue)
                    if pvalue < significance:
                        self.high_lr = self.low_lr
                        self.low_lr *= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
                    elif pvalue > 1 - significance:
                        self.low_lr = self.high_lr
                        self.high_lr /= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr


class Ours_test(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.stat_ema = 0.9
        self.ema_model = copy.deepcopy(self.model)
        self.grad_results = {}
        self.backprop_results = {}

    def online_step(self, sample, sample_num, n_worker):
        super().online_step(sample, sample_num, n_worker)
        self.update_ema_model()
        if sample_num > 100 and sample_num % 10 == 0:
            self.backprop_test(sample_num)

    def backprop_test(self, sample_num, last_only=False):
        data = self.memory.get_batch(16)
        x = data['image']
        y = data['label']
        with torch.cuda.amp.autocast(self.use_amp):
            logit, features = self.model(x, get_features=True)
            self.model.train()
            self.ema_model.eval()
            # self.ema_bn_set()
            self.model.zero_grad()
            with torch.no_grad():
                logit2, features2 = self.ema_model(x, get_features=True)
            for tensor in features:
                tensor.retain_grad()

            if last_only:
                feature = features[-1].flatten(start_dim=1)
                feature2 = features2[-1].flatten(start_dim=1).detach()
            else:
                feature = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features[:-1]], dim=1)
                feature2 = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features2[:-1]],
                                     dim=1).detach()
            distill_loss = ((feature2 - feature) ** 2).mean()
            distill_loss.backward()
            cos_sim = []
            for i, tensor in enumerate(features):
                grad = copy.deepcopy(tensor.grad)
                if grad is None:
                    continue
                diff = (tensor - features2[i])
                cos_sim.append(F.cosine_similarity(grad, diff).mean().item())
        self.backprop_results[sample_num] = cos_sim
        self.model.zero_grad()
        np.save('backprop_all.npy', self.backprop_results)

    def grad_test(self, sample_num):
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
        gpu_test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
                                                 transforms.Normalize(mean, std)])
        testset_grad = []
        memory_grad = []

        self.model.eval()
        self.ema_model.eval()

        test_list = get_test_datalist(self.dataset)
        test_df = pd.DataFrame(test_list)
        for klass in self.exposed_classes:
            exp_test_df = test_df[test_df['klass'].isin([klass])]
            test_dataset = ImageDataset(
                exp_test_df,
                dataset=self.dataset,
                transform=self.test_transform,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
            test_loader = DataLoader(
                test_dataset,
                shuffle=True,
                batch_size=1000,
                num_workers=1,
            )
            data = next(iter(test_loader))
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)

            y = y.to(self.device)
            logit = self.model(x)

            self.model.zero_grad()
            loss = self.criterion(logit, y)
            loss.backward()
            grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad.append(copy.deepcopy(p.grad.view(-1)))
            grad = torch.cat(grad)
            testset_grad.append(grad)
        mean_grad = sum(testset_grad) / len(testset_grad)

        for sample in self.memory.datalist:
            sample_dict = {'klass': sample['klass']}
            sample_dataset = StreamDataset([sample], dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=True)
            stream_data = sample_dataset.get_data()
            x = stream_data['image'].to(self.device)
            y = stream_data['label'].to(self.device)
            sample_dict['label'] = y
            logit, feature = self.model(x, get_feature=True)
            loss = self.criterion(logit, y)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad.append(copy.deepcopy(p.grad.view(-1)))
            grad = torch.cat(grad)
            cls_sim = [F.cosine_similarity(grad, cls_grad, dim=0).item() for cls_grad in testset_grad]
            mean_sim = F.cosine_similarity(grad, mean_grad, dim=0).item()
            sample_dict['cls_grad_sim'] = cls_sim
            sample_dict['cls_grad_mean_sim'] = mean_sim

            self.ema_model.eval()
            logit2, feature2 = self.ema_model(x, get_feature=True)
            cls_loss_2 = self.criterion(logit2, y).mean()
            feature2.retain_grad()
            cls_loss_2.backward()
            feat_grad = copy.deepcopy(feature2.grad)
            weight = torch.abs(feat_grad)
            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8

            feature2 = feature2.detach()
            full_prob = torch.softmax(logit2, dim=-1)
            prob = full_prob[torch.arange(y.size(0)), y]
            sample_dict['pred_prob'] = prob

            distill_loss = ((feature2 - feature) ** 2).sum(dim=1)
            self.model.zero_grad()
            distill_loss.backward(retain_graph=True)
            grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad.append(copy.deepcopy(p.grad.view(-1)))
            grad = torch.cat(grad)
            cls_sim = [F.cosine_similarity(grad, cls_grad, dim=0).item() for cls_grad in testset_grad]
            mean_sim = F.cosine_similarity(grad, mean_grad, dim=0).item()
            sample_dict['distill_grad_sim'] = cls_sim
            sample_dict['distill_grad_mean_sim'] = mean_sim

            weighted_distill_loss = (weight * (feature2 - feature) ** 2).sum(dim=1)
            self.model.zero_grad()
            weighted_distill_loss.backward()
            grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad.append(copy.deepcopy(p.grad.view(-1)))
            grad = torch.cat(grad)
            cls_sim = [F.cosine_similarity(grad, cls_grad, dim=0).item() for cls_grad in testset_grad]
            mean_sim = F.cosine_similarity(grad, mean_grad, dim=0).item()
            sample_dict['w_distill_grad_sim'] = cls_sim
            sample_dict['w_distill_grad_mean_sim'] = mean_sim
            memory_grad.append(sample_dict)

        self.grad_results[sample_num] = memory_grad
        np.save('grads.npy', self.grad_results)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True)

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

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if self.weighted:
                distill_weight = self.get_class_weight().to(self.device)
            else:
                distill_weight = torch.ones([self.num_learned_class, self.model.fc.in_features]).to(
                    self.device) / self.model.fc.in_features
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = (lam * full_prob[torch.arange(y.size(0)), labels_a] + (1 - lam) * full_prob[
                                torch.arange(y.size(0)), labels_b]) / (1 - 2 * lam * (1 - lam))
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                        weight = lam * distill_weight[labels_a] + (1 - lam) * distill_weight[labels_b]
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)

                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    with torch.no_grad():
                        self.ema_model.eval()
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                        if self.pred_based:
                            prob = criterion(logit2, y)
                            if self.num_learned_class > 1:
                                randloss = np.log(self.num_learned_class)
                                prob = F.relu((randloss - prob) / randloss)
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                        weight = distill_weight[y]
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight


class Ours_minieval(Ours_v9):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.eval_period = kwargs['eval_period']
        self.test_datalist = get_test_datalist(self.dataset)
        self.mini_eval_size = 0.1
        self.mini_eval_period = int(0.1 * self.eval_period)
        self.mini_testset = []
        self.results_train = []
        self.time_train = []
        self.results_eval = []
        self.time_eval = []
        self.prev_test_loss = 0
        self.bn_train = False
        self.note = kwargs['note']
        self.rnd_seed = kwargs['rnd_seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}_bn'

    def construct_mini_eval_set(self, test_list):
        test_df = pd.DataFrame(test_list)
        self.mini_testset = []
        for cls_name in self.exposed_classes:
            cls_df = test_df[test_df['klass'] == cls_name]
            self.mini_testset += cls_df.sample(n=int(len(cls_df) * self.mini_eval_size)).to_dict(orient='records')

    def online_step(self, sample, sample_num, n_worker):
        super().online_step(sample, sample_num, n_worker)
        if sample_num % self.mini_eval_period == 0 and len(self.mini_testset) > 0:
            loss = self.get_test_loss(self.mini_testset)
            if self.bn_train:
                self.results_train.append(loss - self.prev_test_loss)
                self.time_train.append(sample_num)
                self.bn_train = False
            else:
                self.results_eval.append(loss - self.prev_test_loss)
                self.time_eval.append(sample_num)
                self.bn_train = True
            self.prev_test_loss = loss
            np.save(self.save_path + '_train.npy', self.results_train)
            np.save(self.save_path + '_train_time.npy', self.time_train)
            np.save(self.save_path + '_eval.npy', self.results_eval)
            np.save(self.save_path + '_eval_time.npy', self.time_eval)
        if sample_num % self.eval_period == 0:
            self.construct_mini_eval_set(self.test_datalist)
            self.prev_test_loss = self.get_test_loss(self.mini_testset)

    def get_test_loss(self, test_list, batch_size=100, n_worker=1):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=min(batch_size, len(self.mini_testset)),
            num_workers=n_worker,
        )
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                loss = self.criterion(logit, y)
                total_loss += loss.item()
        return total_loss / len(test_loader)

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if self.bn_train:
            self.ema_model.train()
        else:
            self.ema_model.eval()
        # self.ema_model.eval()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    # self.ema_model.train()
                    # self.ema_bn_set()
                    self.ema_model.zero_grad()
                    logit2, features2 = self.ema_model(x, get_features=True)
                    if self.weighted:
                        cls_loss_all = (lam * criterion(logit2, labels_a) + (1 - lam) * criterion(logit2, labels_b))
                        cls_loss_2 = cls_loss_all.mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        if self.feature_only:
                            feature = features[-1].flatten(start_dim=1)
                            feature2 = features2[-1].flatten(start_dim=1).detach()
                            weight = torch.abs(torch.flatten(grads[-1], start_dim=1))
                            # weight = torch.exp(weight*(1-cls_loss_all.detach().unsqueeze(1)))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                        else:
                            feature = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features[:-1]], dim=1)
                            feature2 = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features2[:-1]],
                                                 dim=1).detach()

                            weight = torch.abs(
                                torch.cat([torch.flatten(tensor, start_dim=1) for tensor in grads[:-1]], dim=1))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                    else:
                        feature = features[-1].flatten(start_dim=1)
                        feature2 = features2[-1].flatten(start_dim=1).detach()
                        weight = torch.ones(feature2.shape).to(self.device) / self.model.fc.in_features

                    with torch.no_grad():
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = lam * full_prob[torch.arange(y.size(0)), labels_a] + (1 - lam) * full_prob[
                                torch.arange(y.size(0)), labels_b]
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    if self.pred_based:
                        self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                        self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                        if y.size(0) > 1:
                            self.cls_var += (1 - self.stat_ema) * (
                                    cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                            self.distill_var += (1 - self.stat_ema) * (
                                    distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                            distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                                self.distill_var + 1e-8) + self.cls_mean

                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True)
                    cls_loss = criterion(logit, y)
                    # self.ema_model.eval()
                    # self.ema_bn_set()
                    self.ema_model.zero_grad()
                    logit2, features2 = self.ema_model(x, get_features=True)
                    if self.weighted:
                        cls_loss_all = criterion(logit2, y)
                        cls_loss_2 = cls_loss_all.mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        if self.feature_only:
                            feature = features[-1].flatten(start_dim=1)
                            feature2 = features2[-1].flatten(start_dim=1).detach()
                            weight = torch.abs(torch.flatten(grads[-1], start_dim=1))
                            # weight = torch.exp(weight * (1 - cls_loss_all.detach().unsqueeze(1)))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                        else:
                            feature = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features[:-1]], dim=1)
                            feature2 = torch.cat([torch.flatten(tensor, start_dim=1) for tensor in features2[:-1]],
                                                 dim=1).detach()
                            weight = torch.abs(
                                torch.cat([torch.flatten(tensor, start_dim=1) for tensor in grads[:-1]], dim=1))
                            weight /= torch.sum(weight, dim=1).reshape(-1, 1) + 1e-8
                    else:
                        feature = features[-1].flatten(start_dim=1)
                        feature2 = features2[-1].flatten(start_dim=1).detach()
                        weight = torch.ones(feature2.shape).to(self.device) / self.model.fc.in_features
                    with torch.no_grad():
                        if self.pred_based:
                            full_prob = torch.softmax(logit2, dim=-1)
                            prob = full_prob[torch.arange(y.size(0)), y]
                            if self.num_learned_class > 1:
                                prob = F.relu((prob * self.num_learned_class - 1) / (self.num_learned_class - 1))
                            else:
                                prob = 0
                        else:
                            prob = 0.5
                    distill_loss = beta * (weight * (feature2 - feature) ** 2).sum(dim=1)
                    if self.pred_based:
                        self.cls_mean += (1 - self.stat_ema) * (cls_loss.detach().mean() - self.cls_mean)
                        self.distill_mean += (1 - self.stat_ema) * (distill_loss.detach().mean() - self.distill_mean)
                        if y.size(0) > 1:
                            self.cls_var += (1 - self.stat_ema) * (
                                    cls_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.cls_var)
                            self.distill_var += (1 - self.stat_ema) * (
                                    distill_loss.detach().var() * y.size(0) / (y.size(0) - 1) - self.distill_var)
                            distill_loss = (distill_loss - self.distill_mean) * torch.sqrt(self.cls_var) / torch.sqrt(
                                self.distill_var + 1e-8) + self.cls_mean
                    loss = ((1 - prob) * cls_loss + prob * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)


class Ours_v11(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.ema_ratio = kwargs['ema_ratio']
        self.ema_model = copy.deepcopy(self.model)
        self.reduce_bpdepth = kwargs['reduce_bpdepth']
        self.importance_measure = kwargs['importance']
        self.importance_ema = kwargs['imp_ema']
        self.importance = None
        self.norm_importance = None
        self.norm_loss = kwargs['norm_loss']
        self.loss_ratio = kwargs['loss_ratio']
        self.cls_pred_mean = torch.zeros(self.n_classes).to(self.device)

        self.stat_cnt = 0
        self.distill_mean = torch.zeros(1).to(self.device)
        self.distill_sumsq = torch.zeros(1).to(self.device)
        self.distill_var = torch.ones(1).to(self.device)
        self.cls_mean = torch.zeros(1).to(self.device)
        self.cls_sumsq = torch.zeros(1).to(self.device)
        self.cls_var = torch.ones(1).to(self.device)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, sample_num, n_worker):
        super().online_step(sample, sample_num, n_worker)
        self.update_ema_model()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and memory_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)
            if len(self.memory) > 0:
                cnt = memory_data['usage']
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True, cnt=cnt)

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

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True, cnt=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.ema_model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam, cnt_a, cnt_b = cutmix_data(x=x, y=y, alpha=1.0, z=cnt)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True, detached=self.reduce_bpdepth)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    self.ema_model.zero_grad()
                    if self.importance_measure == 'none':
                        with torch.no_grad():
                            logit2, features2 = self.ema_model(x, get_features=True, detached=False)
                        layer_loss = [((features2[i].detach() - feature) ** 2).mean(dim=(2, 3)).sum(dim=1) for
                                      i, feature in enumerate(features)]
                    else:
                        logit2, features2 = self.ema_model(x, get_features=True, detached=False)
                        cls_loss_2 = (
                                lam * criterion(logit2, labels_a) + (1 - lam) * criterion(logit2, labels_b)).mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        self.calculate_importance(grads, features, features2)
                        layer_loss = [
                            (self.norm_importance[i] * ((features2[i].detach() - feature) ** 2).mean(dim=(2, 3))).sum(
                                dim=1) for i, feature in enumerate(features)]
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean[labels_a[i]] += (1 - self.ema_ratio) ** lam * (
                                    torch.clamp((prob[labels_a[i]] / lam), 0, 1) - self.cls_pred_mean[labels_a[i]])
                            self.cls_pred_mean[labels_b[i]] += (1 - self.ema_ratio) ** (1 - lam) * (
                                    torch.clamp((prob[labels_b[i]] / (1 - lam)), 0, 1) - self.cls_pred_mean[
                                labels_b[i]])
                        sample_weight = torch.clamp((self.cls_pred_mean[labels_a] / lam), 0, 1) * lam + torch.clamp(
                            (self.cls_pred_mean[labels_b] / lam), 0, 1) * (1 - lam)
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * (cnt_a * lam + cnt_b * (1 - lam)) * np.log(2)).to(
                            self.device)
                    else:
                        sample_weight = 0.5
                    distill_loss = sum(layer_loss)
                    if self.norm_loss:
                        self.update_stats(cls_loss.detach(), distill_loss.detach())
                        if self.distill_var > 0:
                            beta = torch.sqrt(self.cls_var / self.distill_var)
                        else:
                            beta = 0
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, features = self.model(x, get_features=True, detached=self.reduce_bpdepth)
                    cls_loss = criterion(logit, y)
                    self.ema_model.zero_grad()
                    if self.importance_measure == 'none':
                        with torch.no_grad():
                            logit2, features2 = self.ema_model(x, get_features=True, detached=False)
                        layer_loss = [((features2[i].detach() - feature) ** 2).mean(dim=(2, 3)).sum(dim=1) for
                                      i, feature in enumerate(features)]
                    else:
                        logit2, features2 = self.ema_model(x, get_features=True, detached=False)
                        cls_loss_2 = criterion(logit2, y).mean()
                        for tensor in features2:
                            tensor.retain_grad()
                        cls_loss_2.backward()
                        grads = []
                        for tensor in features2:
                            grads.append(copy.deepcopy(tensor.grad))
                        self.calculate_importance(grads, features, features2)
                        layer_loss = [
                            (self.norm_importance[i] * ((features2[i].detach() - feature) ** 2).mean(dim=(2, 3))).sum(
                                dim=1) for i, feature in
                            enumerate(features)]
                    distill_loss = sum(layer_loss)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean[y[i]] += (1 - self.ema_ratio) * (prob[y[i]] - self.cls_pred_mean[y[i]])
                        sample_weight = self.cls_pred_mean[y]
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * cnt * np.log(2)).to(self.device)
                    else:
                        sample_weight = 0.5
                    if self.norm_loss:
                        self.update_stats(cls_loss.detach(), distill_loss.detach())
                        if self.distill_var > 0:
                            beta = torch.sqrt(self.cls_var / self.distill_var)
                        else:
                            beta = 0
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def update_stats(self, cls_loss, distill_loss):
        cls_loss_diff = cls_loss.mean() - self.cls_mean
        cls_loss_sumsq = torch.sum((cls_loss - self.cls_mean) ** 2)
        dist_loss_diff = distill_loss.mean() - self.distill_mean
        dist_loss_sumsq = torch.sum((distill_loss - self.distill_mean) ** 2)
        self.cls_mean += cls_loss_diff * len(cls_loss) / (self.stat_cnt + len(cls_loss))
        self.distill_mean += dist_loss_diff * len(distill_loss) / (self.stat_cnt + len(distill_loss))
        self.cls_sumsq += cls_loss_sumsq + cls_loss_diff ** 2 * self.stat_cnt * len(cls_loss) / (
                self.stat_cnt + len(cls_loss))
        self.distill_sumsq += dist_loss_sumsq + dist_loss_diff ** 2 * self.stat_cnt * len(distill_loss) / (
                self.stat_cnt + len(distill_loss))
        self.stat_cnt += len(cls_loss)
        if self.stat_cnt >= 2:
            self.cls_var = self.cls_sumsq / (self.stat_cnt - 1)
            self.distill_var = self.distill_sumsq / (self.stat_cnt - 1)

    def calculate_importance(self, grads, features, features2):
        if self.importance is None:
            self.importance = [torch.zeros(grad.size(1), device=self.device) for grad in grads]
            self.norm_importance = copy.deepcopy(self.importance)
        if self.importance_measure == 'grad':
            for i, importance in enumerate(self.importance):
                self.importance[i] = self.importance_ema * importance + (1 - self.importance_ema) * (
                        grads[i] ** 2).mean(dim=(0, 2, 3))
                self.norm_importance[i] = (self.importance[i] + 1e-8) / torch.mean(self.importance[i] + 1e-8)
        if self.importance_measure == 'sample_grad':
            self.importance = [(grad ** 2).mean(dim=(2, 3)) for grad in grads]
            self.norm_importance = [(importance + 1e-8) / torch.mean(importance + 1e-8) for importance in
                                    self.importance]
        elif self.importance_measure == 'grad_consistency':
            for i, importance in enumerate(self.importance):
                if grads[i].size(0) > 1:
                    self.importance[i] = self.importance_ema * importance + (1 - self.importance_ema) * (
                        grads[i].std(dim=0).mean(dim=(1, 2)))
                self.norm_importance[i] = (self.importance[i] + 1e-8) / torch.mean(self.importance[i] + 1e-8)
        elif self.importance_measure == 'correlation':
            for i, importance in enumerate(self.importance):
                self.importance[i] = self.importance_ema * importance + (1 - self.importance_ema) * (
                        (grads[i] * (features[i].detach() - features2[i].detach())).sum(dim=(2, 3)) / ((
                                                                                                               (
                                                                                                                       features[
                                                                                                                           i].detach() -
                                                                                                                       features2[
                                                                                                                           i].detach()) ** 2).mean(
                    dim=(2, 3)) + 1e-8)).mean(dim=0)
            # T = 1000.0
            # overall_importance = F.relu(torch.cat(self.importance))
            # overall_importance /= torch.mean(overall_importance) + 1e-8
            # overall_importance *= len(overall_importance)
            j = 0
            for i, importance in enumerate(self.importance):
                self.norm_importance[i] = F.relu(importance)
                self.norm_importance[i] /= torch.mean(self.norm_importance[i]) + 1e-8

                # j += len(importance)


class Ours_v12(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.model_2 = copy.deepcopy(self.model)
        self.optimizer_2 = select_optimizer(self.opt_name, self.lr / 10, self.model_2)
        self.feature_only = kwargs['feature_only']
        self.temp_batch = []
        self.reset_period = 100

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.update_memory(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.batch_size:
            train_loss, train_acc, train_loss_2, train_acc_2 = self.online_train(self.temp_batch, self.batch_size,
                                                                                 n_worker,
                                                                                 iterations=int(self.num_updates) // 2)
            self.report_training(sample_num, train_loss, train_acc)
            self.report_training(sample_num, train_loss_2, train_acc_2)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates) // 2 * 2
            self.update_schedule()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        total_loss_2, correct_2 = 0.0, 0.0
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform)
        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            x_2 = stream_data['image']
            y_2 = stream_data['label']

            x_2 = x_2.to(self.device)
            y_2 = y_2.to(self.device)

            memory_data = self.memory.get_batch(min(batch_size, len(self.memory)))
            x = memory_data['image']
            y = memory_data['label']

            x = x.to(self.device)
            y = y.to(self.device)

            self.model_2.train()
            self.optimizer_2.zero_grad()
            logit_2, loss_2 = self.model_forward_(self.model_2, self.model, x_2, y_2, beta=self.beta)
            _, preds_2 = logit_2.topk(self.topk, 1, True, True)
            if self.use_amp:
                self.scaler.scale(loss_2).backward()
                self.scaler.step(self.optimizer_2)
                self.scaler.update()
            else:
                loss_2.backward()
                self.optimizer_2.step()

            self.model.train()
            self.optimizer.zero_grad()
            logit, loss = self.model_forward_(self.model, self.model_2, x, y, beta=self.beta)
            _, preds = logit.topk(self.topk, 1, True, True)
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
            total_loss_2 += loss_2.item()
            correct_2 += torch.sum(preds_2 == y_2.unsqueeze(1)).item()

        return total_loss / iterations, correct / num_data, total_loss_2 / iterations, correct_2 / num_data

    def model_forward_(self, model, model_2, x, y, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        model_2.train()
        # if np.random.rand(1) < 0.5:
        #     model_2.train()
        # else:
        #     model_2.eval()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = model(x, get_feature=True)
                cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                with torch.no_grad():
                    logit2, feature2 = model_2(x, get_feature=True)
                distill_loss = ((feature2.detach() - feature) ** 2).mean(dim=1)
                loss = (cls_loss + beta * distill_loss).mean()
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = model(x, get_feature=True)
                cls_loss = criterion(logit, y)
                with torch.no_grad():
                    logit2, feature2 = model_2(x, get_feature=True)
                distill_loss = ((feature2.detach() - feature) ** 2).mean(dim=1)
                loss = (cls_loss + beta * distill_loss).mean()
        return logit, loss

    def online_evaluate_2(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation_2(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        return eval_dict

    def evaluation_2(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model_2.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model_2(x)

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


class Ours_v13(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.weighted = kwargs['weighted']
        self.pred_based = kwargs['pred_based']
        self.ema_ratio = kwargs['ema_ratio']
        self.model_2 = copy.deepcopy(self.model)
        self.optimizer_2 = select_optimizer(self.opt_name, self.lr / 10, self.model_2)
        self.model_3 = copy.deepcopy(self.model)
        self.feature_only = kwargs['feature_only']
        self.temp_batch = []
        self.reset_period = 100

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def reset_models(self):
        self.model_3 = copy.deepcopy(self.model_2)
        self.model_2 = copy.deepcopy(self.model)
        self.optimizer_2 = select_optimizer(self.opt_name, self.lr / 10, self.model_2)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        while len(self.temp_batch) > self.batch_size:
            del self.temp_batch[0]
        self.update_memory(sample)
        self.num_updates += self.online_iter

        if (sample_num - 1) % self.reset_period == 0:
            self.reset_models()

        if self.num_updates >= 1:
            train_loss, train_acc, train_loss_2, train_acc_2 = self.online_train(self.temp_batch, self.batch_size,
                                                                                 n_worker,
                                                                                 iterations=int(self.num_updates),
                                                                                 sample_num=sample_num)
            self.report_training(sample_num, train_loss, train_acc)
            self.report_training(sample_num, train_loss_2, train_acc_2)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, sample_num=0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        total_loss_2, correct_2 = 0.0, 0.0
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform)
        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            x_2 = stream_data['image']
            y_2 = stream_data['label']

            x_2 = x_2.to(self.device)
            y_2 = y_2.to(self.device)

            memory_data = self.memory.get_batch(min(batch_size, len(self.memory)))
            x = memory_data['image']
            y = memory_data['label']

            x = x.to(self.device)
            y = y.to(self.device)

            self.model_2.train()
            self.optimizer_2.zero_grad()
            logit_2, loss_2 = self.model_forward_(self.model_2, self.model_3, x_2, y_2, beta=self.beta,
                                                  distill=sample_num > self.reset_period)
            _, preds_2 = logit_2.topk(self.topk, 1, True, True)
            if self.use_amp:
                self.scaler.scale(loss_2).backward()
                self.scaler.step(self.optimizer_2)
                self.scaler.update()
            else:
                loss_2.backward()
                self.optimizer_2.step()

            self.model.train()
            self.optimizer.zero_grad()
            logit, loss = self.model_forward_(self.model, self.model_3, x, y, beta=self.beta,
                                              distill=sample_num > self.reset_period)
            _, preds = logit.topk(self.topk, 1, True, True)
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
            total_loss_2 += loss_2.item()
            correct_2 += torch.sum(preds_2 == y_2.unsqueeze(1)).item()

        return total_loss / iterations, correct / num_data, total_loss_2 / iterations, correct_2 / num_data

    def model_forward_(self, model, model_2, x, y, distill=True, beta=10.0, use_cutmix=True):
        if distill:
            criterion = nn.CrossEntropyLoss(reduction='none')
            model_2.train()
            # if np.random.rand(1) < 0.5:
            #     model_2.train()
            # else:
            #     model_2.eval()
            do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        logit2, feature2 = model_2(x, get_feature=True)
                    distill_loss = ((feature2.detach() - feature) ** 2).mean(dim=1)
                    loss = (cls_loss + beta * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    with torch.no_grad():
                        logit2, feature2 = model_2(x, get_feature=True)
                    distill_loss = ((feature2.detach() - feature) ** 2).mean(dim=1)
                    loss = (cls_loss + beta * distill_loss).mean()
            return logit, loss
        else:
            criterion = nn.CrossEntropyLoss()
            do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = model(x, get_feature=True)
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = model(x, get_feature=True)
                    loss = criterion(logit, y)
            return logit, loss

    def online_evaluate_2(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation_2(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        return eval_dict

    def evaluation_2(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model_2.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model_2(x)

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


class Ours_v14(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.ema_ratio = kwargs['ema_ratio']
        self.ema_model = copy.deepcopy(self.model)
        self.importance_measure = kwargs['importance']
        self.importance_ema = kwargs['imp_ema']
        self.importance = None
        self.norm_importance = None
        self.norm_loss = kwargs['norm_loss']
        self.loss_ratio = kwargs['loss_ratio']
        self.online_fc = nn.Linear(self.model.fc.in_features, self.model.fc.out_features).to(self.device)
        params = [param for name, param in self.online_fc.named_parameters()]
        self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.temp_ret = None

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        fc_prev_weight = copy.deepcopy(self.online_fc.weight.data)
        self.online_fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.online_fc.weight[:self.num_learned_class - 1] = fc_prev_weight
        params = [param for name, param in self.online_fc.named_parameters()]
        self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        while len(self.temp_batch) > self.batch_size:
            del self.temp_batch[0]
        self.update_memory(sample)
        self.num_updates += self.online_iter

        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

        self.update_ema_model()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            memory_data = self.memory.get_batch(memory_batch_size)
            x = memory_data['image']
            y = memory_data['label']
            x = x.to(self.device)
            y = y.to(self.device)
            if len(self.memory) > 0:
                cnt = memory_data['usage']
            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True, cnt=cnt)

            _, preds = logit.topk(self.topk, 1, True, True)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            stream_data = sample_dataset.get_data()
            x_fc = stream_data['image'].to(self.device)
            y_fc = stream_data['label'].to(self.device)
            fc_loss = self.fc_forward(x_fc, y_fc, use_cutmix=True)
            self.fc_optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(fc_loss).backward()
                self.scaler.step(self.fc_optimizer)
                self.scaler.update()
            else:
                fc_loss.backward()
                self.fc_optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def fc_forward(self, x, y, use_cutmix=True):
        criterion = nn.CrossEntropyLoss()
        model = copy.deepcopy(self.model)
        model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
        else:
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = criterion(logit, y)
        return cls_loss

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True, cnt=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if np.random.rand(1) < 0.5:
            self.ema_model.train()
        else:
            self.ema_model.eval()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam, cnt_a, cnt_b = cutmix_data(x=x, y=y, alpha=1.0, z=cnt)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    else:
                        grad2 = lam * self.get_grad(logit2, labels_a, self.ema_model.fc.weight) + (
                                1 - lam) * self.get_grad(logit2, labels_b, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio) ** lam * (
                                    torch.clamp((prob[labels_a[i]] / lam), 0, 1) - self.cls_pred_mean)
                            self.cls_pred_mean += (1 - self.ema_ratio) ** (1 - lam) * (
                                    torch.clamp((prob[labels_b[i]] / (1 - lam)), 0, 1) - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = lam * torch.clamp((probs[torch.arange(y.size(0)), labels_a] / lam), 0, 1) + (
                                1 - lam) * torch.clamp((probs[torch.arange(y.size(0)), labels_b] / (1 - lam)), 0, 1)
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * (cnt_a * lam + cnt_b * (1 - lam)) * np.log(2)).to(
                            self.device)
                    else:
                        sample_weight = 0.5
                    if self.norm_loss:
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() + 1e-8))
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    self.ema_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    else:
                        grad2 = self.get_grad(logit2, y, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio) * (prob[y[i]] - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = probs[torch.arange(y.size(0)), y]
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * cnt * np.log(2)).to(self.device)
                    else:
                        sample_weight = 0.5
                    if self.norm_loss:
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8))
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def calculate_importance(self, grad, feature, feature2):
        if self.importance is None:
            self.importance = torch.zeros(grad.size(1), device=self.device)
            self.norm_importance = copy.deepcopy(self.importance)
        if self.importance_measure == 'grad':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad ** 2).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        if self.importance_measure == 'sample_grad':
            self.importance = grad ** 2
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8, dim=1, keepdim=True)
        elif self.importance_measure == 'grad_consistency':
            if grad.size(0) > 1:
                self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad.std(dim=0))
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        elif self.importance_measure == 'correlation':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (
                    F.relu(grad * (feature.detach() - feature2.detach())) ** 2 / (
                    ((feature.detach() - feature2.detach()) ** 2) + 1e-8)).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)
        return torch.matmul((prob - oh_label), weight)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        fc_eval_dict = eval_dict["fc_ret"]
        fc_online_acc = self.calculate_online_acc(fc_eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        fc_eval_dict["online_acc"] = fc_online_acc
        self.temp_ret = fc_eval_dict

        return eval_dict

    def online_evaluate_2(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        eval_dict = self.temp_ret
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        total_correct_fc, total_loss_fc = 0.0, 0.0
        correct_l_fc = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
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

                logit2 = self.online_fc(feature)
                loss2 = criterion(logit2, y)
                pred2 = torch.argmax(logit2, dim=-1)
                _, preds2 = logit2.topk(self.topk, 1, True, True)
                total_correct_fc += torch.sum(preds2 == y.unsqueeze(1)).item()
                xlabel_cnt2, correct_xlabel_cnt2 = self._interpret_pred(y, pred2)
                correct_l_fc += correct_xlabel_cnt2.detach().cpu()
                total_loss_fc += loss2.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        avg_acc_fc = total_correct_fc / total_num_data
        avg_loss_fc = total_loss_fc / len(test_loader)
        cls_acc_fc = (correct_l_fc / (num_data_l + 1e-5)).numpy().tolist()
        fc_ret = {"avg_loss": avg_loss_fc, "avg_acc": avg_acc_fc, "cls_acc": cls_acc_fc}
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "fc_ret": fc_ret}

        return ret


class Ours_v15(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.ema_ratio_1 = kwargs['ema_ratio']
        self.ema_ratio_2 = kwargs['ema_ratio_2']
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model_1 = copy.deepcopy(self.model)
        self.ema_model_2 = copy.deepcopy(self.model)
        # self.initialize_ema_model()
        self.importance_measure = kwargs['importance']
        self.importance_ema = kwargs['imp_ema']
        self.importance = None
        self.norm_importance = None
        self.ema_updates = 0
        self.num_steps = 0
        self.norm_loss = kwargs['norm_loss']
        self.loss_ratio = kwargs['loss_ratio']
        self.online_fc = nn.Linear(self.model.fc.in_features, self.model.fc.out_features).to(self.device)
        params = [param for name, param in self.online_fc.named_parameters()]
        self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.temp_ret = None
        self.cls_pred_length = 100
        self.cls_pred = []

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        fc_prev_weight = copy.deepcopy(self.online_fc.weight.data)
        self.online_fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.online_fc.weight[:self.num_learned_class - 1] = fc_prev_weight
        params = [param for name, param in self.online_fc.named_parameters()]
        self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)
        self.cls_pred.append([])

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        while len(self.temp_batch) > self.batch_size:
            del self.temp_batch[0]
        self.update_memory(sample)
        self.num_updates += self.online_iter
        self.num_steps = sample_num
        if self.loss_ratio == 'cls_pred_based':
            self.sample_inference([sample])

        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

        self.update_ema_model()

    def sample_inference(self, sample):
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=False)
        self.ema_model.eval()
        stream_data = sample_dataset.get_data()
        x = stream_data['image']
        y = stream_data['label']
        x = x.to(self.device)
        logit = self.ema_model(x)
        prob = F.softmax(logit, dim=1)
        self.cls_pred[y].append(prob[0, y].item())
        if len(self.cls_pred[y]) > self.cls_pred_length:
            del self.cls_pred[y][0]
        self.cls_pred_mean = np.clip(np.mean([np.mean(cls_pred) for cls_pred in self.cls_pred]) - 1/self.num_learned_class, 0, 1) * self.num_learned_class/(self.num_learned_class + 1)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size)
        else:
            memory_batch_size = 0

        for i in range(iterations):
            self.model.train()
            memory_data = self.memory.get_batch(memory_batch_size)
            x = memory_data['image']
            y = memory_data['label']
            x = x.to(self.device)
            y = y.to(self.device)
            if len(self.memory) > 0:
                cnt = memory_data['usage']

            logit, loss = self.model_forward(x, y, memory_batch_size // 2, self.beta, use_cutmix=True, cnt=cnt)

            _, preds = logit.topk(self.topk, 1, True, True)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            stream_data = sample_dataset.get_data()
            x_fc = stream_data['image'].to(self.device)
            y_fc = stream_data['label'].to(self.device)
            fc_loss = self.fc_forward(x_fc, y_fc, use_cutmix=True)
            self.fc_optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(fc_loss).backward()
                self.scaler.step(self.fc_optimizer)
                self.scaler.update()
            else:
                fc_loss.backward()
                self.fc_optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def initialize_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        for name, param in model_params.items():
            ema_params_1[name].copy_(torch.zeros_like(param))
            ema_params_2[name].copy_(torch.zeros_like(param))
            ema_params[name].copy_(torch.zeros_like(param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())
        ema_buffers_1 = OrderedDict(self.ema_model_1.named_buffers())
        ema_buffers_2 = OrderedDict(self.ema_model_2.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)
            ema_buffers_1[name].copy_(buffer)
            ema_buffers_2[name].copy_(buffer)

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        self.ema_updates += 1
        for name, param in model_params.items():
            ema_params_1[name].sub_((1. - self.ema_ratio_1) * (ema_params_1[name] - param))
            ema_params_2[name].sub_((1. - self.ema_ratio_2) * (ema_params_2[name] - param))
            ema_params[name].copy_(
                (1. - self.ema_ratio_2) / (self.ema_ratio_1 - self.ema_ratio_2) * ema_params_1[name] - (
                            1. - self.ema_ratio_1) / (self.ema_ratio_1 - self.ema_ratio_2) * ema_params_2[
                    name])
            # + ((1. - self.ema_ratio_2)*self.ema_ratio_1**self.ema_updates - (1. - self.ema_ratio_1)*self.ema_ratio_2**self.ema_updates) / (self.ema_ratio_1 - self.ema_ratio_2) * param)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def fc_forward(self, x, y, use_cutmix=True):
        criterion = nn.CrossEntropyLoss()
        model = copy.deepcopy(self.model)
        model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
        else:
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = criterion(logit, y)
        return cls_loss

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True, cnt=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.ema_model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam, cnt_a, cnt_b = cutmix_data(x=x, y=y, alpha=1.0, z=cnt)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    else:
                        grad2 = lam * self.get_grad(logit2, labels_a, self.ema_model.fc.weight) + (
                                1 - lam) * self.get_grad(logit2, labels_b, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio_1) ** lam * (
                                    torch.clamp((prob[labels_a[i]] / lam), 0, 1) - self.cls_pred_mean)
                            self.cls_pred_mean += (1 - self.ema_ratio_1) ** (1 - lam) * (
                                    torch.clamp((prob[labels_b[i]] / (1 - lam)), 0, 1) - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = lam * torch.clamp((probs[torch.arange(y.size(0)), labels_a] / lam), 0, 1) + (
                                1 - lam) * torch.clamp((probs[torch.arange(y.size(0)), labels_b] / (1 - lam)), 0, 1)
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'cls_pred_based':
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * (cnt_a * lam + cnt_b * (1 - lam)) * np.log(2)).to(
                            self.device)
                    else:
                        sample_weight = 0.5
                    if self.norm_loss == 'sample':
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() + 1e-8))
                    elif self.norm_loss == 'batch':
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() + 1e-8)).mean()
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    self.ema_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    else:
                        grad2 = self.get_grad(logit2, y, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio_1) * (prob[y[i]] - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = probs[torch.arange(y.size(0)), y]
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'usage_cnt':
                        sample_weight = 1 - torch.exp(-1 * cnt * np.log(2)).to(self.device)
                    elif self.loss_ratio == 'cls_pred_based':
                        sample_weight = self.cls_pred_mean
                    else:
                        sample_weight = 0.5
                    if self.norm_loss == 'sample':
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8))
                    elif self.norm_loss == 'batch':
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def calculate_importance(self, grad, feature, feature2):
        if self.importance is None:
            self.importance = torch.zeros(grad.size(1), device=self.device)
            self.norm_importance = copy.deepcopy(self.importance)
        if self.importance_measure == 'grad':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad ** 2).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        if self.importance_measure == 'sample_grad':
            self.importance = grad ** 2
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8, dim=1, keepdim=True)
        elif self.importance_measure == 'grad_consistency':
            if grad.size(0) > 1:
                self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad.std(dim=0))
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        elif self.importance_measure == 'correlation':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (
                    F.relu(grad * (feature.detach() - feature2.detach())) ** 2 / (
                    ((feature.detach() - feature2.detach()) ** 2) + 1e-8)).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)
        return torch.matmul((prob - oh_label), weight)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        fc_eval_dict = eval_dict["fc_ret"]
        fc_online_acc = self.calculate_online_acc(fc_eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        fc_eval_dict["online_acc"] = fc_online_acc
        self.temp_ret = fc_eval_dict

        return eval_dict

    def online_evaluate_2(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        eval_dict = self.temp_ret
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        total_correct_fc, total_loss_fc = 0.0, 0.0
        correct_l_fc = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
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

                logit2 = self.online_fc(feature)
                loss2 = criterion(logit2, y)
                pred2 = torch.argmax(logit2, dim=-1)
                _, preds2 = logit2.topk(self.topk, 1, True, True)
                total_correct_fc += torch.sum(preds2 == y.unsqueeze(1)).item()
                xlabel_cnt2, correct_xlabel_cnt2 = self._interpret_pred(y, pred2)
                correct_l_fc += correct_xlabel_cnt2.detach().cpu()
                total_loss_fc += loss2.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        avg_acc_fc = total_correct_fc / total_num_data
        avg_loss_fc = total_loss_fc / len(test_loader)
        cls_acc_fc = (correct_l_fc / (num_data_l + 1e-5)).numpy().tolist()
        fc_ret = {"avg_loss": avg_loss_fc, "avg_acc": avg_acc_fc, "cls_acc": cls_acc_fc}
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "fc_ret": fc_ret}

        return ret



class Ours_v16(Ours):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.beta = kwargs['beta']
        self.dma_mean = kwargs['dma_mean']
        self.dma_varcoeff = kwargs['dma_var']
        assert 0.5 - 1 / self.dma_mean < self.dma_varcoeff < 1 - 1 / self.dma_mean
        self.ema_ratio_1 = (1 - np.sqrt(2*self.dma_varcoeff - 1 + 2/self.dma_mean))/(self.dma_mean - 1 - self.dma_mean*self.dma_varcoeff)
        self.ema_ratio_2 = (1 + np.sqrt(2 * self.dma_varcoeff - 1 + 2 / self.dma_mean)) / (
                    self.dma_mean - 1 - self.dma_mean * self.dma_varcoeff)
        self.cur_time = None
        if kwargs['sched_name'] == 'default':
            self.sched_name = 'pred_based'
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model_1 = copy.deepcopy(self.model)
        self.ema_model_2 = copy.deepcopy(self.model)
        # self.initialize_ema_model()
        self.importance_measure = kwargs['importance']
        self.importance_ema = kwargs['imp_ema']
        self.importance = None
        self.norm_importance = None
        self.ema_updates = 0
        self.num_steps = 0
        self.norm_loss = kwargs['norm_loss']
        if self.norm_loss == 'cumulative':
            self.cls_grad_sum = 0.0
            self.dis_grad_sum = 0.0
        self.loss_ratio = kwargs['loss_ratio']
        self.fc_mode = kwargs['fc_train']
        if self.fc_mode == 'finetune':
            self.saved_fc_weight = copy.deepcopy(self.model.fc.weight)
            self.saved_fc_bias = copy.deepcopy(self.model.fc.bias)
        self.online_fc_mode = kwargs['online_fc_mode']
        if self.online_fc_mode != 'none':
            self.online_fc = nn.Linear(self.model.fc.in_features, self.model.fc.out_features).to(self.device)
            params = [param for name, param in self.online_fc.named_parameters()]
            self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.temp_ret = None
        self.cls_pred_length = 100
        self.cls_pred = []

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        if self.online_fc_mode != 'none':
            fc_prev_weight = copy.deepcopy(self.online_fc.weight.data)
            self.online_fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
            with torch.no_grad():
                if self.num_learned_class > 1:
                    self.online_fc.weight[:self.num_learned_class - 1] = fc_prev_weight
            params = [param for name, param in self.online_fc.named_parameters()]
            self.fc_optimizer = torch.optim.Adam(params, lr=self.lr / 10, weight_decay=0)
        self.cls_pred.append([])
        if self.fc_mode != 'none':
            sdict = copy.deepcopy(self.optimizer.state_dict())
            fc_params = sdict['param_groups'][1]['params']
            if len(sdict['state']) > 0:
                fc_weight_state = sdict['state'][fc_params[0]]
                fc_bias_state = sdict['state'][fc_params[1]]
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
            if len(sdict['state']) > 0:
                if 'adam' in self.opt_name:
                    fc_weight = self.optimizer.param_groups[1]['params'][0]
                    fc_bias = self.optimizer.param_groups[1]['params'][1]
                    self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                    self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                            torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                                dim=1)]).to(self.device)], dim=0)
                    self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                               torch.zeros([1, fc_weight_state[
                                                                                   'exp_avg_sq'].size(dim=1)]).to(
                                                                                   self.device)], dim=0)
                    self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                    self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                          torch.tensor([0]).to(
                                                                              self.device)], dim=0)
                    self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                             torch.tensor([0]).to(
                                                                                 self.device)], dim=0)
            if self.fc_mode == 'finetune':
                prev_saved_weight = copy.deepcopy(self.saved_fc_weight.data)
                prev_saved_bias = copy.deepcopy(self.saved_fc_bias.data)
                self.saved_fc_weight = copy.deepcopy(self.model.fc.weight)
                self.saved_fc_bias = copy.deepcopy(self.model.fc.bias)
                with torch.no_grad():
                    if self.num_learned_class > 1:
                        self.saved_fc_weight[:self.num_learned_class - 1] = prev_saved_weight
                        self.saved_fc_bias[:self.num_learned_class - 1] = prev_saved_bias

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        while len(self.temp_batch) > self.batch_size:
            del self.temp_batch[0]
        self.update_memory(sample)
        self.num_updates += self.online_iter
        self.num_steps = sample_num
        if self.loss_ratio == 'cls_pred_based':
            self.sample_inference([sample])

        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
        self.update_schedule()


    def sample_inference(self, sample):
        sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=False, use_kornia=False)
        self.ema_model.eval()
        stream_data = sample_dataset.get_data()
        x = stream_data['image']
        y = stream_data['label']
        x = x.to(self.device)
        logit = self.ema_model(x)
        
        # 어차피 samplewise이므로 len(x) = 1
        self.total_flops += self.forward_flops
        
        prob = F.softmax(logit, dim=1)
        self.cls_pred[y].append(prob[0, y].item())
        if len(self.cls_pred[y]) > self.cls_pred_length:
            del self.cls_pred[y][0]
        self.cls_pred_mean = np.clip(np.mean([np.mean(cls_pred) for cls_pred in self.cls_pred]) - 1/self.num_learned_class, 0, 1) * self.num_learned_class/(self.num_learned_class + 1)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0, beta=10.0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, 0)
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            logit, loss = self.model_forward(x, y, batch_size // 2, self.beta, use_cutmix=True)

            _, preds = logit.topk(self.topk, 1, True, True)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.fc_mode == 'finetune':
                with torch.no_grad():
                    self.model.fc.weight.sub_(0.01 * (self.model.fc.weight - self.saved_fc_weight))
                    self.model.fc.bias.sub_(0.01 * (self.model.fc.bias - self.saved_fc_bias))

            if self.online_fc_mode != 'none':
                if len(sample) > 0:
                    self.memory.register_stream(sample)
                stream_data = self.memory.get_batch(len(sample), len(sample))
                x_fc = stream_data['image'].to(self.device)
                y_fc = stream_data['label'].to(self.device)
                fc_loss = self.fc_forward(x_fc, y_fc, use_cutmix=True)
                self.fc_optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(fc_loss).backward()
                    self.scaler.step(self.fc_optimizer)
                    self.scaler.update()
                else:
                    fc_loss.backward()
                    self.fc_optimizer.step()
                if self.online_fc_mode == 'finetune':
                    with torch.no_grad():
                        self.online_fc.weight.sub_(0.01 * (self.online_fc.weight - self.model.fc.weight))
                        self.online_fc.bias.sub_(0.01 * (self.online_fc.bias - self.model.fc.bias))

            self.update_schedule()
            self.update_ema_model(num_updates=1.0)

            # ema model도 forward하므로 forward_flops * 2
            self.total_flops += batch_size * (2 * self.forward_flops + self.backward_flops)
            print("total_flops", self.total_flops)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def update_schedule(self, reset=False):
        if self.sched_name == 'pred_based':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * (1 - self.cls_pred_mean)
        else:
            super().update_schedule(reset)

    @torch.no_grad()
    def update_ema_model(self, num_updates=1.0):
        ema_inv_ratio_1 = (1 - self.ema_ratio_1) ** num_updates
        ema_inv_ratio_2 = (1 - self.ema_ratio_2) ** num_updates
        
        # OrderedDict를 통해서 무작위 순서로 data를 받는 것이 아니라, data의 순서를 보장받을 수 있다.
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        ema_params_1 = OrderedDict(self.ema_model_1.named_parameters())
        ema_params_2 = OrderedDict(self.ema_model_2.named_parameters())
        assert model_params.keys() == ema_params.keys()
        assert model_params.keys() == ema_params_1.keys()
        assert model_params.keys() == ema_params_2.keys()
        self.ema_updates += 1
        for name, param in model_params.items():
            ema_params_1[name].sub_((1. - ema_inv_ratio_1) * (ema_params_1[name] - param)) # - 1개, * 1개, sub 1개
            ema_params_2[name].sub_((1. - ema_inv_ratio_2) * (ema_params_2[name] - param)) # - 1개, * 1개, sub 1개
            ema_params[name].copy_( # * 2개, sub 1개
                self.ema_ratio_2 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_1[name] - self.ema_ratio_1 / (self.ema_ratio_2 - self.ema_ratio_1) * ema_params_2[
                    name])    
        self.total_flops += (9 * self.params)
        
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        self.ema_model_1.fc = copy.deepcopy(self.model.fc)
        self.ema_model_2.fc = copy.deepcopy(self.model.fc)
        #self.total_flops += (3 * self.fc_params)
        
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)
        #self.total_flops += self.buffers


    def fc_forward(self, x, y, use_cutmix=True):
        criterion = nn.CrossEntropyLoss()
        model = copy.deepcopy(self.model)
        model.train()
        
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
        else:
            with torch.no_grad():
                logit, feature = model(x, get_feature=True)
            logit = self.online_fc(feature.detach())
            cls_loss = criterion(logit, y)
        return cls_loss

    def model_forward(self, x, y, distill_size=0, beta=10.0, use_cutmix=True):
        criterion = nn.CrossEntropyLoss(reduction='none')
        self.ema_model.train()
        do_cutmix = use_cutmix and self.cutmix and np.random.rand(1) < 0.5
        if distill_size > 0:
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(logit, labels_b)
                    self.total_flops += (len(logit) * 4) / 10e9
                    
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        # feature shape [20, 512]
                        # distill_loss length 20
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                        self.total_flops += ((feature.shape[0] * feature.shape[1] * 3) / 10e9)
                        
                    else:
                        grad2 = lam * self.get_grad(logit2, labels_a, self.ema_model.fc.weight) + (
                                1 - lam) * self.get_grad(logit2, labels_b, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio_1) ** lam * (
                                    torch.clamp((prob[labels_a[i]] / lam), 0, 1) - self.cls_pred_mean)
                            self.cls_pred_mean += (1 - self.ema_ratio_1) ** (1 - lam) * (
                                    torch.clamp((prob[labels_b[i]] / (1 - lam)), 0, 1) - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = lam * torch.clamp((probs[torch.arange(y.size(0)), labels_a] / lam), 0, 1) + (
                                1 - lam) * torch.clamp((probs[torch.arange(y.size(0)), labels_b] / (1 - lam)), 0, 1)
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'cls_pred_based':
                        sample_weight = self.cls_pred_mean
                    else:
                        sample_weight = 0.5
                    if self.norm_loss == 'sample':
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() + 1e-8))
                    elif self.norm_loss == 'batch': #here
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        # grad shape [16, 512]
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() + 1e-8)).mean()
                        self.total_flops += (grad.shape[0] * grad.shape[1] * 3 + len(distill_loss)) / 10e9
                    elif self.norm_loss == 'cumulative':
                        grad = lam * self.get_grad(logit.detach(), labels_a, self.model.fc.weight) + (
                                1 - lam) * self.get_grad(logit.detach(), labels_b, self.model.fc.weight)
                        self.cls_grad_sum += torch.sqrt((grad.detach() ** 2).sum(dim=1)).sum()
                        self.dis_grad_sum += torch.sqrt(distill_loss.detach()).sum()
                        beta = self.cls_grad_sum / (self.dis_grad_sum + 1e-8)
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            else:
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = self.model(x, get_feature=True)
                    cls_loss = criterion(logit, y)
                    self.ema_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.ema_model(x, get_feature=True)
                    if self.importance_measure == 'none':
                        distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                        self.total_flops += ((feature.shape[0] * feature.shape[1] * 3) / 10e9)
                    else:
                        grad2 = self.get_grad(logit2, y, self.ema_model.fc.weight)
                        self.calculate_importance(grad2, feature, feature2)
                        distill_loss = ((self.norm_importance * (feature - feature2.detach())) ** 2).sum(dim=1)
                        
                    if self.loss_ratio == 'pred_based':
                        probs = F.softmax(logit2, dim=1)
                        for i, prob in enumerate(probs):
                            self.cls_pred_mean += (1 - self.ema_ratio_1) * (prob[y[i]] - self.cls_pred_mean)
                        sample_weight = self.cls_pred_mean
                    elif self.loss_ratio == 'batch_pred_based':
                        probs = F.softmax(logit2, dim=1)
                        pred = probs[torch.arange(y.size(0)), y]
                        sample_weight = pred.detach().mean()
                    elif self.loss_ratio == 'cls_pred_based':
                        sample_weight = self.cls_pred_mean
                    else:
                        sample_weight = 0.5
                        
                    if self.norm_loss == 'sample':
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8))
                    elif self.norm_loss == 'batch':
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                        self.total_flops += (grad.shape[0] * grad.shape[1] * 3 + len(distill_loss)) / 10e9
                    elif self.norm_loss == 'cumulative':
                        grad = self.get_grad(logit.detach(), y, self.model.fc.weight)
                        self.cls_grad_sum += torch.sqrt((grad.detach() ** 2).sum(dim=1)).sum()
                        self.dis_grad_sum += torch.sqrt(distill_loss.detach()).sum()
                        beta = self.cls_grad_sum / (self.dis_grad_sum + 1e-8)
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
            return logit, loss
        else:
            return super().model_forward(x, y)

    def calculate_importance(self, grad, feature, feature2):
        if self.importance is None:
            self.importance = torch.zeros(grad.size(1), device=self.device)
            self.norm_importance = copy.deepcopy(self.importance)
        if self.importance_measure == 'grad':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad ** 2).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        if self.importance_measure == 'sample_grad':
            self.importance = grad ** 2
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8, dim=1, keepdim=True)
        elif self.importance_measure == 'grad_consistency':
            if grad.size(0) > 1:
                self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (grad.std(dim=0))
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)
        elif self.importance_measure == 'correlation':
            self.importance = self.importance_ema * self.importance + (1 - self.importance_ema) * (
                    F.relu(grad * (feature.detach() - feature2.detach())) ** 2 / (
                    ((feature.detach() - feature2.detach()) ** 2) + 1e-8)).mean(
                dim=0)
            self.norm_importance = (self.importance + 1e-8) / torch.mean(self.importance + 1e-8)

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)
        
        # matmul의 flops
        # (n×p) * (p×m) 꼴 일때
        # flops: nm(2p−1)
        front = (prob-oh_label).shape
        back = weight.shape
        self.total_flops += ((front[0] * back[1] * (2*front[1] - 1)) / 10e9)
        
        return torch.matmul((prob - oh_label), weight)

    def get_class_weight(self):
        fc_weights = self.model.fc.weight.data.detach().cpu()
        distill_weight = torch.abs(fc_weights - torch.mean(fc_weights, dim=0))
        distill_weight /= torch.sum(distill_weight, dim=1).reshape(-1, 1) + 1e-8
        return distill_weight

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        if self.online_fc_mode != 'none':
            fc_eval_dict = eval_dict["fc_ret"]
            fc_online_acc = self.calculate_online_acc(fc_eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
            fc_eval_dict["online_acc"] = fc_online_acc
            self.temp_ret = fc_eval_dict
        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
            self.f_calculated = True
        else:
            self.f_calculated = False
        return eval_dict

    def online_evaluate_2(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        eval_dict = self.temp_ret
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])
        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        total_correct_fc, total_loss_fc = 0.0, 0.0
        correct_l_fc = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
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

                if self.online_fc_mode != 'none':

                    logit2 = self.online_fc(feature)
                    loss2 = criterion(logit2, y)
                    pred2 = torch.argmax(logit2, dim=-1)
                    _, preds2 = logit2.topk(self.topk, 1, True, True)
                    total_correct_fc += torch.sum(preds2 == y.unsqueeze(1)).item()
                    xlabel_cnt2, correct_xlabel_cnt2 = self._interpret_pred(y, pred2)
                    correct_l_fc += correct_xlabel_cnt2.detach().cpu()
                    total_loss_fc += loss2.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        avg_acc_fc = total_correct_fc / total_num_data
        avg_loss_fc = total_loss_fc / len(test_loader)
        cls_acc_fc = (correct_l_fc / (num_data_l + 1e-5)).numpy().tolist()
        fc_ret = {"avg_loss": avg_loss_fc, "avg_acc": avg_acc_fc, "cls_acc": cls_acc_fc}
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "fc_ret": fc_ret}

        return ret
