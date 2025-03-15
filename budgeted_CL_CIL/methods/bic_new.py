# @inproceedings{wu2019large,
#   title={Large scale incremental learning},
#   author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={374--382},
#   year={2019}
# }
import logging
import copy
from copy import deepcopy

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from methods.cl_manager import MemoryBase
from methods.er_new import ER
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader
from utils.train_utils import select_model, cycle
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


class BiasCorrectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # ax + b의 꼴이므로 len(x) * 2 만큼의 flops가 추가됨
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        # unsqueeze 함수는 dimension을 늘려주는 함수
        correction = self.linear(x.unsqueeze(dim=2))
        correction = correction.squeeze(dim=2)
        return correction


class BiasCorrection(ER):
    def __init__(
        self, train_datalist, test_datalist, device, **kwargs
    ):
        kwargs["memory_size"] -= kwargs["val_memory_size"]

        self.bias_layer = None
        self.valid_list = []
        self.future_valid_list = []

        self.valid_size = kwargs["val_memory_size"]

        self.n_tasks = kwargs["n_tasks"]
        self.bias_layer_list = []
        self.distilling = True
        self.samples_per_task = kwargs["samples_per_task"]

        self.val_per_cls = self.valid_size
        self.val_full = False
        self.future_val_full = False

        self.cur_iter = 0
        self.bias_labels = []
        self.use_sample = dict()
        """
        self.valid_list: valid set which is used for training bias correction layer.
        self.memory_list: training set only including old classes. As already mentioned in the paper,
            memory list and valid list are exclusive.
        self.bias_layer_list - the list of bias correction layers. The index of the list means the task number.
        """
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        self.prev_model = select_model(
            self.model_name, self.dataset, 1
        )

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = MemoryBase(self.memory_size)

        for _ in range(-(self.total_samples//-self.samples_per_task)):
            bias_layer = BiasCorrectionLayer().to(self.device)
            self.bias_layer_list.append(bias_layer)

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
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()


    def online_before_task(self, sample_num):
        self.cur_iter = sample_num // self.samples_per_task
        self.bias_labels.append([])
        if self.distilling:
            self.prev_model = deepcopy(self.model)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.val_full = False
        use_sample = self.future_valid_update(sample)
        self.use_sample[self.future_sample_num] = use_sample
        self.future_num_updates += self.online_iter
        if use_sample:
            self.temp_future_batch.append(sample)
            if len(self.temp_future_batch) >= self.temp_batch_size:
                self.generate_waiting_batch(int(self.future_num_updates))
                for stored_sample in self.temp_future_batch:
                    self.update_memory(stored_sample)
                self.temp_future_batch = []
                self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def online_step(self, sample, sample_num, n_worker):
        if (sample_num-1) % self.samples_per_task == 0:
            self.online_before_task(sample_num)
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.num_updates += self.online_iter
        if self.use_sample[sample_num-1]:
            self.temp_batch.append(sample)
            if len(self.temp_batch)>= self.temp_batch_size:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.temp_batch = []
                self.num_updates -= int(self.num_updates)
        else:
            self.online_valid_update(sample)

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
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})

        self.bias_labels[self.cur_iter].append(self.num_learned_class - 1)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_valid_update(self, sample):
        self.valid_list.append(sample)
        if len(self.valid_list) == self.val_per_cls * self.num_learned_class:
            self.val_full = True
        else:
            self.val_full = False

    def future_valid_update(self, sample):
        val_df = pd.DataFrame(self.future_valid_list, columns=['klass', 'file_name', 'label'])
        if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls:
            self.future_valid_list.append(sample)
            use_sample = False
        else:
            use_sample = True
        if len(self.future_valid_list) == self.val_per_cls * self.num_learned_class:
            self.future_val_full = True
        else:
            self.future_val_full = False
        return use_sample

    def online_train(self, iterations=1):
        self.model.train()
        total_loss, distill_loss, classify_loss, correct, num_data = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(iterations):
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old = self.prev_model(x)
                            else:
                                logit_old = self.prev_model(x)
                                logit_old = self.online_bias_forward(logit_old, self.cur_iter - 1)
                            
                            self.total_flops += (len(x) * (self.forward_flops + (len(logit_old[0])*6)/10e9))

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(self.use_amp):
                    logit_new = self.model(x)
                    loss_c = lam * self.criterion(logit_new, labels_a) + (1 - lam) * self.criterion(
                    logit_new, labels_b)
                    self.total_flops += (len(logit_new) * 4) / 10e9
                self.total_flops += (len(x) * (self.forward_flops + self.backward_flops))
            else:
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(self.use_amp):
                                logit_old = self.prev_model(x)
                                logit_old = self.online_bias_forward(logit_old, self.cur_iter - 1)
                            self.total_flops += (len(x) * (self.forward_flops + (len(logit_old[0])*6)/10e9))

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new = self.model(x)
                        loss_c = self.criterion(logit_new, y)
                        self.total_flops += (len(logit_new) * 2) / 10e9
                else:
                    logit_new = self.model(x)
                    loss_c = self.criterion(logit_new, y)
                    self.total_flops += (len(logit_new) * 2) / 10e9

                self.total_flops += (len(x) * (self.forward_flops + self.backward_flops))

            if self.distilling:
                if self.cur_iter == 0:
                    loss_d = torch.tensor(0.0).to(self.device)
                else:
                    loss_d = self.distillation_loss(logit_old, logit_new[:, : logit_old.size(1)])
            else:
                loss_d = torch.tensor(0.0).to(self.device)

            _, preds = logit_new.topk(self.topk, 1, True, True)
            loss = loss_c + loss_d
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()

            self.update_schedule()
            total_loss += loss.item()
            distill_loss += loss_d.item()
            classify_loss += loss_c.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def online_bias_forward(self, input, iter):
        bias_labels = self.bias_labels[iter]
        bias_layer = self.bias_layer_list[iter]
        if len(bias_labels) > 0:
            input[:, bias_labels] = bias_layer(input[:, bias_labels])
        return input

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        self.online_bias_correction()
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
        total_correct, total_num_data, total_loss = (
            0.0,
            0.0,
            0.0,
        )
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        self.bias_layer.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                xlabel = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = self.online_bias_forward(logit, self.cur_iter)
                logit = logit.detach().cpu()
                loss = self.criterion(logit, xlabel)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == xlabel.unsqueeze(1)).item()
                total_num_data += xlabel.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(xlabel, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += xlabel.tolist()

        eval_dict = self.evaluation(test_loader, self.criterion)
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])

        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
        return eval_dict

    def online_bias_correction(self, n_iter=256, batch_size=100, n_worker=4):
        self.bias_layer_list[self.cur_iter] = BiasCorrectionLayer().to(self.device)
        self.bias_layer = self.bias_layer_list[self.cur_iter]
        print(self.cur_iter, self.bias_labels)

        if self.val_full and self.cur_iter > 0 and len(self.bias_labels[self.cur_iter]) > 0:
            val_df = pd.DataFrame(self.valid_list)
            val_dataset = ImageDataset(val_df, dataset=self.dataset, transform=self.test_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, preload=True)
            bias_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=n_worker)
            criterion = self.criterion
            self.bias_layer = self.bias_layer_list[self.cur_iter]
            optimizer = torch.optim.Adam(params=self.bias_layer.parameters(), lr=0.001)
            self.model.eval()
            total_loss = None
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
                self.total_flops += len(x) * self.forward_flops 
            for iteration in range(n_iter):
                self.bias_layer.train()
                total_loss = 0.0
                for i, out in enumerate(model_out):
                    logit = self.online_bias_forward(out.to(self.device), self.cur_iter)
                    xlabel = xlabels[i]
                    loss = criterion(logit, xlabel.to(self.device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    # forward할 때 *2, backward 할 때 * 4 해서 총 *6
                    # self.backward_flops를 더해주지 않는 이유는 self.model.eval()이므로 어차피 model update는 되지 않기 때문
                    self.total_flops += (len(out)*6)/10e9

                logger.info(
                    "[Stage 2] [{}/{}]\tloss: {:.4f}\talpha: {:.4f}\tbeta: {:.4f}".format(
                        iteration + 1,
                        n_iter,
                        total_loss,
                        self.bias_layer.linear.weight.item(),
                        self.bias_layer.linear.bias.item(),
                    )
                )
            assert total_loss is not None
            self.print_bias_layer_parameters()

    def distillation_loss(self, old_logit, new_logit):
        # new_logit should have same dimension with old_logit.(dimension = n)
        assert new_logit.size(1) == old_logit.size(1)
        T = 2
        old_softmax = torch.softmax(old_logit / T, dim=1)
        new_log_softmax = torch.log_softmax(new_logit / T, dim=1)
        loss = -(old_softmax * new_log_softmax).sum(dim=1).mean()
        return loss

    def print_bias_layer_parameters(self):
        for i, layer in enumerate(self.bias_layer_list):
            logger.info(
                "[{}] alpha: {:.4f}, beta: {:.4f}".format(
                    i, layer.linear.weight.item(), layer.linear.bias.item()
                )
            )

