# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from utils.cka import linear_CKA
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()



def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ER:
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
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
        self.mode = kwargs["mode"]
        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]

        self.note = kwargs["note"]
        self.rnd_seed = kwargs["rnd_seed"]
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
        self.memory_size = kwargs['memory_size']
        self.get_flops_parameter()
        self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')

    def get_total_flops(self):
        return self.total_flops
    '''
    def get_flops_parameter(self):
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset)
        [forward_mac, backward_mac, params, fc_params, buffers], \
        [initial_forward_mac, initial_backward_mac, initial_params],\
        [group1_forward_mac, group1_backward_mac, group1_params],\
        [group2_forward_mac, group2_backward_mac, group2_params],\
        [group3_forward_mac, group3_backward_mac, group3_params],\
        [group4_forward_mac, group4_backward_mac, group4_params],\
        [fc_forward_mac, fc_backward_mac, _] = get_model_complexity_info(self.model, (inp_channel, inp_size, inp_size), as_strings=False,
                                           print_per_layer_stat=False, verbose=True, criterion = self.criterion, original_opt=self.optimizer, opt_name = self.opt_name, lr=self.lr)
        
        # flops = float(mac) * 2 # mac은 string 형태
        print("forward mac", forward_mac, "backward mac", backward_mac, "params", params, "fc_params", fc_params, "buffers", buffers)
        print("initial forward mac", initial_forward_mac, "initial backward mac", initial_backward_mac, "initial params", initial_params)
        print("group1 forward mac", group1_forward_mac, "group1 backward mac", group1_backward_mac, "group1 params", group1_params)
        print("group2 forward mac", group2_forward_mac, "group2 backward mac", group2_backward_mac, "group2 params", group2_params)
        print("group3 forward mac", group3_forward_mac, "group3 backward mac", group3_backward_mac, "group3 params", group3_params)
        print("group4 forward mac", group4_forward_mac, "group4 backward mac", group4_backward_mac, "group4 params", group4_params)
        print("fc forward mac", fc_forward_mac, "fc backward mac", fc_backward_mac, "fc params", fc_params)
        
        self.forward_flops = forward_mac / 10e9
        self.initial_forward_flops = initial_forward_mac / 10e9
        self.group1_forward_flops = group1_forward_mac / 10e9
        self.group2_forward_flops = group2_forward_mac / 10e9
        self.group3_forward_flops = group3_forward_mac / 10e9
        self.group4_forward_flops = group4_forward_mac / 10e9


        self.backward_flops = backward_mac / 10e9
        self.initial_backward_flops = initial_backward_mac / 10e9
        self.group1_backward_flops = group1_backward_mac / 10e9
        self.group2_backward_flops = group2_backward_mac / 10e9
        self.group3_backward_flops = group3_backward_mac / 10e9
        self.group4_backward_flops = group4_backward_mac / 10e9
        self.comp_backward_flops = [self.initial_backward_flops, self.group1_backward_flops, self.group2_backward_flops, self.group3_backward_flops, self.group4_backward_flops]

        self.params = params / 10e9
        self.fc_params = fc_params / 10e9
        self.buffers = buffers / 10e9
    '''

        
    def online_step(self, sample, sample_num, n_worker):
        self.temp_batchsize = self.batch_size
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            iteration = int(self.num_updates)
            if iteration != 0:
                train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
                self.report_training(sample_num, train_loss, train_acc)
                
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)

                self.temp_batch = []
                self.num_updates -= int(self.num_updates)


    def save_std_pickle(self):
        '''
        class_std, sample_std = self.memory.get_std()
        self.class_std_list.append(class_std)
        self.sample_std_list.append(sample_std)
        '''
        
        cls_file_name = self.mode + '_final_cls_std.pickle'
        sample_file_name = self.mode + '_sample_std.pickle'
        
        with open(cls_file_name, 'wb') as handle:
            pickle.dump(self.class_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        '''
        with open(sample_file_name, 'wb') as handle:
            pickle.dump(self.sample_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''

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
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)

        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            # std check 위해서
            class_std, sample_std = self.memory.get_std()
            self.class_std_list.append(class_std)
            self.sample_std_list.append(sample_std)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x,y)

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
            self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            print("self.total_flops", self.total_flops)

        return total_loss / iterations, correct / num_data

    '''
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
    '''

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                    self.total_flops += (len(logit) * 4) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                self.total_flops += (len(logit) * 4) / 10e9
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    loss = self.criterion(logit, y)
                    self.total_flops += (len(logit) * 2) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                loss = self.criterion(logit, y)
                self.total_flops += (len(logit) * 2) / 10e9

        return logit, loss

    def report_training(self, sample_num, train_loss, train_acc, train_initial_cka=None, train_group1_cka=None, train_group2_cka=None, train_group3_cka=None, train_group4_cka=None):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)

        if train_initial_cka is not None:
            '''
            self.writer.add_scalar(f"train/initial_cka", train_initial_cka, sample_num)
            self.writer.add_scalar(f"train/group1_cka", train_group1_cka, sample_num)
            self.writer.add_scalar(f"train/group2_cka", train_group2_cka, sample_num)
            self.writer.add_scalar(f"train/group3_cka", train_group3_cka, sample_num)
            self.writer.add_scalar(f"train/group4_cka", train_group4_cka, sample_num)
            '''
            cka_dict = {}
            cka_dict["initial_cka"] = train_initial_cka
            cka_dict["train_group1_cka"] = train_group1_cka
            cka_dict["train_group2_cka"] = train_group2_cka
            cka_dict["train_group3_cka"] = train_group3_cka
            cka_dict["train_group4_cka"] = train_group4_cka

            self.writer.add_scalars(f"train/cka", cka_dict, sample_num)
            #self.writer.add_scalar(f"train/fc_cka", train_fc_cka, sample_num)

        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, online_acc):
        self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        self.writer.add_scalar(f"test/online_acc", online_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | online_acc {online_acc:.4f} "
        )

    def report_val(self, sample_num, avg_loss, avg_acc, class_loss, class_magnitude):
        self.writer.add_scalar(f"val/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"val/acc", avg_acc, sample_num)

        '''
        my_summary_writer.add_scalars(f'loss/check_info', {
            'score': score[iteration],
            'score_nf': score_nf[iteration],
        }, iteration)
        '''

        kls_loss_dict = {}
        kls_magnitude_dict = {}
        for kls, loss in enumerate(class_loss):
            if not math.isnan(loss):
                name = "class" + str(kls)
                kls_loss_dict[name] = loss
                '''
                if class_magnitude is not None:
                    kls_magnitude_dict[name] = class_magnitude[kls]
                '''
                
        print("kls_loss_dict")
        print(kls_loss_dict)
        print("class_magnitude")
        print(class_magnitude)
        
        self.writer.add_scalars(f"val/class_loss", kls_loss_dict, sample_num)
        if class_magnitude is not None:
            self.writer.add_scalars(f"val/class_magnitude", kls_magnitude_dict, sample_num)

        logger.info(
            f"Val | Sample # {sample_num} | val_loss {avg_loss:.4f} | val_acc {avg_acc:.4f}"
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

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

        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
            self.f_calculated = True
        else:
            self.f_calculated = False
        
        self.save_std_pickle()
        
        return eval_dict

    def get_forgetting(self, sample_num, test_list, cls_dict, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=list(cls_dict.keys()),
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )

        preds = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        if self.gt_label is None:
            gts = np.concatenate(gts)
            self.gt_label = gts
        self.test_records.append(preds)
        self.n_model_cls.append(copy.deepcopy(self.num_learned_class))
        if len(self.test_records) > 1:
            forgetting, knowledge_gain, total_knowledge, retained_knowledge = self.calculate_online_forgetting(self.n_classes, self.gt_label, self.test_records[-2], self.test_records[-1], self.n_model_cls[-2], self.n_model_cls[-1])
            self.forgetting.append(forgetting)
            self.knowledge_gain.append(knowledge_gain)
            self.total_knowledge.append(total_knowledge)
            self.retained_knowledge.append(retained_knowledge)
            self.forgetting_time.append(sample_num)
            logger.info(f'Forgetting {forgetting} | Knowledge Gain {knowledge_gain} | Total Knowledge {total_knowledge} | Retained Knowledge {retained_knowledge}')
            np.save(self.save_path + '_forgetting.npy', self.forgetting)
            np.save(self.save_path + '_knowledge_gain.npy', self.knowledge_gain)
            np.save(self.save_path + '_total_knowledge.npy', self.total_knowledge)
            np.save(self.save_path + '_retained_knowledge.npy', self.retained_knowledge)
            np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)
        else:
            print("else")

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j, mode=self.mode, online_iter=self.online_iter)
        else:
            self.memory.replace_sample(sample, mode=self.mode, online_iter=self.online_iter)

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

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

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach()
                num_data_l += xlabel_cnt.detach()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def calculate_online_acc(self, cls_acc, data_time, cls_dict, cls_addition):
        mean = (np.arange(self.n_classes*self.repeat)/self.n_classes).reshape(-1, self.n_classes)
        cls_weight = np.exp(-0.5*((data_time-mean)/(self.sigma/100))**2)/(self.sigma/100*np.sqrt(2*np.pi))
        cls_addition = np.array(cls_addition).astype(np.int)
        for i in range(self.n_classes):
            cls_weight[:cls_addition[i], i] = 0
        cls_weight = cls_weight.sum(axis=0)
        cls_order = [cls_dict[cls] for cls in self.exposed_classes]
        for i in range(self.n_classes):
            if i not in cls_order:
                cls_order.append(i)
        cls_weight = cls_weight[cls_order]/np.sum(cls_weight)
        online_acc = np.sum(np.array(cls_acc)*cls_weight)
        return online_acc

    def calculate_online_forgetting(self, n_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2, significance=0.99):
        cnt = {}
        total_cnt = len(y_gt)
        uniform_cnt = len(y_gt)
        cnt_gt = np.zeros(n_classes)
        cnt_y1 = np.zeros(n_cls_t1)
        cnt_y2 = np.zeros(n_cls_t2)
        num_relevant = 0
        for i, gt in enumerate(y_gt):
            y1, y2 = y_t1[i], y_t2[i]
            cnt_gt[gt] += 1
            cnt_y1[y1] += 1
            cnt_y2[y2] += 1
            if (gt, y1, y2) in cnt.keys():
                cnt[(gt, y1, y2)] += 1
            else:
                cnt[(gt, y1, y2)] = 1
        cnt_list = list(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
        for i, item in enumerate(cnt_list):
            chi2_value = total_cnt
            for j, item_ in enumerate(cnt_list[i + 1:]):
                expect = total_cnt / (n_classes * n_cls_t1 * n_cls_t2 - i)
                chi2_value += (item_[1] - expect) ** 2 / expect - expect
            if chi2.cdf(chi2_value, n_classes ** 3 - 2 - i) < significance:
                break
            uniform_cnt -= item[1]
            num_relevant += 1
        probs = uniform_cnt * np.ones([n_classes, n_cls_t1, n_cls_t2]) / ((n_classes * n_cls_t1 * n_cls_t2 - num_relevant) * total_cnt)
        for j in range(num_relevant):
            gt, y1, y2 = cnt_list[j][0]
            probs[gt][y1][y2] = cnt_list[j][1] / total_cnt
        forgetting = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        prob_gt_y2 = probs.sum(axis=1)
        total_knowledge = np.sum(prob_gt_y2*np.log(prob_gt_y2/(np.sum(prob_gt_y2, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y2, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(self.n_classes)
        retained_knowledge = total_knowledge - knowledge_gain

        return forgetting, knowledge_gain, total_knowledge, retained_knowledge

    def n_samples(self, n_samples):
        self.total_samples = n_samples
        
