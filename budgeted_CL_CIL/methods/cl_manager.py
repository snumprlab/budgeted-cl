import logging
import os
import copy

import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flops_counter.ptflops import get_model_complexity_info
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.augment import get_transform
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.block_utils import MODEL_BLOCK_DICT, get_blockwise_flops

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLManagerBase:
    def __init__(self, train_datalist, test_datalist, device, **kwargs):

        self.device = device
        self.task_id = 0
        self.method_name = kwargs["mode"]
        self.dataset = kwargs["dataset"]
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.init_cls = kwargs["init_cls"]
        self.samples_per_task = kwargs["samples_per_task"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]

        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'const'
        self.lr = kwargs["lr"]
        self.block_names = MODEL_BLOCK_DICT[self.model_name]
        self.num_blocks = len(self.block_names) - 1
        self.vit_pretrain = kwargs["vit_pretrain"]
        assert kwargs["temp_batchsize"] <= kwargs["batchsize"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batch_size = kwargs["temp_batchsize"]
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]

        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        self.data_dir = kwargs["data_dir"]
        if self.data_dir is None:
            self.data_dir = os.path.join("dataset", self.dataset)
        self.n_worker = kwargs["n_worker"]
        self.future_steps = kwargs["future_steps"]
        self.transform_on_gpu = kwargs["transform_on_gpu"]
        self.use_kornia = kwargs["use_kornia"]
        self.transform_on_worker = kwargs["transform_on_worker"]

        self.eval_period = kwargs["eval_period"]
        self.topk = kwargs["topk"]
        self.f_period = kwargs["f_period"]

        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.cls_dict = {}
        self.total_samples = len(self.train_datalist)
        
        if 'vit' in self.model_name:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu, 224)
        else:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.transform_on_gpu)
        self.cutmix = "cutmix" in kwargs["transforms"]
        
        self.model = select_model(self.model_name, self.dataset, num_classes = 1, channel_constant=kwargs["channel_constant"], kwinner=kwargs["kwinner"], vit_pretrain=self.vit_pretrain).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        self.data_stream = iter(self.train_datalist)
        # self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)

        self.memory = MemoryBase(self.memory_size)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.sample_num = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0

        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []
        self.knowledge_loss_rate = []
        self.knowledge_gain_rate = []
        self.forgetting_time = []
        self.note = kwargs['note']
        self.rnd_seed = kwargs['rnd_seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, '5_dataset': 260101, 'clear10':30000, 'clear100':110000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        self.total_samples = num_samples[self.dataset]

        self.exposed_domains = []
        self.waiting_batch = []
        self.get_flops_parameter()
        self.initialize_future()
        self.total_flops = 0.0
        self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')


    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = MemoryBase(self.memory_size)

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
            self.generate_waiting_batch(int(self.future_num_updates))
            for stored_sample in self.temp_future_batch:
                self.update_memory(stored_sample)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def update_memory(self, sample):
        pass

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            del self.waiting_batch[0]

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size))

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def add_new_class(self, class_name):
        if hasattr(self.model, 'fc'):
            fc_name = 'fc'
        elif hasattr(self.model, 'head'):
            fc_name = 'head'
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
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

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            self.before_model_update()

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

            self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def before_model_update(self):
        pass

    def after_model_update(self):
        self.update_schedule()

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)

        self.total_flops += (len(y) * self.forward_flops)
        return logit, loss

    def report_training(self, sample_num, train_loss, train_acc):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | TFLOPs {self.total_flops/1000:.2f}"
        )

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()


    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        if "clear" in self.dataset:
            exp_test_df = test_df[test_df['time'] <= self.exposed_domains[-1]]
            if len(self.exposed_domains) == 1:
                exp_test_df = exp_test_df[exp_test_df['klass'].isin(self.exposed_classes)]
        else:
            exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
            
        # exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        
        print("exposed_domains", self.exposed_domains)
        print("exposed_classes", self.exposed_classes)
        print("exp_test_df", len(exp_test_df))
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

        # print("sample_num", sample_num, "self.f_next_time", self.f_next_time)
        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
            self.f_calculated = True
        else:
            self.f_calculated = False
        
        if self.dataset == "5_dataset":
            self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["seen_cls_acc"])
        else:    
            self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        # self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        return eval_dict

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
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        seen_cls_acc = np.mean(cls_acc[:len(self.exposed_classes)])

        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "seen_cls_acc": seen_cls_acc}

        return ret

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def _interpret_pred(self, y, pred):
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
            klr, kgr, = self.calculate_online_forgetting(self.n_classes, self.gt_label, self.test_records[-2], self.test_records[-1], self.n_model_cls[-2], self.n_model_cls[-1])
            self.knowledge_loss_rate.append(klr)
            self.knowledge_gain_rate.append(kgr)
            self.forgetting_time.append(sample_num)
            logger.info(f'KLR {klr} | KGR {kgr}')
            print("###")
            print(self.save_path + '_KLR.npy', self.knowledge_loss_rate)
            print("###")
            np.save(self.save_path + '_KLR.npy', self.knowledge_loss_rate)
            np.save(self.save_path + '_KGR.npy', self.knowledge_gain_rate)
            np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)


    def calculate_online_forgetting(self, n_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2):
        total_cnt = len(y_gt)
        cnt_gt = np.zeros(n_classes)
        cnt_y1 = np.zeros(n_cls_t1)
        cnt_y2 = np.zeros(n_cls_t2)
        correct_y1 = np.zeros(n_classes)
        correct_y2 = np.zeros(n_classes)
        correct_both = np.zeros(n_classes)
        for i, gt in enumerate(y_gt):
            y1, y2 = y_t1[i], y_t2[i]
            cnt_gt[gt] += 1
            cnt_y1[y1] += 1
            cnt_y2[y2] += 1
            if y1 == gt:
                correct_y1[gt] += 1
                if y2 == gt:
                    correct_y2[gt] += 1
                    correct_both[gt] += 1
            elif y2 == gt:
                correct_y2[gt] += 1

        gt_prob = cnt_gt/total_cnt
        y1_prob = cnt_y1/total_cnt
        y2_prob = cnt_y2/total_cnt

        probs = np.zeros([n_classes, n_cls_t1, n_cls_t2])

        for i in range(n_classes):
            cls_prob = gt_prob[i]
            notlearned_prob = 1 - (correct_y1[i] + correct_y2[i] - correct_both[i])/cnt_gt[i]
            forgotten_prob = (correct_y1[i] - correct_both[i]) / cnt_gt[i]
            newlearned_prob = (correct_y2[i] - correct_both[i]) / cnt_gt[i]
            if i < n_cls_t1:
                marginal_y1 = y1_prob/((1-y1_prob[i]) + 1e-10)
                marginal_y1[i] = forgotten_prob/(notlearned_prob+1e-10)
            else:
                marginal_y1 = y1_prob
            if i < n_cls_t2:
                marginal_y2 = y2_prob/((1-y2_prob[i]) + 1e-10)
                marginal_y2[i] = newlearned_prob/(notlearned_prob+1e-10)
            else:
                marginal_y2 = y2_prob
            probs[i] = np.expand_dims(marginal_y1, 1) * np.expand_dims(marginal_y2, 0) * notlearned_prob * cls_prob
            if i < n_cls_t1 and i < n_cls_t2:
                probs[i][i][i] = correct_both[i]/total_cnt

        knowledge_loss = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        prob_gt_y1 = probs.sum(axis=2)
        prev_total_knowledge = np.sum(prob_gt_y1*np.log(prob_gt_y1/(np.sum(prob_gt_y1, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y1, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        max_knowledge = np.log(n_cls_t2)/np.log(n_classes)

        knowledge_loss_rate = knowledge_loss/prev_total_knowledge
        knowledge_gain_rate = knowledge_gain/(max_knowledge-prev_total_knowledge)
        return knowledge_loss_rate, knowledge_gain_rate
    
    def get_flops_parameter(self, method=None):
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset)
        if 'vit' in self.model_name:
            inp_size = 224
        
        self.flops_dict = get_model_complexity_info(self.model, (inp_channel, inp_size, inp_size),
                                                                             as_strings=False,
                                                                             print_per_layer_stat=False, verbose=True,
                                                                             criterion=self.criterion,
                                                                             original_opt=self.optimizer,
                                                                    opt_name=self.opt_name, lr=self.lr)
        forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops  = get_blockwise_flops(self.flops_dict, self.model_name, method)
        self.forward_flops = sum(forward_flops)
        self.backward_flops = sum(backward_flops)
        self.blockwise_forward_flops = forward_flops
        self.blockwise_backward_flops = backward_flops
        self.total_model_flops = self.forward_flops + self.backward_flops
        
        self.G_forward_flops, self.G_backward_flops = sum(G_forward_flops), sum(G_backward_flops)
        self.F_forward_flops, self.F_backward_flops = sum(F_forward_flops), sum(F_backward_flops)
        self.G_blockwise_forward_flops, self.G_blockwise_backward_flops = G_forward_flops, G_backward_flops
        self.F_blockwise_forward_flops, self.F_blockwise_backward_flops = F_forward_flops, F_backward_flops
        
         
    
class MemoryBase:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.images = []
        self.labels = []
        self.update_buffer = ()
        self.cls_dict = dict()
        self.cls_list = []
        self.cls_count = []
        self.cls_idx = []
        self.usage_count = np.array([])
        self.class_usage_count = np.array([])
        self.current_images = []
        self.current_labels = []
        self.current_cls_count = [0 for _ in self.cls_list]
        self.current_cls_idx = [[] for _ in self.cls_list]

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = np.append(self.class_usage_count, 0.0)

    def retrieval(self, size, return_index=False):
        sample_size = min(size, len(self.images))
        memory_batch = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
        if return_index:
            return memory_batch, indices
        else:
            return memory_batch
