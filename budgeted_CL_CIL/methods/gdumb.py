import logging
import random
import copy
import math
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import torch

from methods.er_baseline import ER
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.data_loader import ImageDataset, GdumbMemory, cutmix_data
from torch.utils.data import DataLoader

import ray
from configuration import config

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
args = config.base_parser()
if args.mode == 'gdumb':
    ray.init(num_gpus=args.num_gpus)


class GDumb(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.memory_size = kwargs["memory_size"]
        self.n_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.batch_size = kwargs["batchsize"]
        self.eval_period = kwargs["eval_period"]
        self.eval_samples = []
        self.eval_time = []
        self.task_time = []
        self.eval_data_time = []
        self.memory = GdumbMemory(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia)

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.exposed_classes.append(sample['klass'])
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(cls_list=self.exposed_classes)
        self.update_memory(sample)

    def update_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        if sample_num not in self.eval_time and sample_num % self.eval_period == 0:
            self.eval_samples.append(copy.deepcopy(self.memory.datalist))
            self.eval_time.append(sample_num)
            self.eval_data_time.append(data_time)
        return {'avg_loss': 0.0, 'avg_acc': 0.0, 'cls_acc': np.zeros(self.n_classes), 'online_acc': 0.0}

    def evaluate_all(self, test_list, n_epoch, batch_size, n_worker, cls_dict, cls_addition):
        eval_results = defaultdict(list)
        num_workers = args.num_gpus*args.workers_per_gpu
        num_evals = len(self.eval_samples)
        forgetting_time = [self.f_period*i for i in range(self.total_samples//self.f_period + 1)]
        forgetting_measure = []
        i = 0
        for j, eval_t in enumerate(self.eval_time):
            if i >= len(forgetting_time):
                break
            if eval_t >= forgetting_time[i]:
                forgetting_measure.append(j)
                i += 1
        gts_saved = []
        preds_saved = []
        for i in range(math.ceil(num_evals/num_workers)):
            workers = [RemoteTrainer.remote(self.model_name, self.dataset, self.n_classes, self.opt_name, self.lr,
                                            'cos', self.eval_samples[i*num_workers+j], test_list, self.criterion,
                                            self.train_transform, self.test_transform, self.cutmix,
                                            use_amp=self.use_amp, data_dir=self.data_dir, gpu_transform=self.gpu_transform,
                                            get_forgetting=i*num_workers+j in forgetting_measure, cls_dict=cls_dict)
                       for j in range(min(num_workers, num_evals-num_workers*i))]
            eval_dicts = ray.get([workers[j].eval_worker.remote(n_epoch, batch_size, n_worker) for j in range(min(num_workers, num_evals-num_workers*i))])
            for j, eval_dict in enumerate(eval_dicts):
                eval_results["test_acc"].append(eval_dict['avg_acc'])
                eval_results["percls_acc"].append(eval_dict['cls_acc'])
                online_accuracy = self.calculate_online_acc(eval_dict['cls_acc'], self.eval_data_time[i*num_workers+j], cls_dict, cls_addition)
                eval_results["online_acc"].append(online_accuracy)
                eval_results["data_cnt"].append(self.eval_time[i*num_workers+j])
                if i*num_workers+j in forgetting_measure:
                    gts_saved.append(eval_dict['gts'])
                    preds_saved.append(eval_dict['preds'])
                    eval_results["forgetting_acc"].append(eval_dict['cls_acc'])
                writer.add_scalar(f"test/loss", eval_dict["avg_loss"], self.eval_time[i*num_workers+j])
                writer.add_scalar(f"test/acc", eval_dict["avg_acc"], self.eval_time[i*num_workers+j])
                writer.add_scalar(f"test/online_acc", online_accuracy, self.eval_time[i*num_workers+j])
                logger.info(
                    f"Test | Sample # {self.eval_time[i*num_workers+j]} | test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | online_acc {online_accuracy:.4f} | "
                )
        for i, j in enumerate(forgetting_measure):
            if self.gt_label is None:
                self.gt_label = gts_saved[i]
            self.test_records.append(preds_saved[i])
            self.n_model_cls.append(copy.deepcopy(self.num_learned_class))
            if len(self.test_records) > 1:
                forgetting, knowledge_gain, total_knowledge, retained_knowledge = self.calculate_online_forgetting(
                    self.n_classes, self.gt_label, self.test_records[-2], self.test_records[-1], self.n_model_cls[-2],
                    self.n_model_cls[-1])
                self.forgetting.append(forgetting)
                self.knowledge_gain.append(knowledge_gain)
                self.total_knowledge.append(total_knowledge)
                self.retained_knowledge.append(retained_knowledge)
                self.forgetting_time.append(self.eval_time[j])
                logger.info(
                    f'Forgetting {forgetting} | Knowledge Gain {knowledge_gain} | Total Knowledge {total_knowledge} | Retained Knowledge {retained_knowledge}')
                np.save(self.save_path + '_forgetting.npy', self.forgetting)
                np.save(self.save_path + '_knowledge_gain.npy', self.knowledge_gain)
                np.save(self.save_path + '_total_knowledge.npy', self.total_knowledge)
                np.save(self.save_path + '_retained_knowledge.npy', self.retained_knowledge)
                np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)
        return eval_results

    def after_task(self, cur_iter):
        pass


@ray.remote(num_gpus=1 / args.workers_per_gpu)
class RemoteTrainer:
    def __init__(self, model_name, dataset, n_classes, opt_name, lr, sched_name, train_list, test_list,
                 criterion, train_transform, test_transform, cutmix, device=0, use_amp=False, data_dir=None, gpu_transform=False,
                 get_forgetting=False, cls_dict=None):
        self.model_name = model_name
        self.dataset = dataset
        self.n_classes = n_classes
        self.cls_dict = cls_dict

        self.train_list = train_list
        self.test_list = test_list

        self.train_transform = train_transform
        self.test_transform = test_transform
        self.cutmix = cutmix

        self.exposed_classes = pd.DataFrame(self.train_list)["klass"].unique().tolist()
        self.num_learned_class = len(self.exposed_classes)

        self.model = select_model(
            model_name, dataset, self.num_learned_class
        )
        self.device = device
        self.model = self.model.cuda(self.device)
        self.criterion = criterion.cuda(self.device)
        self.topk = 1

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.lr = lr
        # Initialize the optimizer and scheduler
        logger.info("Reset the optimizer and scheduler states")
        self.optimizer = select_optimizer(
            opt_name, self.lr, self.model
        )
        self.scheduler = select_scheduler(sched_name, self.optimizer)
        self.data_dir = data_dir
        self.gpu_transform = gpu_transform
        self.forgetting = get_forgetting

    def eval_worker(self, n_epoch, batch_size, n_worker):
        train_dataset = ImageDataset(
            pd.DataFrame(self.train_list),
            dataset=self.dataset,
            transform=self.train_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            preload=self.gpu_transform,
            device=self.device,
            transform_on_gpu=self.gpu_transform,
            use_kornia=self.gpu_transform
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=min(len(self.train_list), batch_size),
            num_workers=n_worker,
        )

        self.model.train()

        for epoch in range(n_epoch):
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            total_loss, correct, num_data = 0.0, 0.0, 0.0

            idxlist = train_dataset.generate_idx(batch_size)
            for idx in idxlist:
                data = train_dataset.get_data_gpu(idx)
                x = data["image"]
                y = data["label"]

                x = x.cuda(self.device)
                y = y.cuda(self.device)

                self.optimizer.zero_grad()

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
            # n_batches = len(train_loader)
            # train_loss, train_acc = total_loss / n_batches, correct / num_data

        test_df = pd.DataFrame(self.test_list)
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

        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.cuda(self.device)
                y = y.cuda(self.device)
                logit = self.model(x)

                loss = self.criterion(logit, y)
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
        if self.forgetting:
            preds, gts = self.get_forgetting(self.test_list, batch_size, n_worker)
            ret['preds'] = preds
            ret['gts'] = gts

        return ret


    def get_forgetting(self, test_list, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=list(self.cls_dict.keys()),
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
        gts = np.concatenate(gts)
        return preds, gts

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