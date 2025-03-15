# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
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
from utils.train_utils import select_model, select_optimizer, select_scheduler, get_data_loader
from utils.afd import MultiTaskAFDAlternative
from utils.data_loader import partial_distill_loss
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class TWF(ER):
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

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
        self.kwargs = kwargs

        self.lambda_fp_replay = kwargs["lambda_fp_replay"]
        self.lambda_diverse_loss = kwargs["lambda_diverse_loss"]
        self.min_resize_threshold = kwargs["min_resize_threshold"]
        self.resize_maps = kwargs["resize_maps"]

        # 임시 방편
        self.n_tasks = 5
        self.cpt = 2 #classes per task

        # opt_dict 설정
        '''
        opt_dict = {}
        opt_dict['device'] = self.device
        opt_dict['batch_size'] = self.batch_size
        opt_dict['n_worker'] = kwargs["n_worker"]
        opt_dict['data_dir'] = kwargs["data_dir"]
        opt_dict['sigma'] = kwargs["sigma"]
        opt_dict['repeat'] = kwargs['repeat']
        opt_dict['init_cls'] = kwargs['init_cls']
        opt_dict["lr"] = kwargs["lr"]
        opt_dict["optim_wd"] = 0
        opt_dict["optim_mom"] = 0
        opt_dict[]
        '''
        kwargs["device"] = self.device
        self.model = select_model(self.model_name, self.dataset, 1, opt_dict = kwargs).to(self.device)

        # pre_trained model 설정
        self.pretrain_model = copy.deepcopy(self.model.eval()) 
        for p in self.pretrain_model.parameters():
            p.requires_grad = False


        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        print("self.train_transform")
        print(self.train_transform)
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia, save_test = True)
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

    def get_flops_parameter(self):
        self.dataset
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset)
        forward_mac, backward_mac, params, fc_params, buffers = get_model_complexity_info(self.model, (inp_channel, inp_size, inp_size), as_strings=False,
                                           print_per_layer_stat=False, verbose=True, criterion = self.criterion, original_opt=self.optimizer, opt_name = self.opt_name, lr=self.lr)
        
        # flops = float(mac) * 2 # mac은 string 형태
        print("forward mac", forward_mac, "backward mac", backward_mac, "params", params, "fc_params", fc_params, "buffers", buffers)
        self.forward_flops = forward_mac / 10e9
        self.backward_flops = backward_mac / 10e9
        self.params = params / 10e9
        self.fc_params = fc_params / 10e9
        self.buffers = buffers / 10e9
        

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            train_loss, train_acc, logits, attention_maps = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            print("logits")
            print(logits.shape)
            print("attention maps")
            print(len(attention_maps))
            print("attention_shape")
            print(attention_maps[0].shape)
            self.report_training(sample_num, train_loss, train_acc)
            for idx, stored_sample in enumerate(self.temp_batch):
                self.update_memory(stored_sample, logits[idx], [at[idx].byte() for at in attention_maps])
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

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

            stream_batch_size = min(stream_batch_size, len(self.memory.stream_images))
            batch_size = min(batch_size, stream_batch_size + len(self.memory.images))
            memory_batch_size = batch_size - stream_batch_size

            data, buf_data = self.memory.get_batch(batch_size, stream_batch_size, twf=True)
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            self.optimizer.zero_grad()
            logit, loss, all_features = self.model_forward(x, y, get_features=True)
            self.model.train()
            # TODO cut mix 고려 안함
            # logit, all_features = self.model(x, get_features=True)

            stream_logit = logit[:stream_batch_size]
            stream_partial_features = [feature[:stream_batch_size] for feature in all_features]
            memory_logit = logit[stream_batch_size:]

            # Use pre-trained model
            all_pret_logits, all_pret_features = self.pretrain_model(x, get_features=True)
            stream_pret_logits = all_pret_logits[:stream_batch_size]
            stream_pret_partial_features = [feature[:stream_batch_size] for feature in all_pret_features]

            logits = torch.cat([stream_logit, stream_pret_logits], dim=1)

            if memory_batch_size == 0:
                loss_afd, stream_attention_maps = partial_distill_loss(self.model, 
                        stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, y, self.device)

            else:
                '''
                buffer_teacher_forcing = torch.div(
                    buf_labels, self.cpt, rounding_mode='floor') != self.task
                '''
                buf_logits = buf_data["logits"]
                d = buf_data["d"]
                task_ids = buf_data["task_ids"]
                print("d")
                print(len(d))
                buf_labels = y[stream_batch_size:]
                buf_inputs, buf_attention_maps = torch.stack([v[0] for v in d]).to(self.device), [[o.to(self.device) for o in v[1]] for v in d]

                buffer_teacher_forcing = task_ids != self.task
                teacher_forcing = torch.cat(
                    (torch.zeros((stream_batch_size)).bool().to(self.device), buffer_teacher_forcing))
                attention_maps = [
                    [torch.ones_like(map) for map in buf_attention_maps[0]]]*stream_batch_size + buf_attention_maps
                loss_afd, all_attention_maps = partial_distill_loss(self.model, all_partial_features[-len(
                    all_pret_partial_features):], all_pret_partial_features, y, self.device,
                    teacher_forcing, attention_maps)

                stream_attention_maps = [ap[:stream_batch_size] for ap in all_attention_maps]

                loss_er = self.loss(buf_outputs[:, :(self.task+1)*self.cpt], buf_labels)
                loss_der = F.mse_loss(buf_outputs, buf_logits[:, :self.num_classes])


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
        return total_loss / iterations, correct / num_data, logits, stream_attention_maps


    def model_forward(self, x, y, get_features=False):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if get_features:
                        logit, features = self.model(x, get_features=True)
                    else:
                        logit = self.model(x)
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
            else:
                if get_features:
                    logit, features = self.model(x, get_features=True)
                else:
                    logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if get_features:
                        logit, features = self.model(x, get_features=True)
                    else:
                        logit = self.model(x)
                    loss = self.criterion(logit, y)
            else:
                if get_features:
                    logit, features = self.model(x, get_features=True)
                else:
                    logit = self.model(x)
                loss = self.criterion(logit, y)

        if get_features:
            return logit, loss, features
        else:
            return logit, loss

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

    def update_memory(self, sample, logit, attention_map):
        self.reservoir_memory(sample, logit, attention_map)

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

    def online_before_task(self, task_id):
        self.task_id = task_id
        train_loader, _ = get_data_loader(self.kwargs, self.dataset)

        for data in train_loader:
            x = data["image"]
            x = x.to(self.device)
            _, feats_t = self.model(x, get_features=True)
            _, pret_feats_t = self.pretrain_model(x, get_features=True)
            print(x.shape)
            break

        for i, (x, pret_x) in enumerate(zip(feats_t, pret_feats_t)):
            # clear_grad=self.args.detach_skip_grad == 1
            adapt_shape = x.shape[1:]
            pret_shape = pret_x.shape[1:]
            if len(adapt_shape) == 1:
                adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
                pret_shape = (pret_shape[0], 1, 1)

            setattr(self.model, f"adapter_{i+1}", MultiTaskAFDAlternative(
                adapt_shape, self.n_tasks, self.cpt, clear_grad=False,
                teacher_forcing_or=False,
                lambda_forcing_loss=self.lambda_fp_replay,
                use_overhaul_fd=True, use_hard_softmax=True,
                lambda_diverse_loss=self.lambda_diverse_loss,
                attn_mode="chsp",
                min_resize_threshold=self.min_resize_threshold,
                resize_maps=self.resize_maps == 1,
            ).to(self.device))

    def online_after_task(self, cur_iter):
        self.model.eval()

        self.memory.loop_over_buffer(self.model, self.pretrain_model)

        self.model.train()

    def reservoir_memory(self, sample, logit=None, attention_map=None):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                if logit is None:
                    self.memory.replace_sample(sample, j)
                else:
                    self.memory.replace_sample(sample, j, logit, attention_map)

        else:
            self.memory.replace_sample(sample, logit=logit, attention_map = attention_map)

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
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

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
    # def calculate_online_forgetting(self, n_classes, gt, y_t1, y_t2):
    #     cnt = np.array([])
    #     gt_y1_y2 = np.zeros([0, 3])
    #     for i, y_gt in enumerate(gt):
    #         gt, y1, y2 = y_gt, y_t1[i], y_t2[i]
    #         len_cnt = len(cnt)
    #         gt_idx = np.searchsorted(gt_y1_y2[:, 0], gt)
    #         if gt_idx < len_cnt:
    #             if gt_y1_y2[gt_idx, 0] == gt:
    #                 y1_idx = np.searchsorted(gt_y1_y2[gt_idx:, 1], y1)
    #
    #             else:
    #                 cnt = np.insert(cnt, gt_idx, 1)
    #                 gt_y1_y2 = np.insert(gt_y1_y2, gt_idx, [gt, y1, y2], axis=0)
    #         else:
    #             cnt = np.insert(cnt, gt_idx, 1)
    #             gt_y1_y2 = np.insert(gt_y1_y2, gt_idx, [gt, y1, y2], axis=0)






