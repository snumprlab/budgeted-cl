# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from operator import attrgetter
import time
import datetime
import random
import numpy as np
import torch
import pickle
import math
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import pickle

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import cutmix_data, MultiProcessLoader
from utils import autograd_hacks
logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class Ours(CLManagerBase):
    def __init__(
            self, train_datalist, test_datalist, device, **kwargs
    ):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        
        
        # for ours
        self.T = kwargs["temperature"]
        self.corr_warm_up = kwargs["corr_warm_up"]
        self.selected_num = 512
        self.corr_map = {}
        self.count_decay_ratio = kwargs["count_decay_ratio"]
        self.k_coeff = kwargs["k_coeff"]
        
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        self.freeze_idx = []
        self.add_new_class_time = []
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.device = device
        self.grad_cls_score_mavg = {}
        self.corr_map_list = []
        self.sample_count_list = []
        self.labels_list=[]
        self._supported_layers = ['Linear', 'Conv2d']
        self.freeze_warmup = 500
        self.grad_dict = {}
        
        self.last_grad_mean = 0.0

        self.grad_mavg = []
        self.grad_mavgsq = []
        self.grad_mvar = []
        self.grad_criterion = []
        
        self.grad_ema_ratio = 0.001
        
        self.freezing_stats = {'i/c': [], 'bi/c': [], 'grad_magnitude': [], 'grad_size': [], 'input_size':[], 'flops':[self.blockwise_forward_flops]}
        
        # Information based freezing
        self.unfreeze_rate = kwargs["unfreeze_rate"]
        self.fisher_ema_ratio = 0.001
        self.fisher = [0.0 for _ in range(self.num_blocks)]
        
        self.cumulative_fisher = []
        self.frozen = False

        self.cumulative_fisher = []


        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp =  kwargs["use_amp"]
        self.cls_weight_decay = kwargs["cls_weight_decay"]
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # for name, p in self.model.named_parameters():
        #     print(name, p.shape)


    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = OurMemory(self.memory_size, self.T, self.count_decay_ratio, self.k_coeff, self.device)

        self.grad_score_per_layer = None
        self.target_layers = []
        if 'resnet' in self.model_name:
            for n, p in self.model.named_parameters():
                if '0.weight' in n and 'conv2' in n:
                    self.target_layers.append(n)
        elif 'cnn' in self.model_name:
            for n, p in self.model.named_parameters():
                if 'conv.weight' in n:
                    self.target_layers.append(n)

        autograd_hacks.add_hooks(self.model)
        self.selected_mask = {}
        self.grad_mavg_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        self.param_count = {n: int((p.numel()) * 0.0001) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        #print("self.param_count")
        #print(self.param_count)
        for key in self.grad_mavg_base.keys():
            a = self.grad_mavg_base[key].flatten()
            #selected_indices = torch.randperm(len(a))[:self.selected_num]
            selected_indices = torch.randperm(len(a))[:self.param_count[key]]
            self.selected_mask[key] = selected_indices
        
        self.grad_size = sum(self.param_count.values())
        self.cls_grad = torch.zeros([0, self.grad_size]).to(self.device)
        self.self_cls_sim = torch.ones([0]).to(self.device)
        self.cls_ema_coeff = torch.zeros([0]).to(self.device)
        
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

        self.sim_matrix = torch.zeros([0, 0]).to(self.device)
        # self.self_cls_sim = 0.0
        # self.other_cls_sim = 0.0

        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def generate_waiting_batch(self, iterations, similarity_matrix=None):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size, similarity_matrix=similarity_matrix))
            # with open(f'{self.save_path}_retrieval.pkl', 'wb') as fp:
            #     pickle.dump(self.memory.retrieval_stats, fp)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.future_add_new_class()
            
        if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
            self.exposed_domains.append(sample["time"])
            
        self.update_memory(sample, self.future_sample_num)
        self.future_num_updates += self.online_iter

        if self.future_num_updates >= 1:
            if self.future_sample_num >= self.corr_warm_up and 'vit' not in self.model_name:
                self.generate_waiting_batch(int(self.future_num_updates), self.sim_matrix)
            else:
                self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def _layer_type(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def unfreeze_layers(self):
        self.frozen = False
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def freeze_layers(self):
        if len(self.freeze_idx) > 0:
            self.frozen = True
        for i in self.freeze_idx:
            self.freeze_layer(i)

    def freeze_layer(self, block_index):
        # blcok(i)가 들어간 layer 모두 freeze
        block_name = self.block_names[block_index]
        # print("freeze", group_name)
        for subblock_name in block_name:
            for name, param in self.model.named_parameters():
                if subblock_name in name:
                    param.requires_grad = False

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
            self.add_new_class_time.append(sample_num)
            # print("seed", self.rnd_seed, "dd_new_class_time")
            # print(self.add_new_class_time)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()


    def future_add_new_class(self):
        n = self.sim_matrix.size(0)
        if n > 0:
            self.cls_grad = torch.cat([self.cls_grad, torch.zeros([1, self.grad_size]).to(self.device)])
            self.self_cls_sim = torch.cat([self.self_cls_sim, self.self_cls_sim.mean(dim=0, keepdim=True)])
            self.cls_ema_coeff = torch.cat([self.cls_ema_coeff, torch.Tensor([0.0]).to(self.device)])
            prev_sim_matrix = copy.deepcopy(self.sim_matrix)
            diagonal_avg = torch.diag(self.sim_matrix).mean()
            if n > 1:
                off_diagonal_avg = (torch.mean(self.sim_matrix) - diagonal_avg/n)*n/(n-1)
            else:
                off_diagonal_avg = 0
            self.sim_matrix = torch.ones([n+1, n+1]).to(self.device)*off_diagonal_avg
            self.sim_matrix[n, n] = diagonal_avg
            self.sim_matrix[0:n, 0:n] = prev_sim_matrix
        else:
            self.sim_matrix = torch.zeros([1, 1]).to(self.device)
            self.cls_grad = torch.cat([self.cls_grad, torch.zeros([1, self.grad_size]).to(self.device)])
            self.self_cls_sim = torch.Tensor([1.0]).to(self.device)
            self.cls_ema_coeff = torch.Tensor([0.0]).to(self.device)
        
        ### fixed similarity ###
        # self.sim_matrix = torch.ones([len(self.memory.cls_list), len(self.memory.cls_list)]).to(self.device)*self.other_cls_sim
        # self.sim_matrix.fill_diagonal_(self.self_cls_sim)
        
        ### for calculating similarity ###
        # len_key = len(self.corr_map.keys())
        # if len_key > 1:
        #     total_corr = 0.0
        #     total_corr_count = 0
        #     self_corr_count = 0
        #     for i in range(len_key):
        #         for j in range(i+1, len_key):
        #             if self.corr_map[i][j] is not None:
        #                 total_corr += self.corr_map[i][j]
        #                 total_corr_count += 1
        #     if total_corr_count >= 1:
        #         self.initial_corr = total_corr / (total_corr_count)
        #     else:
        #         self.initial_corr = 0.0
        #     self_corr = 0.0
        #     for i in range(len_key):
        #         if self.corr_map[i][i] is not None:
        #             self_corr += self.corr_map[i][i]
        #             self_corr_count += 1
        #     if self_corr_count >= 1:
        #         self_corr_avg = total_corr / (self_corr_count+1e-10)
        #     else:
        #         self_corr_avg = 0.5
        # else:
        #     self.initial_corr = None
        
        # for i in range(len_key):
        #     # 모든 class의 avg_corr로 initialize
        #     self.corr_map[i][len_key] = self.initial_corr
        
        # # 자기 자신은 1로 initialize
        # self.corr_map[len_key] = {}
        # if len_key > 1:
        #     self.corr_map[len_key][len_key] = self_corr_avg
        # else:
        #     self.corr_map[len_key][len_key] = None


    def add_new_class(self, class_name, sample=None):
        if hasattr(self.model, 'fc'):
            fc_name = 'fc'
        elif hasattr(self.model, 'head'):
            fc_name = 'head'
        model_fc = getattr(self.model, fc_name)
        # print("!!add_new_class seed", self.rnd_seed)
        self.cls_dict[class_name] = len(self.exposed_classes)
        
        # self.grad_cls_score_mavg[len(self.exposed_classes)] = copy.deepcopy(self.grad_cls_score_mavg_base)
        # self.grad_dict[len(self.exposed_classes)] = copy.deepcopy(self.grad_dict_base)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(model_fc.weight.data)
        prev_bias = copy.deepcopy(model_fc.bias.data)
        setattr(self.model, fc_name, nn.Linear(model_fc.in_features, self.num_learned_class).to(self.device))
        model_fc = getattr(self.model, fc_name)
        with torch.no_grad():
            if self.num_learned_class > 1:
                model_fc.weight[:self.num_learned_class - 1] = prev_weight
                model_fc.bias[:self.num_learned_class - 1] = prev_bias
        # for param in self.optimizer.param_groups[1]['params']:
        #     if param in self.optimizer.state.keys():
        #         del self.optimizer.state[param]
        # del self.optimizer.param_groups[1]
        # self.optimizer.add_param_group({'params': model_fc.parameters()})

        sdict = copy.deepcopy(self.optimizer.state_dict())
        
        fc_params = sdict['param_groups'][1]['params']
        if len(sdict['state']) > 0:
            fc_weight_state = sdict['state'][fc_params[0]]
            fc_bias_state = sdict['state'][fc_params[1]]
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': model_fc.parameters()})
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
                
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

        autograd_hacks.remove_hooks(self.model)
        autograd_hacks.add_hooks(self.model)
        
        
        
        # for unfreezing model

        # initialize with mean
        # if len(self.grad_mavg) >= 2:
        #     self.grad_mavg_base = {key: torch.mean(torch.stack([self.grad_mavg[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavg_base.keys()}
        #     self.grad_mavgsq_base = {key: torch.mean(torch.stack([self.grad_mavgsq[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavgsq_base.keys()}
        #     self.grad_mvar_base = {key: torch.mean(torch.stack([self.grad_mvar[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mvar_base.keys()}
        #
        # self.grad_mavg.append(copy.deepcopy(self.grad_mavg_base))
        # self.grad_mavgsq.append(copy.deepcopy(self.grad_mavgsq_base))
        # self.grad_mvar.append(copy.deepcopy(self.grad_mvar_base))
        # self.grad_criterion.append(copy.deepcopy(self.grad_criterion_base))
        
        
        # ### update similarity map ###
        # len_key = len(self.corr_map.keys())
        # if len_key > 1:
        #     total_corr = 0.0
        #     total_corr_count = 0
        #     for i in range(len_key):
        #         for j in range(i+1, len_key):
        #             total_corr += self.corr_map[i][j]
        #             total_corr_count += 1
        #     self.initial_corr = total_corr / total_corr_count
        # else:
        #     self.initial_corr = None
        #
        # for i in range(len_key):
        #     # 모든 class의 avg_corr로 initialize
        #     self.corr_map[i][len_key] = self.initial_corr
        #
        # # 자기 자신은 1로 initialize
        # self.corr_map[len_key] = {}
        # self.corr_map[len_key][len_key] = None
        #print("self.corr_map")
        #print(self.corr_map)

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            # print("y")
            # print(y)
            #self.before_model_update()
            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)

            if self.train_count > 2:
                if self.unfreeze_rate < 1.0:
                    self.get_freeze_idx(logit.detach(), y)
                if np.random.rand() > self.unfreeze_rate:
                    self.freeze_layers()

            _, preds = logit.topk(self.topk, 1, True, True)


            if self.use_amp:
                self.scaler.scale(loss).backward()
                if 'resnet' in self.model_name or 'cnn' in self.model_name:
                    with torch.cuda.amp.autocast(self.use_amp):
                        autograd_hacks.compute_grad1(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()       
                if 'resnet' in self.model_name or 'cnn' in self.model_name:
                    autograd_hacks.compute_grad1(self.model)
                self.optimizer.step()

            # loss.backward()
            autograd_hacks.compute_grad1(self.model)
            #
            # self.optimizer.step()
            #self.update_gradstat(self.sample_num, y)
            
            if self.sample_num >= 2 and ('resnet' in self.model_name or 'cnn' in self.model_name):
                self.update_correlation(y)

            if not self.frozen:
                self.calculate_fisher()

            autograd_hacks.clear_backprops(self.model)
            
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (len(y) * (self.forward_flops + self.backward_flops))
            else:
                self.total_flops += (len(y) * (self.forward_flops + self.get_backward_flops()))
                
            # print("total_flops", self.total_flops)
            # self.writer.add_scalar(f"train/total_flops", self.total_flops, self.sample_num)

            self.unfreeze_layers()
            self.freeze_idx = []
            self.after_model_update()

        # print("self.corr_map")
        # print(self.corr_map)

        return total_loss / iterations, correct / num_data

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        # self.corr_map_list.append(copy.deepcopy(self.corr_map))
        # self.sample_count_list.append(copy.deepcopy(self.memory.usage_count))
        # self.labels_list.append(copy.deepcopy(self.memory.labels))
        #
        # # store한 애들 저장
        # corr_map_name = "corr_map_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        # sample_count_name = "sample_count_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        # labels_list_name = "labels_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        #
        # # print("corr_map_name", corr_map_name)
        # # print("sample_count_name", sample_count_name)
        # # print("labels_list_name", labels_list_name)
        #
        # with open(corr_map_name, 'wb') as f:
        #     pickle.dump(self.corr_map_list, f, pickle.HIGHEST_PROTOCOL)
        #
        # with open(sample_count_name, 'wb') as f:
        #     pickle.dump(self.sample_count_list, f, pickle.HIGHEST_PROTOCOL)
        #
        # with open(labels_list_name, 'wb') as f:
        #     pickle.dump(self.labels_list, f, pickle.HIGHEST_PROTOCOL)
        
        return super().online_evaluate(test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time)


    def after_model_update(self):
        self.train_count += 1

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        if self.frozen:
            for i in self.freeze_idx:
                backward_flops -= self.blockwise_backward_flops[i]
        return backward_flops

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) # 4
                self.total_flops += (len(logit) * 4) / 10e9
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.total_flops += (len(logit) * 2) / 10e9
        return logit, loss

    def update_memory(self, sample, sample_num=None):
        
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.memory.cls_dict[sample['klass']]] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            idx_to_replace = random.choice(cand_idx)
            self.memory.replace_sample(sample, sample_num, idx_to_replace)
        else:
            self.memory.replace_sample(sample, sample_num)
        '''
        self.memory.cls_seen_count[self.memory.cls_dict[sample['klass']]] += 1
        if len(self.memory.images) >= self.memory_size:
            replace=True
            if self.memory.cls_count[self.memory.cls_dict[sample['klass']]] > self.memory_size // len(self.memory.cls_list):
                j = np.random.randint(0, self.memory.cls_seen_count[self.memory.cls_dict[sample['klass']]])
                if j > self.memory_size // len(self.memory.cls_list):
                    replace = False
            
            if replace:
                label_frequency = copy.deepcopy(self.memory.cls_count)
                label_frequency[self.memory.cls_list.index(sample['klass'])] += 1
                cls_to_replace = np.argmax(np.array(label_frequency))
                cand_idx = self.memory.cls_idx[cls_to_replace]
                idx_to_replace = random.choice(cand_idx)
                self.memory.replace_sample(sample, sample_num, idx_to_replace)
                
        else:
            self.memory.replace_sample(sample, sample_num)
        '''
        

    @torch.no_grad()
    def update_correlation(self, labels):
        n_classes = self.sim_matrix.size(0)
        batch_size = len(labels)
        update_matrix = torch.zeros_like(self.sim_matrix).flatten()
        labelcount_matrix = torch.zeros_like(self.sim_matrix).flatten()
        selected_grads = []
        for n, p in self.model.named_parameters():
            if p.requires_grad is True and p.grad is not None and n in self.selected_mask.keys():
                selected_grads.append(p.grad1.detach().clone().clamp(-1, 1).flatten(start_dim=1)[:, self.selected_mask[n]])
        if len(selected_grads) == 0:
            return None
        stacked_grads = torch.cat(selected_grads, dim=1)
        
        ### cls_grad_similarty
        for i, y in enumerate(labels):
            self.cls_grad[y][-stacked_grads.size(1):] += self.grad_ema_ratio * (stacked_grads[i] - self.cls_grad[y][-stacked_grads.size(1):])
            self.self_cls_sim[y] += self.grad_ema_ratio * (F.cosine_similarity(stacked_grads[i], self.cls_grad[y][-stacked_grads.size(1):], dim=0) - self.self_cls_sim[y])
            self.cls_ema_coeff[y] += self.grad_ema_ratio * (1 - self.cls_ema_coeff[y])
        # self.cls_ema_coeff *= (1-self.grad_ema_ratio) ** (batch_size/n_classes)
        # grad_norm = torch.sqrt(torch.sum(self.cls_grad**2, dim=1))
        # grad_norm_ratio = grad_norm.unsqueeze(1)/(grad_norm.unsqueeze(0)+1e-8)
        ema_coeff = self.cls_ema_coeff.unsqueeze(0)  * self.cls_ema_coeff.unsqueeze(1)
        self.sim_matrix = F.cosine_similarity(self.cls_grad.unsqueeze(1), self.cls_grad.unsqueeze(0), dim=2)
        # self.sim_matrix = torch.diagonal_scatter(self.sim_matrix, self.self_cls_sim)
        self.sim_matrix *= torch.sqrt(self.self_cls_sim.unsqueeze(0) * self.self_cls_sim.unsqueeze(1))
        self.sim_matrix = self.sim_matrix * ema_coeff
        # logging.info(self.sim_matrix)
        # similarity_matrix = F.cosine_similarity(stacked_grads.unsqueeze(1), stacked_grads.unsqueeze(0), dim=2)
        # self.total_flops += stacked_grads.shape[0]*stacked_grads.shape[0]*2*stacked_grads.shape[1]/10e9



        ### simple similarity ###
        '''
        same_labels = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff_labels = ~same_labels
        same_labels.fill_diagonal_(False)
        if torch.any(same_labels):
            self.self_cls_sim += self.grad_ema_ratio*(torch.mean(similarity_matrix[same_labels]) - self.self_cls_sim)
        if torch.any(diff_labels):
            self.other_cls_sim += self.grad_ema_ratio*(torch.mean(similarity_matrix[diff_labels]) - self.other_cls_sim)
        self.sim_matrix = torch.ones([len(self.memory.cls_list), len(self.memory.cls_list)]).to(self.device)*self.other_cls_sim
        self.sim_matrix.fill_diagonal_(self.self_cls_sim)
        n = self.sim_matrix.size(0)
        i1, j1 = torch.triu_indices(n, n, 1)
        self.sim_matrix[j1, i1] = self.sim_matrix[i1, j1]
        '''

        ### similarity implementation v1 ###
        '''
        unique_labels, idxs = torch.unique(labels, sorted=True, return_inverse=True)
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i:]):
                if label1 == label2:
                    num_elements = (idxs == i).sum()
                    if num_elements > 1:
                        label_matrix = similarity_matrix[idxs == i][:, idxs == i+j]
                        non_overlap_idx = torch.triu_indices(num_elements, num_elements, 1)
                        self.sim_matrix[label1][label2] += self.grad_ema_ratio * (
                                    (label_matrix[non_overlap_idx[0], non_overlap_idx[1]]).mean() - self.sim_matrix[label1][label2])
                else:
                    self.sim_matrix[label1][label2] += self.grad_ema_ratio *(
                            (similarity_matrix[idxs == i][:, idxs == i+j]).mean() - self.sim_matrix[label1][label2])
        '''
        '''
        ### similarity implementation v2 ###
        count_matrix = torch.ones_like(similarity_matrix).fill_diagonal_(0.0).flatten()
        index_matrix = (labels.unsqueeze(1) * n_classes + labels.unsqueeze(0)).flatten()
        similarity_matrix = similarity_matrix.fill_diagonal_(0.0).flatten()
        update_matrix = update_matrix.scatter_add(0, index_matrix, similarity_matrix)
        labelcount_matrix = labelcount_matrix.scatter_add(0, index_matrix, count_matrix)
        update_matrix = update_matrix.view(self.sim_matrix.shape)
        labelcount_matrix = labelcount_matrix.view(self.sim_matrix.shape)
        update_mask = labelcount_matrix > 0
        update_matrix[update_mask] /= labelcount_matrix[update_mask]
        self.sim_matrix[update_mask] += self.grad_ema_ratio * (update_matrix[update_mask] - self.sim_matrix[update_mask])
        '''

    def get_layer_number(self, n):
        name = n.split('.')
        if name[0] == 'initial':
            return 0
        elif 'group' in name[0]:
            group_num = int(name[0][-1])
            block_num = int(name[2][-1])
            return group_num * 2 + block_num - 1

    @torch.no_grad()
    def calculate_fisher(self):
        # block_grad_size = [0.0 for _ in range(self.num_blocks)]
        # block_input_size = [0.0 for _ in range(self.num_blocks)]
        # block_grad_list = [[] for _ in range(self.num_blocks)]
        
        block_fisher = [0.0 for _ in range(self.num_blocks)]
        for i, block_name in enumerate(self.block_names[:-1]):
            for subblock_name in block_name:
                get_attr = attrgetter(subblock_name)
                block = get_attr(self.model)
                block_grad = []
                block_input = []
                for n, p in block.named_parameters():
                    if p.requires_grad is True and p.grad is not None:
                        if not p.grad.isnan().any():
                            block_fisher[i] += (p.grad.clone().detach().clamp(-1, 1) ** 2).sum().item()
                            if self.unfreeze_rate < 1:
                                self.total_flops += len(p.grad.clone().detach().flatten())*2 / 10e9
                            # if 'resnet' in self.model_name and i < self.num_blocks:
                            #     group_name = '.'.join((subblock_name+'.'+n).split('.')[:-3])
                            #     get_group_attr = attrgetter(group_name)
                                
                            #     block_fisher[i] += (p.grad.clone().detach().clamp(-1, 1) ** 2).sum().item()
                            #     if self.unfreeze_rate < 1:
                            #         self.total_flops += len(p.grad.clone().detach().flatten())*2 / 10e9
                            #     # block_input.append(get_group_attr(self.model).input.clone().detach().flatten())
                            # else:
                            #     block_fisher[i] += (p.grad.clone().detach().clamp(-1, 1) ** 2).sum().item()
                            #     if self.unfreeze_rate < 1:
                            #         self.total_flops += len(p.grad.clone().detach().flatten())*2 / 10e9
                            # block_grad.append(p.grad.clone().detach().flatten())
                # block_grad_size[i] = torch.mean(torch.abs(torch.cat(block_grad))).item()
                # block_input_size[i] = torch.mean(torch.abs(torch.cat(block_input))).item()
                # block_grad_list[i] = (copy.deepcopy(block_grad))

        
        # logging.info(f'Grad Size: {block_grad_size}')
        # self.freezing_stats['grad_size'].append(block_grad_size)
        # self.freezing_stats['input_size'].append(block_input_size)
        
        
        # with open(f'{self.save_path}_freezing.pkl', 'wb') as fp:
        #     pickle.dump(self.freezing_stats, fp)
            
        # with open(f'{self.save_path}_grad.pkl', 'wb') as fp:
        #     pickle.dump(block_grad_list, fp)
        
        for i in range(self.num_blocks):
            if i not in self.freeze_idx or not self.frozen:
                self.fisher[i] += self.fisher_ema_ratio * (block_fisher[i] - self.fisher[i])
        # print(group_fisher)
        self.total_fisher = sum(self.fisher)
        self.cumulative_fisher = [sum(self.fisher[0:i+1]) for i in range(self.num_blocks)]
        # print(self.fisher, self.layerwise_backward_flops)

    def get_flops_parameter(self):
        super().get_flops_parameter()
        self.cumulative_backward_flops = [sum(self.blockwise_backward_flops[0:i+1]) for i in range(self.num_blocks)]
        print("self.cumulative_backward_flops")
        print(self.cumulative_backward_flops)

    @torch.no_grad()
    def get_freeze_idx(self, logit, label):
        if hasattr(self.model, 'fc'):
            fc_name = 'fc'
        elif hasattr(self.model, 'head'):
            fc_name = 'head'
        model_fc = getattr(self.model, fc_name)
        grad = self.get_grad(logit, label, model_fc.weight)
        last_grad = (grad ** 2 ).sum().item()
        if self.unfreeze_rate < 1:
            self.total_flops += len(grad.clone().detach().flatten())/10e9
        batch_freeze_score = last_grad/(self.last_grad_mean+1e-10)
        self.last_grad_mean += self.fisher_ema_ratio * (last_grad - self.last_grad_mean)
        freeze_score = []
        freeze_score.append(1)
        # if 'noinitial' in self.note:
        freeze_score.append(1)
        total_model_flops = self.total_model_flops - self.blockwise_forward_flops[0] - self.blockwise_backward_flops[0]
        cumulative_backward_flops = [sum(self.blockwise_backward_flops[1:i+1]) for i in range(1, self.num_blocks)]
        total_fisher = sum(self.fisher[1:])
        cumulative_fisher = [sum(self.fisher[1:i+1]) for i in range(1, self.num_blocks)]
        
        for i in range(self.num_blocks-1):
            freeze_score.append(total_model_flops / (total_model_flops - cumulative_backward_flops[i]) * (
                        total_fisher - cumulative_fisher[i]) / (total_fisher + 1e-10))
        max_score = max(freeze_score)
        modified_score = []
        modified_score.append(batch_freeze_score)
        modified_score.append(batch_freeze_score)
        for i in range(self.num_blocks -1):
            modified_score.append(batch_freeze_score*(total_fisher - cumulative_fisher[i])/(total_fisher + 1e-10) + cumulative_backward_flops[i]/total_model_flops * max_score)
        optimal_freeze = np.argmax(modified_score)
        # else:
        #     for i in range(self.num_blocks):
        #         freeze_score.append(self.total_model_flops / (self.total_model_flops - self.cumulative_backward_flops[i]) * (
        #                     self.total_fisher - self.cumulative_fisher[i]) / (self.total_fisher + 1e-10))
        #     max_score = max(freeze_score)
        #     modified_score = []
        #     modified_score.append(batch_freeze_score)
        #     for i in range(self.num_blocks):
        #         modified_score.append(batch_freeze_score*(self.total_fisher - self.cumulative_fisher[i])/(self.total_fisher + 1e-10) + self.cumulative_backward_flops[i]/self.total_model_flops * max_score)
        #     optimal_freeze = np.argmax(modified_score)
        
        # self.freezing_stats['i/c'].append(freeze_score)
        # self.freezing_stats['bi/c'].append(modified_score)
        # self.freezing_stats['grad_magnitude'].append(batch_freeze_score)
        # with open(f'{self.save_path}_freezing.pkl', 'wb') as fp:
        #     pickle.dump(self.freezing_stats, fp)
        
        logging.info(f'I/C: {freeze_score} \n BI/C: {modified_score} \n Grad_Magnitude: {batch_freeze_score}')
        logging.info(f'Iter: {self.sample_num} Freeze: {optimal_freeze}')
        self.freeze_idx = list(range(self.num_blocks))[0:optimal_freeze]

    @torch.no_grad()
    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)

        front = (prob - oh_label).shape
        back = weight.shape
        if self.unfreeze_rate < 1:
            self.total_flops += ((front[0] * back[1] * (2 * front[1] - 1)) / 10e9)
        
        with torch.cuda.amp.autocast(self.use_amp):
            last_grad = torch.matmul((prob - oh_label), weight)
        
        return last_grad


class OurMemory(MemoryBase):
    def __init__(self, memory_size, T, count_decay_ratio, k_coeff, device='cpu'):
        super().__init__(memory_size)
        self.T = T
        self.k_coeff = k_coeff
        self.entered_time = []
        self.count_decay_ratio = count_decay_ratio
        self.device = device
        self.usage_count = torch.Tensor([]).to(self.device)
        self.class_usage_count = torch.Tensor([]).to(self.device)
        self.cls_seen_count = defaultdict(int)
        self.retrieval_stats = {'sim_matrix':[], 'freq': [], 'prob': [], 'cls_freq_sum': [], 'cls_freq_avg': [], 'cls_prob_sum': [], 'cls_prob_avg':[]}


    def replace_sample(self, sample, sample_num, idx=None):
        super().replace_sample(sample, idx)
        #self.usage_count = torch.cat([self.usage_count, torch.zeros(1).to(self.device)])
        if idx is None:
            self.usage_count = torch.cat([self.usage_count, torch.zeros(1).to(self.device)])
        else:
            self.usage_count[idx] = 0
        self.entered_time.append(sample_num)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = torch.cat([self.class_usage_count, torch.zeros(1).to(self.device)])

    # balanced probability retrieval
    def balanced_retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        cls_idx = np.random.choice(len(self.cls_list), sample_size)
        for cls in cls_idx:
            i = np.random.choice(self.cls_idx[cls], 1)[0]
            memory_batch.append(self.images[i])
            self.usage_count[i] += 1
            self.class_usage_count[self.labels[i]] += 1
        return memory_batch

    
    def retrieval(self, size, similarity_matrix=None):
        # for use count decaying
        if len(self.images) > size:
            self.count_decay_ratio = size / (len(self.images)*self.k_coeff)  #(self.k_coeff / (len(self.images)*self.count_decay_ratio))
            # print("count_decay_ratio", self.count_decay_ratio)
            self.usage_count *= (1-self.count_decay_ratio)
            self.class_usage_count *= (1-self.count_decay_ratio)
        
        if similarity_matrix is None:
            return self.balanced_retrieval(size)
        else:
            sample_size = min(size, len(self.images))
            weight = self.get_similarity_weight(similarity_matrix)
            sample_idx = np.random.choice(len(self.images), sample_size, p=weight, replace=False)
            memory_batch = list(np.array(self.images)[sample_idx])
            for i in sample_idx:
                self.usage_count[i] += 1
                self.class_usage_count[self.labels[i]] += 1
            return memory_batch
        
    def get_similarity_weight(self, sim_matrix):
        # self.retrieval_stats['sim_matrix'].append(sim_matrix.cpu().numpy())
        n_cls = len(self.cls_list)
        
        self_score = torch.ones(n_cls).to(self.device)
        self_score -= sim_matrix.diag()

        cls_score_sum = (sim_matrix * self.class_usage_count).sum(dim=1)

        sample_score = cls_score_sum[torch.LongTensor(self.labels).to(self.device)] + self.usage_count.to(self.device)*self_score[torch.LongTensor(self.labels).to(self.device)]
        sample_score /= len(self.images)
        sample_score = torch.clamp(sample_score, min=0)
        # self.retrieval_stats['freq'].append(sample_score.cpu().numpy())
        
        cls_score_sum = torch.zeros(n_cls).to(self.device)
        # for i in range(len(self.images)):
        #     cls_score_sum[self.labels[i]] += sample_score[i]
        # self.retrieval_stats['cls_freq_sum'].append(cls_score_sum.cpu().numpy())
        # self.retrieval_stats['cls_freq_avg'].append((cls_score_sum.cpu()/torch.Tensor(self.cls_count)).numpy())

        prob = F.softmax(-sample_score/self.T, dim=0)
        # self.retrieval_stats['prob'].append(prob.cpu().numpy())        
        # cls_prob_sum = torch.zeros(n_cls).to(self.device)
        # for i in range(len(self.images)):
        #     cls_prob_sum[self.labels[i]] += prob[i]
        # self.retrieval_stats['cls_prob_sum'].append(cls_prob_sum.cpu().numpy())
        # self.retrieval_stats['cls_prob_avg'].append((cls_prob_sum.cpu()/torch.Tensor(self.cls_count)).numpy())

        return prob.cpu().numpy()
    
    
