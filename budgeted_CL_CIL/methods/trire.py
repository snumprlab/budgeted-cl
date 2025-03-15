# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import torch.nn.functional as F
import numpy as np
import torch
import re
import torch.nn as nn
from utils.train_utils import re_init_weights, create_dense_mask_0, select_model, test_sparsity, copy_paste_fc
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from methods.er_new import ER
import math
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class TriRE(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.beta = kwargs["tri_beta"]
        kwargs["kwinner"] = True
        self.waiting_batch_index = []
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        # self.model = select_model(self.model_name, self.dataset, num_classes = 1, channel_constant=kwargs["channel_constant"], kwinner=True).to(self.device)
        self.task_samples=0
        self.lr_fl = self.lr
        self.lr_sl = self.lr / 10
        self.global_step = 0
        self.ema_update_freq = kwargs["stable_model_update_freq"]
        self.ema_model_alpha = kwargs["stable_model_alpha"]
        ### 필요한 mask : model_mask_current, model_sparse_set, model_epoch_k, ema_model
        self.EXCLUDE_LAYERS_START_WITH = ['initial', 'group1', 'fc']
        self.EXCLUDE_LAYERS_CONTAINING = ['shortcut', 'activation', 'downsample'] #['conv1', 'shortcut', 'fc', 'group1', 'initial', 'downsample', 'activation']
        self.consistency_loss = nn.MSELoss(reduction='none')
        self.model_mask_current = create_dense_mask_0(deepcopy(self.model), self.device, value=0)
        self.model_sparse_set = create_dense_mask_0(deepcopy(self.model), self.device, value=0)
        self.model_copy = create_dense_mask_0(deepcopy(self.model), self.device, value=1)
        self.model_grad = create_dense_mask_0(deepcopy(self.model), self.device, value=0)
        self.sparse_model_grad = create_dense_mask_0(deepcopy(self.model), self.device, value=0)    # for sparse model
        self.model_epoch_k = create_dense_mask_0(deepcopy(self.model), self.device, value=1)    # rewind
        self.ema_model = deepcopy(self.model).to(self.device)
        self.model_init = deepcopy(self.model).to(self.device)
        self.reg_weight = kwargs["reg_weight"]
        self.mask_cum_sparse_weights = kwargs["mask_cum_sparse_weights"]
        self.mask_non_sparse_weights = kwargs["mask_non_sparse_weights"]
        self.train_budget_1 = kwargs["train_budget_1"]
        self.train_budget_2 = kwargs["train_budget_2"]
        self.forget_perc = kwargs["forget_perc"]
        self.sparsity = kwargs["sparsity"]
        self.reparameterize = kwargs["reparameterize"]
        self.pruning_technique = kwargs["pruning_technique"]
        self.kwinner_sparsity = kwargs["kwinner_sparsity"]
        self.het_drop = kwargs["het_drop"]
        self.use_het_drop = kwargs["use_het_drop"]
        self.reinit_technique = kwargs["reinit_technique"]
        self.reset_act_counters = kwargs["reset_act_counters"]
        self.kwinner_mask = {}
        self.gradient_masks = {}
        self.criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
        for name, module in self.model.named_modules():
            if 'kwinner' in name:
                print("kwinner!!", name)

    def adjust_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            print("learning rate: ", param_group['lr'])

    def measure_amount_of_sparsity(self):
        with torch.no_grad():
            N = 0
            total = 0
            for name, mask_set in self.model_sparse_set.named_parameters():
                total += torch.numel(mask_set.data)
                if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                    N += torch.numel(mask_set.data)
                elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                    N += torch.numel(mask_set.data)
                else:
                    N += torch.count_nonzero(mask_set.data)
            overlap = N * 100 / total if total > 0 else 0
            print("@@@@@@@@@@@@@@@@@@")
            print("{}% of weights from the net"
                  " were reused in the sparse set!".format(overlap))
            print("@@@@@@@@@@@@@@@@@@")
        return overlap

    def online_after_task(self):
        self.measure_amount_of_sparsity()
        if self.reset_act_counters:
            for name, module in self.model.named_modules():
                if 'kwinner' in name:
                    if hasattr(module, 'act_count'):
                        module.act_count *= 0

    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        self.model_init.fc = copy.deepcopy(self.model.fc)
        # self.model_mask_current = create_dense_mask_0(deepcopy(self.model), self.device, value=0, fc=True)
        # self.model_sparse_set = create_dense_mask_0(deepcopy(self.model), self.device, value=0, fc=True)
        # self.model_copy = create_dense_mask_0(deepcopy(self.model), self.device, value=1, fc=True)
        # self.model_grad = create_dense_mask_0(deepcopy(self.model), self.device, value=0, fc=True)
        # self.sparse_model_grad = create_dense_mask_0(deepcopy(self.model), self.device, value=0, fc=True)    # for sparse model
        # self.model_epoch_k = create_dense_mask_0(deepcopy(self.model), self.device, value=1, fc=True)    # rewind
        zero_model = create_dense_mask_0(deepcopy(self.model), self.device, value=0)
        one_model = create_dense_mask_0(deepcopy(self.model), self.device, value=1)
        self.model_mask_current.fc = copy_paste_fc(deepcopy(zero_model.fc), deepcopy(self.model_mask_current.fc))
        self.model_sparse_set.fc = copy_paste_fc(deepcopy(zero_model.fc), deepcopy(self.model_sparse_set.fc))
        self.model_copy.fc = copy_paste_fc(deepcopy(one_model.fc), deepcopy(self.model_copy.fc))
        self.model_grad.fc = copy_paste_fc(deepcopy(zero_model.fc), deepcopy(self.model_grad.fc))
        self.sparse_model_grad.fc = copy_paste_fc(deepcopy(zero_model.fc), deepcopy(self.sparse_model_grad.fc))
        self.model_epoch_k.fc = copy_paste_fc(deepcopy(one_model.fc), deepcopy(self.model_epoch_k.fc))

    def online_before_task(self, task):
        self.stage = 1
        self.task_samples = 0
        self.task_id = task
        self.budget_1 = int(self.samples_per_task * self.train_budget_1)
        self.budget_2 = int(self.samples_per_task * self.train_budget_2)
        self.budget_3 = self.samples_per_task  - int(self.samples_per_task * self.train_budget_1) - int(self.samples_per_task * self.train_budget_2)
        self.boundary = [self.budget_1, self.budget_1 + self.budget_2]

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        self.task_samples += 1
        if self.task_samples in self.boundary:
            self.stage+=1
            if self.stage == 2:
                #  Create a new sparse model for the current task
                self.extract_new_sparse_model()

                # Create a copy of the weights before masking out current non-sparse weights
                self.model_copy = deepcopy(self.model)
            elif self.stage == 3:
                self.update_sparse_set()
                if self.reparameterize:
                    self.reparameterize_non_sparse()
            lr = self.lr_sl if self.stage == 2 else self.lr_fl
            self.adjust_learning_rate(lr)
            
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates), stage=self.stage)
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def reparameterize_non_sparse(self):
        with torch.no_grad():
            for (name, param), mask_param, param_rewind in \
                    zip(self.model.named_parameters(),
                        self.model_sparse_set.parameters(),
                        self.model_epoch_k.parameters()):
                if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                    continue
                elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                    continue
                mask = torch.zeros(mask_param.data.shape, device=self.device)
                mask[mask_param.data == 1] = 1
                param.data = param.data * mask
                param_rewind[mask == 1] = 0
                param.data += param_rewind


    def update_sparse_set(self):
        # add current important features to cumulative sparse set, overlapping ones will be overwritten
        total = 0
        num_zero = 0
        num_non_zero = 0
        for mask_current, mask_set in zip(self.model_mask_current.parameters(), self.model_sparse_set.parameters()):
            mask_set.data[mask_current.data == 1] = 1
            total += torch.numel(mask_current)
            num_zero += torch.sum(mask_set.data == 0)
            num_non_zero += torch.sum(mask_set.data == 1)
            
        # sparse set ratio
        print("### Sparse Set Ratio ###")
        print(num_non_zero / total)
        print()
        
        

    def extract_new_sparse_model(self):
        # Re-init the current task mask
        self.model_mask_current = create_dense_mask_0(deepcopy(self.model), self.device, value=0)
        w_grad = None
        # Extract sparse model for the current task
        start_idx = 0
        total_num = 0
        num_zero = 0
        
        with torch.no_grad():
            for (name, param), param_mask in zip(self.model.named_parameters(), self.model_mask_current.parameters()):
                if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                    start_idx += torch.numel(param.data)
                    continue
                elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                    start_idx += torch.numel(param.data)
                    continue
                N = param.data.shape[0]
                k = math.floor(N * self.kwinner_sparsity)
                shape = param.shape
                weight = param.data.detach().clone().cpu().numpy()
                if not param.grad is None:
                    grad_copy = copy.deepcopy(param.grad)
                    w_grad = grad_copy.detach().clone().cpu().numpy()
                weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
                if not w_grad is None:
                    grad_temp = np.abs(w_grad)
                    adjusted_importance = weight_temp + grad_temp
                    adjusted_importance = torch.from_numpy(adjusted_importance).to(self.device)
                else:
                    adjusted_importance = weight_temp
                    adjusted_importance = torch.from_numpy(adjusted_importance).to(self.device)
                self.structured_pruning(k, name, shape, param_mask, adjusted_importance)
                total_num += torch.numel(param.data)
                num_zero += torch.sum(param_mask)
            print("###Current Sparse Ratio###")
            print(num_zero / total_num)
            print()


    def structured_pruning(self, k, name, shape, param_mask, adjusted_importance):
        # Use k-winner take all mask first
        tmp = [int(s) for s in re.findall(r'\d+', name)]
        mask_name = "self.model.group{}.blocks.block{}.kwinner{}.act_count".format(tmp[0], tmp[1], tmp[2])
        kwinner_mask = eval(mask_name)

        if self.use_het_drop:
            if "group2" in mask_name:
                het_drop = self.het_drop
            elif "group3" in mask_name:
                het_drop = self.het_drop * 1.5
            else:
                het_drop = self.het_drop * 3.0

            prob_drop = torch.zeros(kwinner_mask.shape, device=self.device)
            for i, element in enumerate(kwinner_mask):
                norm = -(element / torch.max(kwinner_mask))
                prob_drop[i] = torch.exp(norm * het_drop)
            indices_1 = torch.where(prob_drop > torch.mean(prob_drop))[0]

        mask = torch.empty(shape[0], device=self.device).fill_(float('-inf'))
        mask.scatter_(0, indices_1, 1)
        # Log the mask
        self.kwinner_mask[(self.task_id, mask_name)] = mask

        # Prune the layer based on k-winner mask
        num_filters = shape[0]
        if 'conv' in name and len(shape) > 1:
            for filter_idx in range(num_filters):
                if mask[filter_idx] < 0:
                    adjusted_importance[filter_idx, :, :, :] = mask[filter_idx]

            N = (mask == 1).sum() * torch.numel(adjusted_importance[0, :, :, :])
            l = math.floor(N * self.sparsity)
            indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]

            pruning_mask = torch.zeros(torch.numel(adjusted_importance), device=self.device)
            pruning_mask.scatter_(0, indices_2, 1)
            param_mask.data += pruning_mask.reshape(shape)
            self.gradient_masks[name] = param_mask.data.cuda()
        elif 'fc' in name and len(shape) > 1:                       # For DIL scenario
                N = (mask == 1).sum() * torch.numel(adjusted_importance[0, :])
                l = math.floor(N * self.sparsity)
                indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]

                pruning_mask = torch.zeros(torch.numel(adjusted_importance), device=self.device)
                pruning_mask.scatter_(0, indices_2, 1)
                param_mask.data += pruning_mask.reshape(shape)
                self.gradient_masks[name] = param_mask.data.cuda()
        else:
            adjusted_importance[mask < 0] = float('-inf')
            N = (mask == 1).sum()
            l = math.floor(N * self.sparsity)
            indices_2 = torch.topk(torch.flatten(adjusted_importance), l)[1]
            param_mask.data[indices_2] = 1
        print(name, torch.sum(param_mask.data == 0) / (torch.sum(param_mask.data == 0) + torch.sum(param_mask.data == 1)))

    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch, self.waiting_batch_index.pop(0)
    
    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            mem_data, mem_indices = self.memory.retrieval(self.memory_batch_size, return_index=True)
            self.waiting_batch.append(self.temp_future_batch + mem_data)
            self.waiting_batch_index.append(mem_indices)

    def update_ema_model_variables(self):
        alpha_ema = min(1 - 1 / (self.global_step + 1), self.ema_model_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha_ema).add_(1 - alpha_ema, param.data)
            
    def update_non_cumulative_sparse(self, observe=False):
        with torch.no_grad():
            # update weights NOT in cumulative sparse set. Those in sparse set are not updated.
            for (name, param_net), param_sparse, param_grad_copy in \
                    zip(self.model.named_parameters(),
                        self.model_sparse_set.parameters(),
                        self.model_grad.parameters()):
                param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                # Exclude final linear layer weights from masking
                if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                    param_grad_copy.grad = param_net.grad.clone()
                    continue
                elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                    param_grad_copy.grad = param_net.grad.clone()
                    continue
                param_lr[param_sparse == 1] = 0
                param_grad_copy.grad = param_net.grad.clone() * param_lr
                # For rewind phase
                if observe:
                    param_net.grad += param_grad_copy.grad.clone()

    def update_cumulative_sparse(self):
        with torch.no_grad():
            # update weights in cumulative sparse set. Those NOT in sparse set are not updated.
            for (name, param_net), param_sparse, param_grad_copy in \
                    zip(self.model.named_parameters(),
                        self.model_sparse_set.parameters(),
                        self.model_grad.parameters()):
                # Exclude final linear layer weights from masking
                if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                    param_net.grad += param_grad_copy.grad.clone()
                    continue
                elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                    param_net.grad += param_grad_copy.grad.clone()
                    continue
                param_net.grad *= param_sparse
                param_net.grad += param_grad_copy.grad.clone()
                
    # def maskout_non_sparse(self, is_sparse_m=False):
    #     for (name, param_net), param_current, param_sparse in zip(self.model.named_parameters(),
    #                                                 self.model_mask_current.parameters(),
    #                                                 self.model_sparse_set.parameters()):
    #         if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
    #             continue
    #         elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
    #             continue
    #         mask = torch.zeros(param_current.data.shape, device=self.device)
    #         mask[param_current.data == 1] = 1
    #         mask[param_sparse.data == 1] = 1
    #         param_net.data = param_net.data * mask

    def update_overlapping_sparse(self, is_sparse_m=False):
        with torch.no_grad():
            if is_sparse_m:
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.sparse_model.named_parameters(),
                            self.model_mask_current.parameters(),
                            self.model_sparse_set.parameters(),
                            self.sparse_model_grad.parameters()):
                    # Exclude some layers from masking
                    if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    param_net.grad = param_net.grad * param_sparse  # all f_\theta \in S
                    param_net.grad += param_grad_copy.grad.clone()

            else:
                # Update gradients of current sparse mask that are in sparse set
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.model.named_parameters(),
                            self.model_mask_current.parameters(),
                            self.model_sparse_set.parameters(),
                            self.model_grad.parameters()):
                    # Exclude some layers from masking
                    if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                        param_net.grad += param_grad_copy.grad.clone()
                        continue
                    param_net.grad = param_net.grad * param_sparse  # all f_\theta \in S
                    param_net.grad += param_grad_copy.grad.clone()

    def maskout_cum_sparse(self):
        # Mask out weights not in the current sparse mask
        for (name, param_net), param_current, param_sparse in zip(self.model.named_parameters(),
                                                   self.model_mask_current.parameters(),
                                                   self.model_sparse_set.parameters()):
            if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                continue
            elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                continue
            mask = torch.ones(param_current.data.shape, device=self.device)
            mask[param_current.data == 1] = 0
            mask[param_sparse.data == 1] = 0
            param_net.data = param_net.data * mask                                    

    def model_forward_stage3(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.model(x)

        if self.mask_cum_sparse_weights:
            self.maskout_cum_sparse()

        loss_scores = self.criterion(outputs, y)

        loss = loss_scores.mean()
        assert not torch.isnan(loss)
        loss.backward()

        self.update_non_cumulative_sparse(observe=True)

        self.optimizer.step()
        self.model_grad.zero_grad()

        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()
        self.total_flops += (len(x) * (self.backward_flops + self.forward_flops))
        return outputs, loss


    def model_forward_stage2(self, x, y):
        self.ema_model.train()
        loss_b = torch.tensor(0)
        self.optimizer.zero_grad()

        outputs = self.model(x[:self.temp_batch_size])
        self.total_flops += (len(x[:self.temp_batch_size]) * self.forward_flops)
        loss_scores = self.criterion(outputs, y[:self.temp_batch_size])

        loss = loss_scores.mean()
        assert not torch.isnan(loss)
        loss.backward()
        # Update gradients of current sparse mask that are NOT in sparse set
        self.update_non_overlapping_sparse()

        if len(x) > self.temp_batch_size:
            self.optimizer.zero_grad()

            buf_outputs = self.model(x[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2])
            buf_outputs_ema = self.ema_model(x[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2])
            self.total_flops += (2 * len(y[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2]) * self.forward_flops)
            loss_b = self.reg_weight * F.mse_loss(buf_outputs, buf_outputs_ema.detach())

            buf_outputs2 = self.model(x[self.temp_batch_size + self.memory_batch_size//2:])
            self.total_flops += (len(y[self.temp_batch_size + self.memory_batch_size//2:]) * self.forward_flops)
            loss_b_scores = self.beta * self.criterion(buf_outputs2, y[self.temp_batch_size + self.memory_batch_size//2:])
            loss_b += loss_b_scores.mean()

            #self.memory.update_scores(buf_indexes_2, -loss_b_scores.detach())

            assert not torch.isnan(loss_b)
            loss_b.backward()
            outputs = torch.cat([outputs, buf_outputs, buf_outputs2], dim=0)

            # Update gradients of current sparse mask that are in sparse set
            self.update_overlapping_sparse()

        self.optimizer.step()
        self.model.zero_grad()
        self.total_flops += (len(x) * self.backward_flops)
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_model_variables()

        return outputs, loss_b + loss

    def model_forward_stage1(self, x, y):
        logit = self.model(x[:self.temp_batch_size])
        loss = self.criterion(logit, y[:self.temp_batch_size]).mean()
        loss.backward()
        self.total_flops += (len(x[:self.temp_batch_size]) * self.forward_flops)
        
        self.update_non_cumulative_sparse() ### Sparse한 model에 속해 있는 param의 grad 모두 0으로 
        loss_b = None
        
        if len(x) > self.temp_batch_size:
            self.optimizer.zero_grad()
            buf_outputs = self.model(x[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2])
            buf_outputs_ema = self.ema_model(x[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2])
            self.total_flops += (2 * len(y[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size//2]) * self.forward_flops)
            loss_b = self.reg_weight * F.mse_loss(buf_outputs, buf_outputs_ema.detach())

            buf_outputs2 = self.model(x[self.temp_batch_size + self.memory_batch_size//2:])
            self.total_flops += (len(y[self.temp_batch_size + self.memory_batch_size//2:]) * self.forward_flops)
            loss_b_scores = self.criterion(buf_outputs2, y[self.temp_batch_size + self.memory_batch_size//2:])
            loss_b += loss_b_scores.mean()

            assert not torch.isnan(loss_b)
            loss_b.backward()

            self.update_cumulative_sparse() ### Sparse한 model에 속해 있는 param의 grad만 살림
            logit = torch.cat([logit, buf_outputs, buf_outputs2], dim=0)
            
        self.total_flops += (len(x) * self.backward_flops)
        self.optimizer.step()
        self.model_grad.zero_grad()
        
        return logit, loss + loss_b if loss_b is not None else loss

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

                loss = criterion(logit, y).mean()
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


    def update_non_overlapping_sparse(self, is_sparse_m=False):
        with torch.no_grad():
            if is_sparse_m:
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.sparse_model.named_parameters(),
                            self.model_mask_current.parameters(),
                            self.model_sparse_set.parameters(),
                            self.sparse_model_grad.parameters()):
                    param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                    # Exclude final linear layer weights from masking
                    if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    # no update using overlapping weights' gradients
                    param_lr[param_current == param_sparse] = 0
                    # No gradient for weights that are not part of current sparse mask
                    param_lr[param_current == 0] = 0
                    param_grad_copy.grad = param_net.grad.clone() * param_lr

            else:
                # Update gradients of current sparse mask that are NOT in sparse set
                for (name, param_net), param_current, param_sparse, param_grad_copy in \
                        zip(self.model.named_parameters(),
                            self.model_mask_current.parameters(),
                            self.model_sparse_set.parameters(),
                            self.model_grad.parameters()):
                    param_lr = torch.ones(param_sparse.data.shape, device=self.device)
                    # Exclude final linear layer weights from masking
                    if any(name.startswith(layer) for layer in self.EXCLUDE_LAYERS_START_WITH):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    elif any(layer in name for layer in self.EXCLUDE_LAYERS_CONTAINING):
                        param_grad_copy.grad = param_net.grad.clone()
                        continue
                    # no update using overlapping weights' gradients
                    param_lr[param_current == param_sparse] = 0
                    # No gradient for weights that are not part of current sparse mask
                    param_lr[param_current == 0] = 0
                    param_grad_copy.grad = param_net.grad.clone() * param_lr

  
    def online_train(self, iterations=1, stage=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data, indices = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            self.before_model_update()

            self.optimizer.zero_grad()

            if stage == 1:
                logit, loss = self.model_forward_stage1(x, y)
                if self.task_samples == self.forget_perc * self.budget_1:
                    self.model_epoch_k = deepcopy(self.model)  # Weight saved for Rewind step
            elif stage == 2:
                logit, loss = self.model_forward_stage2(x, y)
            elif stage == 3:
                logit, loss = self.model_forward_stage3(x, y)
            
            _, preds = logit.topk(self.topk, 1, True, True)
            self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data
