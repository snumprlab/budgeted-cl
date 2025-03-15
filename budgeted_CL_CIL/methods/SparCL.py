# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from utils import utils_pr
import numpy as np
import torch
import torch.nn as nn
import pickle5 as pickle
import os
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import CrossEntropyLossMaybeSmooth
from methods.er_new import ER
from utils.utils_pr import weight_pruning, weight_growing, compute_forgetting_statistics, sort_examples_by_forgetting
import sys
from utils.testers import *
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class SparCL(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        
        self.waiting_batch_index = []
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        print("smh")
        for name, W in (self.model.named_parameters()):
            print(name, W.shape)
        # self.criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
        self.kwargs = kwargs
        self.task_iter = 0
        self.gradient_efficient = kwargs["gradient_efficient"]
        self.gradient_efficient_mix = kwargs["gradient_efficient_mix"]
        self.sample_frequency = kwargs["sample_frequency"]
        self.pattern = self.kwargs["retrain_mask_pattern"]
        self.sparsity = self.kwargs["retrain_mask_sparsity"]
        self.seed = self.kwargs["retrain_mask_seed"]
        self.sp_mask_update_freq = kwargs["sp_mask_update_freq"]
        self.update_init_method = kwargs["sp_update_init_method"]
        self.buffer_weight = kwargs["buffer_weight"]
        self.seq_gap_layer_indices = None
        self.masks = {}
        self.gradient_masks = {}
        self.masked_layers = {}
        self.configs, self.prune_ratios = utils_pr.load_configs(self.model, kwargs["sp_config_file"], logger)

        self.prune_init()
        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None

        if "fixed_layers" in self.configs:
            self.fixed_layers = self.configs['fixed_layers']
        else:
            self.fixed_layers = None
        self.fixed_layers_save = {}

        if self.kwargs["upper_bound"] != None:
            self.upper_bound = self.kwargs["upper_bound"]
            print("!!!!! upper_bound", self.upper_bound)
        else:
            self.upper_bound = None

        if self.kwargs["lower_bound"] != None:
            self.lower_bound = self.kwargs["lower_bound"]
            print("!!!!! lower_bound", self.lower_bound)
        else:
            self.lower_bound = None

        if self.kwargs["mask_update_decay_epoch"] != None:
            self.mask_update_decay_epoch = self.kwargs["mask_update_decay_epoch"]
        else:
            self.mask_update_decay_epoch = None

        self.prune_apply_masks()
        self.prune_print_sparsity()
        test_sparsity(self.model, column=False, channel=False, filter=False, kernel=False)
        self.init_mask()

    # def online_before_task(self, task):
    #     self.example_stats_train = {}  
    #     self.task_iter = 0


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        # self.prune_apply_masks()
        test_sparsity(self.model, column=False, channel=False, filter=False, kernel=False)
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

    def online_after_task(self):
        self.task_iter = 0


    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            
        if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
            self.exposed_domains.append(sample["time"])
            
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

    # def model_forward(self, x, y):
    #     logit = self.model(x)
    #     ce_loss = self.criterion(logit, y)
    #     stream_loss = ce_loss[:self.temp_batch_size]
    #     memory_loss = ce_loss[self.temp_batch_size:]
    #     loss = stream_loss.mean() + self.buffer_weight * memory_loss.mean()
    #     self.total_flops += (len(y) * self.forward_flops)
    #     return logit, loss


    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch, self.waiting_batch_index.pop(0)

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
            mem_data, mem_indices = self.memory.retrieval(self.memory_batch_size, return_index=True)
            self.waiting_batch.append(self.temp_future_batch + mem_data)
            self.waiting_batch_index.append(mem_indices)
    
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        
        for i in range(iterations):
            self.prune_update(self.task_iter)
            self.task_iter += 1
            data, indices = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            self.before_model_update()
            self.optimizer.zero_grad()

            # #########remove data at 25 epoch, update dataset ######
            # if self.task_iter > 0 and self.task_iter % self.sp_mask_update_freq == 0 and self.task_iter <= self.remove_data_epoch:
                
            #     print('self.task_iter', self.task_iter)

            #     unlearned_per_presentation_all, first_learned_all = [], []

            #     _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(example_stats_train, int(args.epochs/dataset.N_TASKS))
            #     print('unlearned_per_presentation', len(unlearned_per_presentation))
            #     print('first_learned', len(first_learned))

            #     unlearned_per_presentation_all.append(unlearned_per_presentation)
            #     first_learned_all.append(first_learned)

            #     print('unlearned_per_presentation_all', len(unlearned_per_presentation_all))
            #     print('first_learned_all', len(first_learned_all))

            #     # print('epoch before sort ordered_examples len',len(ordered_examples))

            #     # Sort examples by forgetting counts in ascending order, over one or more training runs
            #     ordered_examples, ordered_values, num_unforget = sort_examples_by_forgetting(unlearned_per_presentation_all, first_learned_all, 10000)

            #     # Get the indices to remove from training
            #     print('epoch before ordered_examples len', len(ordered_examples))
            #     print('epoch before len(train_dataset.targets)', len(train_dataset.targets))
            #     elements_to_remove = np.array(
            #         ordered_examples)[self.keep_lowest_n:self.keep_lowest_n + ( int(self.remove_n/( int(self.remove_data_epoch)/self.sp_mask_update_freq) ) )]
            #     # Remove the corresponding elements
            #     print('elements_to_remove', len(elements_to_remove))

            #     train_indx = np.setdiff1d(
            #         # range(len(train_dataset.targets)), elements_to_remove)
            #         train_indx, elements_to_remove)
            #     print('removed train_indx', len(train_indx))

            #     # Reassign train data and labels
            #     train_dataset.data = full_dataset.data[train_indx, :, :, :]
            #     train_dataset.targets = np.array(
            #         full_dataset.targets)[train_indx].tolist()

            #     print('shape', train_dataset.data.shape)
            #     print('len(train_dataset.targets)', len(train_dataset.targets))

            #     # print('epoch after random ordered_examples len', len(ordered_examples))
            #     #####empty example_stats_train!!! Because in original, forget process come before the whole training process
            #     example_stats_train = {}

            #     ##########

            logit, loss = self.model_forward(x,y)

            _, preds = logit.topk(self.topk, 1, True, True)
            # acc = preds == y
            # for j, index in enumerate(indices):
            #     # Compute missclassification margin
            #     output_correct_class = logit.data[j, y[self.temp_batch_size + j].item()]
            #     sorted_output, _ = torch.sort(logit.data[j, :])
            #     if acc[j]:
            #         # Example classified correctly, highest incorrect class is 2nd largest output
            #         output_highest_incorrect_class = sorted_output[-2]
            #     else:
            #         # Example misclassified, highest incorrect class is max output
            #         output_highest_incorrect_class = sorted_output[-1]
            #     margin = output_correct_class.item(
            #     ) - output_highest_incorrect_class.item()

            #     # Add the statistics of the current training example to dictionary
            #     index_stats = self.example_stats_train.get(index, [[], [], []])

            #     index_stats[0].append(loss[j].item())
            #     index_stats[1].append(acc[j].sum().item())
            #     index_stats[2].append(margin)
            #     self.example_stats_train[index] = index_stats
            
            loss.backward()
            if self.gradient_efficient:
                self.prune_apply_masks_on_grads_efficient()
            elif self.gradient_efficient_mix:
                if self.task_iter % self.sample_frequency == 0:
                    self.prune_apply_masks_on_grads_mix()
                else:
                    self.prune_apply_masks_on_grads_efficient()
            else:
                self.prune_apply_masks_on_grads()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.prune_apply_masks()

            self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

            # if self.gradient_efficient or self.gradient_efficient_mix and self.task_iter % 100 == 0:
            #     self.prune_print_sparsity()
            #     self.show_mask_sparsity()
            #     test_sparsity(self.model, column=False, channel=False, filter=False, kernel=False)

        return total_loss / iterations, correct / num_data


    def init_mask(self):
        self.generate_mask()

    def apply_masks(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    W.mul_((self.masks[name] != 0).type(dtype))
                    pass

    def apply_masks_on_grads(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
                    pass

    def apply_masks_on_grads_efficient(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.gradient_masks:
                    dtype = W.dtype
                    (W.grad).mul_((self.gradient_masks[name] != 0).type(dtype))
                    pass

    def apply_masks_on_grads_mix(self):
        with torch.no_grad():
            for name, W in (self.model.named_parameters()):
                if name in self.masks:
                    dtype = W.dtype
                    # print("***", name, torch.sum(self.masks[name]))
                    # print("before", torch.sum(W.grad))
                    (W.grad).mul_((self.masks[name] != 0).type(dtype))
                    # print("after", torch.sum(W.grad))
            
            for name, W in (self.model.named_parameters()):
                if name not in self.masks:  # ignore layers that do not have rho
                    continue
                # cuda_pruned_weights = None
                percent = self.kwargs["gradient_sparse"] * 100
                # print("before", torch.sum(W.grad))
                weight_temp = np.abs(W.grad.cpu().detach().numpy())  # a buffer that holds weights with absolute values
                # print("after", np.sum(weight_temp))
                # print("###")
                # print(weight_temp)
                percentile = np.percentile(
                    weight_temp,
                    percent)  # get a value for this percentitle
                under_threshold = weight_temp < percentile
                above_threshold = weight_temp > percentile
                above_threshold = above_threshold.astype(
                    np.float32
                )  # has to convert bool to float32 for numpy-tensor conversion
                # print(name, np.sum(under_threshold))
                under_threshold = torch.Tensor(under_threshold).cuda().bool()
                #print(W.grad.shape, under_threshold.shape)
                W.grad[under_threshold] = 0

                gradient = W.grad.cpu().detach().numpy()
                non_zeros = gradient != 0
                non_zeros = non_zeros.astype(np.float32)
                zero_mask = torch.from_numpy(non_zeros).cuda()
                self.gradient_masks[name] = zero_mask

    def test_mask_sparsity(self, column=False, channel=False, filter=False, kernel=False):
        
        # --------------------- total sparsity --------------------
        # comp_ratio_list = []

        total_zeros = 0
        total_nonzeros = 0
        layer_cont = 1
        mask = self.gradient_masks
        # print("### mask ###")
        # print(mask)
        for name, weight in mask.items():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                zeros = np.sum(weight.cpu().detach().numpy() == 0)
                total_zeros += zeros
                non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
                total_nonzeros += non_zeros
                print("(empty/total) masks of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
                    name, layer_cont, zeros, zeros+non_zeros, zeros / (zeros+non_zeros)))

            layer_cont += 1

        comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)
        total_sparsity = total_zeros / (total_zeros + total_nonzeros)

        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

        return comp_ratio, total_sparsity
        

    def show_masks(self, model, debug=False):
        with torch.no_grad():
            if debug:
                name = 'module.layer1.0.conv1.weight'
                np_mask = self.masks[name].cpu().numpy()
                np.set_printoptions(threshold=sys.maxsize)
                print(np.squeeze(np_mask)[0], name)
                return
            for name, W in model.named_parameters():
                if name in self.masks:
                    np_mask = self.masks[name].cpu().numpy()
                    np.set_printoptions(threshold=sys.maxsize)
                    print(np.squeeze(np_mask)[0], name)



    def update_mask(self, epoch, batch_idx):
        # a hacky way to differenate random GaP and others
        if not self.mask_update_decay_epoch:
            return
        if batch_idx != 0:
            return

        freq = self.sp_mask_update_freq
        bound_index = 0

        try: # if mask_update_decay_epoch has only one entry
            int(self.mask_update_decay_epoch)
            freq_decay_epoch = int(self.mask_update_decay_epoch)
            try: # if upper/lower bound have only one entry
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError: # if upper/lower bound have multiple entries
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity
                if epoch >= freq_decay_epoch:
                    freq *= 1
                    bound_index += 1
        except ValueError: # if mask_update_decay_epoch has multiple entries
            freq_decay_epoch = self.mask_update_decay_epoch.split('-')
            for i in range(len(freq_decay_epoch)):
                freq_decay_epoch[i] = int(freq_decay_epoch[i])

            try:
                float(self.upper_bound)
                float(self.lower_bound)
                upper_bound = [str(self.upper_bound)]
                lower_bound = [str(self.lower_bound)]
                bound_index = 0
            except ValueError:
                upper_bound = self.upper_bound.split('-')  # grow-to sparsity
                lower_bound = self.lower_bound.split('-')  # prune-to sparsity

                if len(freq_decay_epoch) + 1 <= len(upper_bound): # upper/lower bound num entries enough for all update
                    for decay in freq_decay_epoch:
                        if epoch >= decay:
                            freq *= 1
                            bound_index += 1
                else: # upper/lower bound num entries less than update needs, use the last entry to do rest updates
                    for idx, _ in enumerate(upper_bound):
                        if epoch >= freq_decay_epoch[idx] and idx != len(upper_bound) - 1:
                            freq *= 1
                            bound_index += 1

        lower_bound_value = float(lower_bound[bound_index])
        upper_bound_value = float(upper_bound[bound_index])

        if epoch % freq == 0:
            '''
            calculate prune_part and grow_part for sequential GaP, if no seq_gap_layer_indices specified in yaml file,
            set prune_part and grow_part to all layer specified in yaml file as random GaP do.
            '''
            prune_part, grow_part = self.seq_gap_partition(self.model)

            with torch.no_grad():
                sorted_to_prune = None
                if self.kwargs["sp_global_magnitude"]:
                    total_size = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        total_size += W.data.numel()
                    to_prune = np.zeros(total_size)
                    index = 0
                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        size = W.data.numel()
                        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
                        index += size
                    sorted_to_prune = np.sort(to_prune)

                # import pdb; pdb.set_trace()
                print(self.masks.keys())
                for name, W in (self.model.named_parameters()):
                    if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                        continue

                    weight = W.cpu().detach().numpy()
                    weight_current_copy = copy.copy(weight)

                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    print("name", name)
                    np_orig_mask = self.masks[name].cpu().detach().numpy()
                    print(("\n==> BEFORE UPDATE: {}: {}, {}, {}".format(name,
                                                                    str(num_nonzeros),
                                                                    str(total_num),
                                                                    str(sparsity))))

                    ############## pruning #############
                    pruned_weight_np = None
                    if name in prune_part:
                        sp_admm_sparsity_type_copy = copy.copy(self.kwargs["sp_admm_sparsity_type"])
                        sparsity_type_list = (self.kwargs["sp_admm_sparsity_type"]).split("+")
                        for i in range(len(sparsity_type_list)):
                            sparsity_type = sparsity_type_list[i]
                            print("* sparsity type {} is {}".format(i, sparsity_type))
                            self.kwargs["sp_admm_sparsity_type"] = sparsity_type

                            pruned_mask, pruned_weight = weight_pruning(self.kwargs,
                                                                        self.configs,
                                                                        name,
                                                                        W,
                                                                        lower_bound_value)
                            self.kwargs["sp_admm_sparsity_type"] = sp_admm_sparsity_type_copy
                            # pruned_mask_np = pruned_mask.cpu().detach().numpy()
                            pruned_weight_np = pruned_weight.cpu().detach().numpy()

                            W.mul_(pruned_mask.cuda())


                            non_zeros_prune = pruned_weight_np != 0
                            num_nonzeros_prune = np.count_nonzero(non_zeros_prune.astype(np.float32))
                            print(("==> PRUNE: {}: {}, {}, {}".format(name,
                                                             str(num_nonzeros_prune),
                                                             str(total_num),
                                                             str(1 - (num_nonzeros_prune * 1.0) / total_num))))

                            self.masks[name] = pruned_mask.cuda()


                            if self.kwargs["gradient_efficient"]:
                                new_lower_bound_value = lower_bound_value + self.kwargs["gradient_remove"]
                                pruned_mask, pruned_weight = weight_pruning(self.kwargs,
                                                                            self.configs,
                                                                            name,
                                                                            W,
                                                                            new_lower_bound_value)
                                self.gradient_masks[name] = pruned_mask.cuda()


                    ############## growing #############
                    if name in grow_part:
                        if pruned_weight_np is None: # use in seq gap
                            pruned_weight_np = weight_current_copy

                        updated_mask = weight_growing(self.kwargs,
                                                      name,
                                                      pruned_weight_np,
                                                      lower_bound_value,
                                                      upper_bound_value,
                                                      self.update_init_method)
                        self.masks[name] = updated_mask
                        pass



    def cut_all_partitions(self, all_update_layer_name):
        # calculate the number of partitions and range
        temp1 = str(self.seq_gap_layer_indices)
        temp1 = (temp1).split('-')
        num_partition = len(temp1) + 1
        head = 0
        end = len(all_update_layer_name)
        all_range = []

        for i, indice in enumerate(temp1):
            assert int(indice) < end, "\n\n * Error, seq_gap_layer_indices must within range [0, {}]".format(end - 1)
        assert len(temp1) == len(set(temp1)), "\n\n * Error, seq_gap_layer_indices can not have duplicate element"

        for i in range(0, num_partition):
            if i == 0:
                range_i = (head, int(temp1[i]))
            elif i == num_partition - 1:
                range_i = (int(temp1[i - 1]), end)
            else:
                range_i = (int(temp1[i - 1]), int(temp1[i]))
            print(range_i)
            all_range.append(range_i)

        for j in range(num_partition):
            range_j = all_range[j]
            self.all_part_name_list.append(all_update_layer_name[range_j[0]:range_j[1]])

    def seq_gap_partition(self, model):
        prune_part = []
        grow_part = []

        if self.seq_gap_layer_indices is None: # Random Gap: add all layer name in prune part and grow part list
            for name, _ in model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                prune_part.append(name)
                grow_part.append(name)
        else: # Sequential gap One-run: partition model
            all_update_layer_name = []
            for name, _ in model.named_parameters():
                if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                    continue
                all_update_layer_name.append(name)
            if not self.all_part_name_list:
                self.cut_all_partitions(all_update_layer_name) # get all partitions by name in self.all_part_name_list

            to_grow = (self.all_part_name_list).pop(0)
            to_prune = self.all_part_name_list

            for layer in to_grow:
                grow_part.append(layer)
            for part in to_prune:
                for layer in part:
                    prune_part.append(layer)

            (self.all_part_name_list).append(to_grow)

        return prune_part, grow_part

    def generate_mask(self):
        if self.pattern == 'weight':

            with torch.no_grad():
                for name, W in (self.model.named_parameters()):

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        print("skipped", name)
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask


        elif self.pattern == 'random':
            if self.seed is not None:
                print("Setting the random mask seed as {}".format(self.seed))
                np.random.seed(self.seed)

            with torch.no_grad():
                # self.sparsity (args.retrain_mask_sparsity) will override prune ratio config file
                if self.sparsity > 0:
                    sparsity = self.sparsity

                    for name, W in (self.model.named_parameters()):
                        if 'weight' in name and 'block.1' not in name:
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        self.masks[name] = zero_mask

                else: #self.sparsity < 0

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) \
                                and (name not in self.prune_ratios.keys()):
                            continue
                        if name in self.prune_ratios:
                            # Use prune_ratio[] to indicate which layers to random masked
                            sparsity = self.prune_ratios[name]
                            '''
                            if sparsity < 0.001:
                                continue
                            '''
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size*(1-sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()

                        self.masks[name] = zero_mask

                # # DEBUG:
                DEBUG = False
                if DEBUG:
                    for name, W in (self.model.named_parameters()):
                        m = self.masks[name].detach().cpu().numpy()
                        total_ones = np.sum(m)
                        total_size = np.size(m)
                        print( name, m.shape, (total_ones+0.0)/total_size)

                #exit()
        # TO DO
        elif self.pattern == 'regular':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    if 'weight' in name and 'block.1' not in name:

                        ouputSize, inputSize = W.data.shape[0], W.data.shape[1]
                        non_zeros = np.zeros(W.data.shape)
                        non_zeros = np.squeeze(non_zeros)

                        if 'sa1.conv_blocks.0.0.weight' in name or 'sa1.conv_blocks.1.0.weight' in name or 'sa1.conv_blocks.2.0.weight' in name:
                            print("@one")
                            non_zeros[::self.kwargs["mask_sample_rate"],::] = 1

                        else:
                            print("@two")
                            non_zeros[::self.kwargs["mask_sample_rate"],::self.kwargs["mask_sample_rate"]] = 1

                        non_zeros = np.reshape(non_zeros, W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()

                    else:
                        non_zeros = 1 - np.zeros(W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask
                    
        elif self.pattern == 'global_weight':
            with torch.no_grad():
                all_w = []
                all_name = []
                print('Concatenating all weights...')
                for name, W in self.model.named_parameters():
                    if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                        continue
                    all_w.append(W.detach().cpu().numpy().flatten())
                    all_name.append(name)
                np_w = all_w[0]
                for i in range(1,len(all_w)):
                    np_w = np.append(np_w, all_w[i])

                #print(np_w.shape)
                print("All weights concatenated!")
                print("Start sorting all the weights...")
                np_w = np.sort(np.abs(np_w))
                print("Sort done!")
                L = len(np_w)
                #print(np_w)
                if self.kwargs["retrain_mask_sparsity"] >= 0.0:
                    thr = np_w[int(L * self.kwargs["retrain_mask_sparsity"])]

                    for name, W in self.model.named_parameters():
                        if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                            continue


                        np_mask = np.abs(W.detach().cpu().numpy())  > thr
                        #print(name, np.size(np_mask), np.sum(np_mask), float(np.sum(np_mask))/np.size(np_mask) )

                        self.masks[name] = torch.from_numpy(np_mask).cuda()

                    total_non_zero = 0
                    total_size = 0
                    with open('gw_sparsity.txt','w') as f:
                        for name, W in sorted(self.model.named_parameters()):
                            if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                                continue
                            np_mask = self.masks[name].detach().cpu().numpy()
                            sparsity = 1.0 - float(np.sum(np_mask))/np.size(np_mask)
                            if sparsity < 0.5:
                                sparsity = 0.0

                            if sparsity < 0.5:
                                total_non_zero += np.size(np_mask)
                            else:
                                total_non_zero += np.sum(np_mask)
                            total_size += np.size(np_mask)

                            f.write("{}: {}\n".format(name,sparsity))
                    print("Thr:{}".format(thr))
                    print("{},{},{}".format(total_non_zero, total_size, float(total_non_zero)/total_size))
                    exit()

        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks


    def prune_init(self):
        self.prune_harden()


    def prune_update(self, epoch=0, batch_idx=0):
        self.update_mask(epoch, batch_idx)


    def prune_harden(self, option=None):

        for key in self.prune_ratios:
            print("prune_ratios[{}]:{}".format(key, self.prune_ratios[key]))

        print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:  # ignore layers that do not have rho
                continue
            cuda_pruned_weights = None
            prune_ratio = self.prune_ratios[name]
            if option == None:
                cuda_pruned_weights = self.prune_weight(name, W, prune_ratio, first)  # get sparse model in cuda
                first = False
            else:
                raise Exception("not implmented yet")
            W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

            # if self.kwargs["sp_admm_sparsity_type"] == "block":
            #     print("@one")
            #     block = eval(self.kwargs["sp_admm_block"])
            #     if block[1] == -1:  # row pruning, need to delete corresponding bias
            #         bias_layer = name.replace(".weight", ".bias")
            #         with torch.no_grad():
            #             bias = self.model.state_dict()[bias_layer]
            #             bias_mask = torch.sum(W, 1)
            #             bias_mask[bias_mask != 0] = 1
            #             bias.mul_(bias_mask)
            # elif self.kwargs["sp_admm_sparsity_type"] == "filter":
            #     print("@two")
            #     if not "downsample" in name:
            #         bn_weight_name = name.replace("conv", "bn")
            #         bn_bias_name = bn_weight_name.replace("weight", "bias")
            #     else:
            #         bn_weight_name = name.replace("downsample.0", "downsample.1")
            #         bn_bias_name = bn_weight_name.replace("weight", "bias")

            #     print("removing bn {}, {}".format(bn_weight_name, bn_bias_name))
            #     # bias_layer_name = name.replace(".weight", ".bias")

            #     with torch.no_grad():
            #         bn_weight = self.model.state_dict()[bn_weight_name]
            #         bn_bias = self.model.state_dict()[bn_bias_name]
            #         # bias = self.model.state_dict()[bias_layer_name]

            #         mask = torch.sum(W, (1, 2, 3))
            #         mask[mask != 0] = 1
            #         bn_weight.mul_(mask)
            #         bn_bias.mul_(mask)
            #         # bias.data.mul_(mask)
            # else:
            #     print("@three")
            
            non_zeros = W.detach().cpu().numpy() != 0
            non_zeros = non_zeros.astype(np.float32)
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
            # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))


    def prune_weight(self, name, weight, prune_ratio, first):
        if prune_ratio == 0.0:
            return weight
        # if pruning too many items, just prune everything
        if prune_ratio >= 0.999:
            return weight * 0.0
        if self.kwargs["sp_admm_sparsity_type"] == "irregular_global":
            _, res = weight_pruning(self.kwargs, self.configs, name, weight, prune_ratio)
        else:
            sp_admm_sparsity_type_copy = copy.copy(self.kwargs["sp_admm_sparsity_type"])
            sparsity_type_list = (self.kwargs["sp_admm_sparsity_type"]).split("+")
            if len(sparsity_type_list) != 1: #multiple sparsity type
                print(sparsity_type_list)
                for i in range(len(sparsity_type_list)):
                    sparsity_type = sparsity_type_list[i]
                    print("* sparsity type {} is {}".format(i, sparsity_type))
                    self.kwargs["sp_admm_sparsity_type"] = sparsity_type
                    _, weight =  weight_pruning(self.kwargs, self.configs, name, weight, prune_ratio)
                    self.kwargs["sp_admm_sparsity_type"] = sp_admm_sparsity_type_copy
                    print(np.sum(weight.detach().cpu().numpy() != 0))
                return weight.to(weight.device).type(weight.dtype)
            else:
                _, res = weight_pruning(self.kwargs, self.configs, name, weight, prune_ratio)

        return res.to(weight.device).type(weight.dtype)


    def prune_apply_masks(self):
        self.apply_masks()

    def prune_apply_masks_on_grads(self):
        self.apply_masks_on_grads()

    def show_mask_sparsity(self):
        self.test_mask_sparsity()

    def prune_apply_masks_on_grads_mix(self):
        self.apply_masks_on_grads_mix()

    def prune_apply_masks_on_grads_efficient(self):
        self.apply_masks_on_grads_efficient()


    # def update_prune_ratio(self, prune_ratios, global_sparsity):
    #     if self.kwargs["sp_predefine_global_weight_sparsity_dir"] is not None:
    #         # use layer sparsity in a predefined sparse model to override prune_ratios
    #         print("=> loading checkpoint for keep ratio: {}".format(args.sp_predefine_global_weight_sparsity_dir))

    #         assert os.path.exists(args.sp_predefine_global_weight_sparsity_dir), "\n\n * Error, pre_defined sparse mask model not exist!"

    #         checkpoint = torch.load(args.sp_predefine_global_weight_sparsity_dir, map_location="cuda")
    #         model_state = checkpoint["state_dict"]
    #         for name, weight in model_state.items():
    #             if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
    #                 continue
    #             zeros = np.sum(weight.cpu().detach().numpy() == 0)
    #             non_zero = np.sum(weight.cpu().detach().numpy() != 0)
    #             new_prune_ratio = float(zeros / (zeros + non_zero))
    #             prune_ratios[name] = new_prune_ratio
    #         return prune_ratios

    #     total_size = 0
    #     for name, W in (model.named_parameters()):
    #         if (canonical_name(name) not in prune_ratios.keys()) \
    #                 and (name not in prune_ratios.keys()):
    #             continue
    #         total_size += W.data.numel()
    #     to_prune = np.zeros(total_size)
    #     index = 0
    #     for name, W in (model.named_parameters()):
    #         if (canonical_name(name) not in prune_ratios.keys()) \
    #                 and (name not in prune_ratios.keys()):
    #             continue
    #         size = W.data.numel()
    #         to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
    #         index += size
    #     #sorted_to_prune = np.sort(to_prune)
    #     threshold = np.percentile(to_prune, global_sparsity*100)

    #     # update prune_ratios key-value pairs
    #     total_zeros = 0
    #     for name, W in (model.named_parameters()):
    #         if (canonical_name(name) not in prune_ratios.keys()) \
    #                 and (name not in prune_ratios.keys()):
    #             continue
    #         size = W.data.numel()
    #         np_W_abs = W.detach().cpu().abs().numpy()
    #         new_prune_ratio = float(np.sum(np_W_abs < threshold))/size
    #         if new_prune_ratio >= 0.999:
    #             new_prune_ratio = 0.99

    #         total_zeros += float(np.sum(np_W_abs < threshold))

    #         prune_ratios[name] = new_prune_ratio

    #     print("Updated prune_ratios:")
    #     for key in prune_ratios:
    #         print("prune_ratios[{}]:{}".format(key,prune_ratios[key]))
    #     total_sparsity = total_zeros / total_size
    #     print("Total sparsity:{}".format(total_sparsity))

    #     return prune_ratios


    def prune_print_sparsity(self, logger=None, show_sparse_only=False, compressed_view=False):
        # elif retrain:
        #     p = retrain.logger.info
        # else:
        p = print

        if show_sparse_only:
            print("The sparsity of all params (>0.01): num_nonzeros, total_num, sparsity")
            for (name, W) in self.model.named_parameters():
                non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                num_nonzeros = np.count_nonzero(non_zeros)
                total_num = non_zeros.size
                sparsity = 1 - (num_nonzeros * 1.0) / total_num
                if sparsity > 0.01:
                    print("{}, {}, {}, {}, {}".format(name, non_zeros.shape, num_nonzeros, total_num, sparsity))
            return

        if compressed_view is True:
            total_w_num = 0
            total_w_num_nz = 0
            for (name, W) in self.model.named_parameters():
                if "weight" in name:
                    non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_w_num_nz += num_nonzeros
                    total_num = non_zeros.size
                    total_w_num += total_num

            sparsity = 1 - (total_w_num_nz * 1.0) / total_w_num
            print("The sparsity of all params with 'weights' in its name: num_nonzeros, total_num, sparsity")
            print("{}, {}, {}".format(total_w_num_nz, total_w_num, sparsity))
            return

        print("The sparsity of all parameters: name, num_nonzeros, total_num, shape, sparsity")
        for (name, W) in self.model.named_parameters():
            non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))





