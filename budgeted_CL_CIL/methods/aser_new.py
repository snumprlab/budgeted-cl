# When we make a new one, we should inherit the Finetune class.
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, MultiProcessLoader

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class ASER(CLManagerBase):
    
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.aser_type = kwargs["aser_type"]
        self.k = kwargs["k"]
        self.n_smp_cls = int(kwargs["n_smp_cls"])
        self.candidate_size = kwargs["aser_cands"]
        print("candidate_size")
        print(self.candidate_size)
        self.waiting_candidate_batch = []
        self.waiting_eval_batch = []
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device)
        self.candidate_train_dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device)
        self.current_dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.test_transform, self.data_dir, False, self.cpu_transform, self.device)
        self.candidate_dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.test_transform, self.data_dir, False, self.cpu_transform, self.device)
        self.eval_dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.test_transform, self.data_dir, False, self.cpu_transform, self.device)
        self.memory = AserMemory(self.memory_size, self.n_smp_cls)
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
        for i in range(self.future_steps):
            self.load_batch()
    '''
    # loader로부터 load된 batch를 받아오는 것
    def get_candidate_batch(self, next_batch):
        batch = self.candidate_dataloader.get_batch()
        self.load_candidate_batch(next_batch)
        return batch

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    def load_candidate_batch(self, batch):
        self.candidate_dataloader.load_batch(batch)
    '''

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.eval_dataloader.add_new_class(self.memory.cls_dict)
            self.current_dataloader.add_new_class(self.memory.cls_dict)
            self.candidate_train_dataloader.add_new_class(self.memory.cls_dict)
            self.candidate_dataloader.add_new_class(self.memory.cls_dict)
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
        self.reservoir_memory(sample)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        if self.sample_num >= 10 * self.candidate_size:
            candidate_batch = self.candidate_dataloader.get_batch()
            eval_batch = self.eval_dataloader.get_batch()
            candidate_train_batch = self.candidate_train_dataloader.get_batch()
            current_eval_batch = self.current_dataloader.get_batch()
            self.load_batch()
            return (batch, current_eval_batch, candidate_train_batch, candidate_batch, eval_batch)
        else:
            self.load_batch()
            return (batch)

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    
    def load_batch(self, aser=False):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            if self.future_sample_num >= 10 * self.candidate_size:
                self.candidate_dataloader.load_batch(self.waiting_candidate_batch[0])
                self.eval_dataloader.load_batch(self.waiting_eval_batch[0])
                self.candidate_train_dataloader.load_batch(self.waiting_candidate_batch[0])
                self.current_dataloader.load_batch(self.waiting_batch[0])
                del self.waiting_candidate_batch[0]
                del self.waiting_eval_batch[0]
            del self.waiting_batch[0]
    
    
    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            if self.future_sample_num >= 10 * self.candidate_size:
                #self.aser=True
                candidate_batch, eval_batch = self.memory.retrieval(self.candidate_size, aser=True)
                self.waiting_batch.append(self.temp_future_batch)
                self.waiting_candidate_batch.append(candidate_batch)
                self.waiting_eval_batch.append(eval_batch)
            else:
                self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size, aser=False))
            

    def deep_features(self, eval_x, n_eval, cand_x, n_cand):
        # Get deep features
        if cand_x is None:
            num = n_eval
            total_x = eval_x
        else:
            num = n_eval + n_cand
            total_x = torch.cat((eval_x, cand_x), 0)

        # compute deep features with mini-batches
        total_x = total_x.to(self.device)
        deep_features_ = self.mini_batch_deep_features(total_x, num)

        eval_df = deep_features_[0:n_eval]
        cand_df = deep_features_[n_eval:]
        return eval_df, cand_df

    def sorted_cand_ind(self, eval_df, cand_df, n_eval, n_cand):

        # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
        # Preprocess feature vectors to facilitate vector-wise distance computation
        eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
        cand_df_tile = cand_df.repeat([n_eval, 1])
        # Compute distance between evaluation and candidate feature vectors
        distance_vector = (eval_df_repeat - cand_df_tile).pow(2).sum(1)
        # Turn distance vector into distance matrix
        distance_matrix = distance_vector.reshape((n_eval, n_cand))
        # Sort candidate set indices based on distance
        sorted_cand_ind_ = distance_matrix.argsort(1)
        return sorted_cand_ind_

    def compute_knn_sv(self, eval_x, eval_y, cand_x, cand_y, k):
        # Compute KNN SV score for candidate samples w.r.t. evaluation samples
        n_eval = eval_x.size(0)
        n_cand = cand_x.size(0)
        
        # Initialize SV matrix to matrix of -1
        sv_matrix = torch.zeros((n_eval, n_cand), device=self.device)
        # Get deep features
        eval_df, cand_df = self.deep_features(eval_x, n_eval, cand_x, n_cand)
        # Sort indices based on distance in deep feature space
        sorted_ind_mat = self.sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

        # Evaluation set labels
        el = eval_y
        el_vec = el.repeat([n_cand, 1]).T
        # Sorted candidate set labels
        cl = cand_y[sorted_ind_mat]

        # Indicator function matrix
        indicator = (el_vec == cl).float()
        indicator_next = torch.zeros_like(indicator, device=self.device)
        indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
        indicator_diff = indicator - indicator_next

        cand_ind = torch.arange(n_cand, dtype=torch.float, device=self.device) + 1
        denom_factor = cand_ind.clone()
        denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
        numer_factor = cand_ind.clone()
        numer_factor[k:n_cand - 1] = k
        numer_factor[n_cand - 1] = 1
        factor = numer_factor / denom_factor

        indicator_factor = indicator_diff * factor
        indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

        # Row indices
        row_ind = torch.arange(n_eval, device=self.device)
        row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

        # Compute SV recursively
        sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum
        return sv_matrix

    def mini_batch_deep_features(self, total_x, num):
        with torch.no_grad():
            bs = 64
            num_itr = num // bs + int(num % bs > 0)
            sid = 0
            deep_features_list = []
            for i in range(num_itr):
                eid = sid + bs if i != num_itr - 1 else num
                batch_x = total_x[sid: eid]
                _, batch_deep_features_ = self.model(batch_x, get_feature=True)
                
                self.total_flops += (len(batch_x) * self.forward_flops)   
                deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
                sid = eid
                
            if num_itr == 1:
                deep_features_ = deep_features_list[0]
            else:
                deep_features_ = torch.cat(deep_features_list, 0)
                
        return deep_features_

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            train_loss, train_acc = self.online_train(sample_num, iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
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
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)


    def online_train(self, sample_num, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            batch_data = self.get_batch()
            
            if type(batch_data)!=dict:
                data, current_eval_data, candidate_train_data, candidate_data, eval_data = batch_data
                eval_adv_x = current_eval_data["image"].to(self.device)
                eval_adv_y = current_eval_data["label"].to(self.device)
                current_train_x = data["image"].to(self.device)
                current_train_y = data["label"].to(self.device)
                cand_train_x = candidate_train_data["image"].to(self.device)
                cand_train_y = candidate_train_data["label"].to(self.device)
                cand_x = candidate_data['image'].to(self.device)
                cand_y = candidate_data['label'].to(self.device)
                eval_coop_x = eval_data['image'].to(self.device)
                eval_coop_y = eval_data['label'].to(self.device)
                '''
                print("eval_adv_x", len(eval_adv_x))
                print("current_train_x", len(current_train_x))
                print("cand_train_x", len(cand_train_x))
                print("cand_x", len(cand_x))
                print("eval_coop_x", len(eval_coop_x))
                '''
                self.model.eval()
                sv_matrix_adv = self.compute_knn_sv(eval_adv_x, eval_adv_y, cand_x, cand_y, self.k)
                sv_matrix_coop = self.compute_knn_sv(eval_coop_x, eval_coop_y, cand_x, cand_y, self.k)
                
                if self.aser_type == "asv":
                    # Use extremal SVs for computation
                    sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
                else:
                    # Use mean variation for aser_type == "asvm" or anything else
                    sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
                    
                ret_ind = sv.argsort(descending=True)
                memory_batch_size = min(len(self.memory.images), self.memory_batch_size)
                x = torch.cat([current_train_x, cand_train_x[ret_ind][:memory_batch_size]])
                y = torch.cat([current_train_y, cand_train_y[ret_ind][:memory_batch_size]])
            else:
                data = batch_data
                x = data['image']
                y = data['label']
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            #else:
                #self.memory.register_batch_indices(batch_indices = cand_index[ret_ind][:memory_batch_size])
            #self.before_model_update()
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

            self.total_flops += (len(y) * (self.forward_flops + self.backward_flops))

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data
    

class AserMemory(MemoryBase):
    def __init__(self, memory_size, n_smp_cls):
        super().__init__(memory_size)
        self.n_smp_cls = n_smp_cls
    
    # balanced probability retrieval
    def balanced_retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        cls_idx = np.random.choice(len(self.cls_list), sample_size)
        for cls in cls_idx:
            i = np.random.choice(self.cls_idx[cls], 1)[0]
            memory_batch.append(self.images[i])
            self.usage_count[i]+=1
            self.class_usage_count[self.labels[i]]+=1
        return memory_batch
    
    def retrieval(self, size, aser=False):
        
        if aser:    
            ##### for candidate data #####
            candidate_indices = self.get_class_balance_indices(self.n_smp_cls, size)
            candidate_batch = list(np.array(self.images)[candidate_indices])
            
            ##### for eval data #####           
            # discard indices는 겹치는 애들을 의미하며, 해당 index eval indices를 뽑을 때 빼주어야 한다.
            eval_indices = self.get_class_balance_indices(self.n_smp_cls, candidate_size= size, discard_indices = candidate_indices)
            eval_batch = list(np.array(self.images)[eval_indices])
            
            return candidate_batch, eval_batch
        
        else:
            # random retrieval 
            sample_size = min(size, len(self.images))
            memory_batch = []
            indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
            for i in indices:
                memory_batch.append(self.images[i])
            return memory_batch

    def get_class_balance_indices(self, n_smp_cls, candidate_size=None, discard_indices = None):
        if candidate_size is not None:
            real_candidate_size = min(n_smp_cls, candidate_size // len(self.cls_idx))
            #real_candidate_size = candidate_size // len(self.cls_idx)
            indices = []

            # balanced sampling
            for klass in range(len(self.cls_idx)):
                candidates = self.cls_idx[klass]
                if discard_indices is not None:
                    candidates = list(set(candidates) - set(discard_indices))
                indices.extend(np.random.choice(candidates, size=min(real_candidate_size, len(candidates)), replace=False))

            # additional sampling for match candidate_size
            additional_size = candidate_size % len(self.cls_idx)
            candidates = list(set(range(len(self.images))) - set(indices))
            indices.extend(np.random.choice(candidates, size=additional_size, replace=False))

        else:
            indices = []
            # balanced sampling
            print()
            for klass in range(len(self.cls_idx)):
                candidates = self.cls_idx[klass]
                if discard_indices is not None:
                    candidates = list(set(candidates) - set(discard_indices))
                indices.extend(np.random.choice(candidates, size=min(n_smp_cls, len(candidates)), replace=False))

        return indices
        
    