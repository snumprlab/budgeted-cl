# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.data_loader import ImageDataset, MultiProcessLoader

from methods.cl_manager import MemoryBase
from methods.er_new import ER
# from models.cifar import ResNet_G, ResNet_F

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class REMIND(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.dataset = kwargs["dataset"]
        
        self.n_codebooks = kwargs['n_codebooks']
        self.codebook_size = kwargs['codebook_size']
        self.mixup_alpha = kwargs['mixup_alpha']
        self.total_baseinit_samples = kwargs['baseinit_samples']
        self.spatial_feat_dim = kwargs['spatial_feat_dim']
        self.feat_memory_size = kwargs['feat_memsize']

        self.train_datalist = train_datalist
        # self.remind_baseinit_datalist = self.train_datalist[:self.total_baseinit_samples]
        # self.remind_train_datalist = self.train_datalist[self.total_baseinit_samples:]
        # self.random_resize_crop = RandomResizeCrop(7, scale=(2 / 7, 1.0))

        self.baseinit = True
        self.iteration = 0
        

        super().__init__(train_datalist, test_datalist, device, **kwargs)


    def initialize_future(self):
        self.memory = REMINDMemory(self.memory_size, self.feat_memory_size)
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        
        self.model = RemindModel(self.model_name, self.dataset, self.device).to(self.device)
        self.get_flops_parameter(self.method_name)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.exposed_classes = []
        self.baseinit_iter_point = round(self.total_baseinit_samples*self.online_iter)
        
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True
        
        self.pq_train_batch_size = self.batch_size
        self.pq_train_samples_limit = self.pq_train_batch_size*(40000//self.pq_train_batch_size)
        


        ### resnet18 ###
        '''
        self.G_forward_flops = self.initial_forward_flops + self.group1_block0_forward_flops + \
            self.group1_block1_forward_flops + self.group2_block0_forward_flops + self.group2_block0_forward_flops + \
            self.group3_block0_forward_flops + self.group3_block1_forward_flops
        
        self.G_backward_flops = self.initial_backward_flops + self.group1_block0_backward_flops + \
            self.group1_block1_backward_flops + self.group2_block0_backward_flops + self.group2_block0_backward_flops + \
            self.group3_block0_backward_flops + self.group3_block1_backward_flops
        
        self.F_forward_flops = self.group4_block0_forward_flops + self.group4_block1_forward_flops + \
            self.fc_forward_flops
            
        self.F_backward_flops = self.group4_block0_backward_flops + self.group4_block1_backward_flops + \
            self.fc_backward_flops
        '''

        ### resnet32 ###
        '''
        self.G_forward_flops = self.initial_forward_flops + \
        self.group1_block0_forward_flops + self.group1_block1_forward_flops + self.group1_block2_forward_flops + self.group1_block3_forward_flops + self.group1_block4_forward_flops + \
        self.group2_block0_forward_flops + self.group2_block1_forward_flops + self.group2_block2_forward_flops + self.group2_block3_forward_flops + self.group2_block4_forward_flops + \
        self.group3_block0_forward_flops + self.group3_block1_forward_flops + self.group3_block2_forward_flops
        
        self.G_backward_flops = self.initial_backward_flops + \
        self.group1_block0_backward_flops + self.group1_block1_backward_flops + self.group1_block2_backward_flops + self.group1_block3_backward_flops + self.group1_block4_backward_flops + \
        self.group2_block0_backward_flops + self.group2_block1_backward_flops + self.group2_block2_backward_flops + self.group2_block3_backward_flops + self.group2_block4_backward_flops + \
        self.group3_block0_backward_flops + self.group3_block1_backward_flops + self.group3_block2_backward_flops
        
        self.F_forward_flops = self.group3_block3_forward_flops + self.group3_block4_forward_flops + self.fc_forward_flops
        self.F_backward_flops = self.group3_block3_backward_flops + self.group3_block4_backward_flops + self.fc_backward_flops
        '''


        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def generate_waiting_batch(self, iterations):
        if self.baseinit:
            for i in range(iterations):
                memory_batch = self.memory.retrieval(self.memory_batch_size)
                self.waiting_batch.append(self.temp_future_batch + memory_batch)
        else:
            for i in range(iterations):
                self.waiting_batch.append(self.temp_future_batch)

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

        if self.baseinit:
            if len(self.temp_future_batch) >= self.temp_batch_size:
                self.generate_waiting_batch(int(self.future_num_updates))
                for stored_sample in self.temp_future_batch:
                    self.update_memory(stored_sample)
                self.temp_future_batch = []
                self.future_num_updates -= int(self.future_num_updates)
        else:
            if len(self.temp_future_batch) >= self.temp_batch_size:
                self.generate_waiting_batch(int(self.future_num_updates))
                self.temp_future_batch = []
                self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        if self.future_sample_num == self.total_baseinit_samples:
            self.prep_pqtrain()
        return 0
    
    # def safe_load_dict(self, model, new_model_state, should_resume_all_params=False):
    #     old_model_state = model.state_dict()
    #     c = 0
    #     if should_resume_all_params:
    #         for old_name, old_param in old_model_state.items():
    #             assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(old_name)
    #     for name, param in new_model_state.items():
    #         n = name.split('.')
    #         beg = n[0]
    #         end = n[1:]
    #         if beg == 'module':
    #             name = '.'.join(end)
    #         if name not in old_model_state:
    #             # print('%s not found in old model.' % name)
    #             continue
    #         if isinstance(param, nn.Parameter):
    #             # backwards compatibility for serialized parameters
    #             param = param.data
    #         c += 1
    #         if old_model_state[name].shape != param.shape:
    #             print('Shape mismatch...ignoring %s' % name)
    #             continue
    #         else:
    #             old_model_state[name].copy_(param)
    #     if c == 0:
    #         raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')
    #     return model
    
    def prep_pqtrain(self):
        print("prep for pq")
        logger.info("prep for pq")
        train_df = pd.DataFrame(list(self.memory.images))
        self.pq_train_dataset = ImageDataset(train_df, self.dataset, self.test_transform, cls_list=self.exposed_classes, data_dir=self.data_dir)
        self.pq_train_loader = DataLoader(self.pq_train_dataset, batch_size=self.pq_train_batch_size, shuffle=False)
        self.pq_train_sample_num = min(self.pq_train_samples_limit, len(self.memory.images))

    def base_initialize(self):
        self.model_G = copy.deepcopy(self.model.model_G).to(self.device)
        self.model_F = copy.deepcopy(self.model.model_F).to(self.device)
        self.model_F.fc = copy.deepcopy(self.model.fc).to(self.device)
            
        
        for name, param in self.model_G.named_parameters():
            param.requires_grad = False


        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model_F)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        
        self.num_channels = self.model_G.num_channels
        self.model_G.eval()
        
        if self.pq_train_samples_limit < len(self.pq_train_dataset):
            self.incre_features_data = np.empty((len(self.pq_train_dataset)-self.pq_train_sample_num, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim), dtype=np.float32)
            self.incre_labels_data = np.empty((len(self.pq_train_dataset)-self.pq_train_sample_num, 1), dtype=int)
            self.incre_item_ixs_data = np.empty((len(self.pq_train_dataset)-self.pq_train_sample_num, 1), dtype=int)
        
        pq_trained = False
        start_ix = 0
        features_data = np.empty((self.pq_train_sample_num, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim), dtype=np.float32)
        labels_data = np.empty((self.pq_train_sample_num, 1), dtype=int)
        item_ixs_data = np.empty((self.pq_train_sample_num, 1), dtype=int)
        logger.info("begin pq train")
        with torch.no_grad():
            for i, data in enumerate(self.pq_train_loader):
                inputs = data['image']
                targets = data['label']
                batchs = inputs.to(self.device)
                output = self.model_G(batchs)
                if not pq_trained:
                    end_ix = min(start_ix + len(output), self.pq_train_sample_num)
                    features_data[start_ix:end_ix] = output.detach().cpu().numpy()
                    labels_data[start_ix:end_ix] = np.atleast_2d(targets.numpy().astype(int)).transpose()
                    item_ixs_data[start_ix:end_ix] = np.atleast_2d(range(start_ix,end_ix)).transpose()
                    if end_ix == self.pq_train_sample_num:
                        self.fit_pq(features_data, labels_data, item_ixs_data, self.num_channels, self.spatial_feat_dim, self.n_codebooks, self.codebook_size)
                        pq_trained = True
                        del features_data
                        del labels_data
                        del item_ixs_data
                else:
                    print("here")
                    logger.info("leftover save")
                    end_ix = min(start_ix + len(output), len(self.pq_train_dataset))
                    self.incre_features_data[start_ix-self.pq_train_sample_num:end_ix-self.pq_train_sample_num] = output.detach().cpu().numpy()
                    self.incre_labels_data[start_ix-self.pq_train_sample_num:end_ix-self.pq_train_sample_num] = np.atleast_2d(targets.numpy().astype(int)).transpose()
                    self.incre_item_ixs_data[start_ix-self.pq_train_sample_num:end_ix-self.pq_train_sample_num] = np.atleast_2d(range(start_ix,end_ix)).transpose()
                    # targets = np.atleast_2d(targets.numpy().astype(int)).transpose()
                    # item_ixs = np.atleast_2d(range(start_ix, end_ix)).transpose()
                    # data_batch = output.detach()
                    # data_batch = data_batch.permute(0, 2, 3, 1)
                    # data_batch = data_batch.reshape(-1, self.num_channels).cpu().numpy()
                    # data_batch = output.detach().cpu().numpy()
                    if end_ix == len(self.pq_train_dataset):
                        self.incre_features_data = np.transpose(self.incre_features_data, (0, 2, 3, 1))
                        self.incre_features_data = np.reshape(self.incre_features_data, (-1, self.num_channels))
                        codes = self.pq.compute_codes(self.incre_features_data)
                        codes = np.reshape(codes, (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks))
                        for i in range(len(codes)):
                            self.memory.replace_completed_sample(codes[i], self.incre_labels_data[i][0], self.incre_item_ixs_data[i][0])
                        del self.incre_features_data
                        del self.incre_labels_data
                        del self.incre_item_ixs_data
                start_ix = end_ix
                self.total_flops += len(targets) * self.G_forward_flops
                
        self.seen = len(self.memory.rehearsal_ixs)
        print("BASEINIT MEMORY UPDATE DONE")
        logger.info("BASEINIT MEMORY UPDATE DONE")
        
    # fit quantization model & store data
    def fit_pq(self, feats_base_init, labels_base_init, item_ix_base_init, num_channels, spatial_feat_dim, num_codebooks, codebook_size, batch_size=128):
        train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
        train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels))
        num_samples = len(labels_base_init)
        # Train PQ
        nbits = int(np.log2(codebook_size))
        self.pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits)
        self.pq.train(train_data_base_init)
        print("TRAINDONE")
        logger.info("TRAINDONE")
        del train_data_base_init
        
        for i in range(0, num_samples, int(num_samples/2)):
            start = i
            end = min(start + int(num_samples/2), num_samples)
            data_batch = feats_base_init[start:end]
            batch_labels = labels_base_init[start:end]
            batch_item_ixs = item_ix_base_init[start:end]
            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, num_channels))
            # data_batch = np.ascontiguousarray(data_batch)
            codes = self.pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, spatial_feat_dim, spatial_feat_dim, num_codebooks))
            for i in range(len(codes)):
                self.memory.replace_completed_sample(codes[i], batch_labels[i][0], batch_item_ixs[i][0])
        
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        if self.baseinit:
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
        else:
            prev_weight = copy.deepcopy(self.model_F.fc.weight.data)
            prev_bias = copy.deepcopy(self.model_F.fc.bias.data)
            self.model_F.fc = nn.Linear(self.model_F.fc.in_features, self.num_learned_class).to(self.device)
            with torch.no_grad():
                if self.num_learned_class > 1:
                    self.model_F.fc.weight[:self.num_learned_class - 1] = prev_weight
                    self.model_F.fc.bias[:self.num_learned_class - 1] = prev_bias
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model_F.fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
            
            
    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            del self.waiting_batch[0]
            
    def mixup_data(self, x1, y1, x2, y2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x1 + (1 - lam) * x2
        y_a, y_b = y1, y2
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())
    
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        # ongoing_class = None
        for iter in range(iterations):
            self.iteration+=1
            if self.baseinit:
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

                self.total_flops += (len(y) * (self.backward_flops))

                self.after_model_update()

                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)
            
            else:
                self.model_F.train()
                self.model_G.eval()
                data = self.get_batch()
                stream_x = data["image"].to(self.device)
                stream_y = data["label"].to(self.device)
                stream_data_batch = self.model_G(stream_x).detach()
                self.total_flops += len(stream_y) * self.G_forward_flops
                stream_data_batch = stream_data_batch.permute(0, 2, 3, 1)
                stream_data_batch = stream_data_batch.reshape(-1, self.num_channels).cpu().numpy()
                # stream_data_batch = np.ascontiguousarray(stream_data_batch)
                stream_codes = torch.from_numpy(self.pq.compute_codes(stream_data_batch)).to(self.device)
                stream_codes = stream_codes.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks)
                stream_batch_length = len(stream_y)
                stream_id = range(self.seen, self.seen+stream_batch_length)

                memory_data = self.memory.train_retrieval(self.memory_batch_size)
                # memory_ids = [data['ix'] for data in memory_data]
                # data_codes = np.empty(((self.temp_batch_size + self.memory_batch_size), self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks), dtype=np.uint8)
                # data_labels = torch.empty((self.temp_batch_size + self.memory_batch_size), dtype=torch.long).to(self.device)
                
                memory_codes = torch.stack([torch.tensor(memory_data[i]['latent_dict'][0]) for i in range(len(memory_data))]).to(self.device)
                memory_labels = torch.stack([torch.tensor(memory_data[i]['latent_dict'][1]) for i in range(len(memory_data))]).to(self.device)
                data_codes = torch.cat(((stream_codes)[:self.temp_batch_size],memory_codes))
                data_labels = torch.cat((stream_y[:self.temp_batch_size],memory_labels))
        
                # for i in range(self.temp_batch_size):
                #     data_codes[i] = stream_codes[i]
                #     data_labels[i] = stream_y[i].long()
                
                # for i in range(len(memory_data)):
                #     data_codes[self.temp_batch_size+i] = memory_data[i]['latent_dict'][0]
                #     data_labels[self.temp_batch_size+i] = torch.tensor(memory_data[i]['latent_dict'][1])
                    
                # Reconstruct
                data_codes = data_codes.reshape((self.temp_batch_size + len(memory_labels)) * self.spatial_feat_dim * self.spatial_feat_dim, self.n_codebooks)
                data_batch_reconstructed = self.pq.decode(data_codes.cpu().numpy())
                data_batch_reconstructed = torch.from_numpy(data_batch_reconstructed).to(self.device)
                data_batch_reconstructed = data_batch_reconstructed.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels)
                data_batch_reconstructed = data_batch_reconstructed.permute(0, 3, 1, 2)
                
                # # resize crop augmentation
                # transform_data_batch = torch.empty_like(data_batch_reconstructed)
                # for tens_ix, tens in enumerate(data_batch_reconstructed):
                #     transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                # data_batch_reconstructed = transform_data_batch
                
                # # mixup
                # x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                #     data_batch_reconstructed[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size],
                #     data_labels[self.temp_batch_size:self.temp_batch_size + self.memory_batch_size],
                #     data_batch_reconstructed[self.temp_batch_size + self.memory_batch_size:],
                #     data_labels[self.temp_batch_size + self.memory_batch_size:],
                #     alpha=self.mixup_alpha)
                
                # data = torch.empty((self.temp_batch_size+self.memory_batch_size, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim))
                # labels_a = torch.zeros(self.memory_batch_size + self.temp_batch_size).long()
                # labels_b = torch.zeros(self.memory_batch_size + self.temp_batch_size).long()
                # data[:self.temp_batch_size] = data_batch_reconstructed[:self.temp_batch_size]
                # labels_a[:self.temp_batch_size] = stream_y
                # labels_b[:self.temp_batch_size] = stream_y
                # data[self.temp_batch_size:] = x_prev_mixed.clone()
                # labels_a[self.temp_batch_size:] = prev_labels_a
                # labels_b[self.temp_batch_size:] = prev_labels_b
                    
                    # fit on replay and new sample
                self.optimizer.zero_grad()
                # data = data.to(self.device)
                output = self.model_F(data_batch_reconstructed)
                loss = self.criterion(output, data_labels)
                # loss = self.mixup_criterion(self.criterion, output, labels_a.to(self.device), labels_b.to(self.device), lam)
                _, preds = output.topk(self.topk, 1, True, True)
                # preds = preds.detach().cpu()
                # data_labels = data_labels.detach().cpu()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.total_flops += (len(data_batch_reconstructed) * (self.F_backward_flops + self.F_forward_flops))
                self.after_model_update()
                    
                total_loss += loss.item()
                correct += torch.sum(preds == data_labels.unsqueeze(1)).item()
                num_data += data_labels.size(0)
                # correct += torch.sum(preds == labels_a.unsqueeze(1)).item()
                # correct += torch.sum(preds == labels_b.unsqueeze(1)).item()
                # num_data += labels_a.size(0)
                # num_data += labels_b.size(0)

                for i in range(self.temp_batch_size):
                    if iter==0:
                        self.update_completed_memory(stream_codes[i].cpu().numpy(),stream_y[i].cpu().numpy(), stream_id[i])
                if iter==iterations-1:
                    self.seen+=stream_batch_length

        return total_loss / (iterations*self.temp_batch_size), correct / num_data

    def finalize_baseinit(self):
        self.memory.baseinit_prep()
        self.base_initialize()
        print("Finish BaseInitialization")
        self.baseinit = False
    
    def update_completed_memory(self, code, label, item_ix):
        # item_ix = str(item_ix)
        self.memory.replace_completed_sample(code, label, item_ix)
        # self.memory.replace_completed_sample(code, label)
    
    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        feature_dict = {}
        label = []

        if self.baseinit:
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
                
        else:
            # probas = torch.zeros((len(test_loader.dataset), len(self.exposed_classes)))
            # all_lbls = torch.zeros((len(test_loader.dataset)))
            # start_ix = 0
            with torch.no_grad():
                self.model_G.to(self.device)
                self.model_F.to(self.device)
                self.model_G.eval()
                self.model_F.eval()
                for i, data in enumerate(test_loader):
                    x = data["image"]
                    y = data["label"]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    batch_length = len(y)
                    data_batch = self.model_G(x).detach()
                    data_batch = data_batch.permute(0, 2, 3, 1)
                    data_batch = data_batch.reshape(-1, self.num_channels).cpu().numpy()
                    # data_batch = np.ascontiguousarray(data_batch)
                    codes = self.pq.compute_codes(data_batch)
                    codes = np.reshape(codes, (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks))
                    data_codes = np.reshape(codes, (batch_length * self.spatial_feat_dim * self.spatial_feat_dim, self.n_codebooks))
                    data_batch_reconstructed = torch.from_numpy(self.pq.decode(data_codes)).to(self.device)
                    data_batch_reconstructed = data_batch_reconstructed.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels)
                    data_batch_reconstructed = data_batch_reconstructed.permute(0, 3, 1, 2)

                    logit = self.model_F(data_batch_reconstructed)
                    
                    # end_ix = start_ix + len(x)
                    # probas[start_ix:end_ix] = F.softmax(logit.data, dim=1)
                    # all_lbls[start_ix:end_ix] = y.squeeze()
                    # start_ix = end_ix
                    loss = criterion(logit, y)

                    pred = torch.argmax(logit, dim=-1)
                    _, preds = logit.topk(self.topk, 1, True, True)
                    total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                    total_num_data += y.size(0)
        
                    xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                    correct_l += correct_xlabel_cnt.cpu()
                    num_data_l += xlabel_cnt.cpu()
                    total_loss += loss.item()
                    label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader.dataset)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret



class REMINDMemory(MemoryBase):
    def __init__(self, baseinit_mem_size, feat_mem_size):
        self.latent_dict = {}
        self.rehearsal_ixs = []
        self.feat_mem_size = feat_mem_size

        super().__init__(baseinit_mem_size)

    def __len__(self):
        return sum(self.cls_count)
    
    def baseinit_prep(self):
        self.cls_idx = []
        self.cls_count = []
        for i in range(len(self.labels)):
            self.cls_idx.append([])
            self.cls_count.append(0)
    
    def replace_completed_sample(self, code, label, id):
        id = str(id)
        if len(self.rehearsal_ixs) < self.feat_mem_size:
            self.latent_dict[id] = [code, label]
            self.cls_idx[label].append(id)
            self.rehearsal_ixs.append(id)
            self.cls_count[label] += 1
        else:
            remove_label = self.cls_count.index(max(self.cls_count))
            cls_ind = np.random.randint(0, len(self.cls_idx[remove_label]))
            remove_id = self.cls_idx[remove_label][cls_ind]

            self.cls_count[remove_label] -= 1
            del self.latent_dict[remove_id]
            self.cls_idx[remove_label].remove(remove_id)
            self.rehearsal_ixs.remove(remove_id)

            self.cls_count[label] += 1
            self.latent_dict[id] = [code, label]
            self.cls_idx[label].append(id)
            self.rehearsal_ixs.append(id)

    def train_retrieval(self, size):
        memory_batch = []
        indices = np.random.choice(range(len(self.rehearsal_ixs)), size=size, replace=False)
        for i in indices:
            data = {}
            data['ix'] = self.rehearsal_ixs[i]
            data['latent_dict'] = self.latent_dict[self.rehearsal_ixs[i]]
            memory_batch.append(data)
        return memory_batch
    

class RemindModel(nn.Module):
    def __init__(self,model_name, dataset, device):
        super(RemindModel, self).__init__()
        self.model_G = select_model(model_name, dataset, 1, G=True)
        self.model_F = select_model(model_name, dataset, 1, F=True)
        self.fc = nn.Linear(self.model_F.fc.in_features, 1)
        self.model_F.fc = nn.Identity()
      
    def forward(self, x):
        x = self.model_G(x)
        out = self.model_F(x)
        out = self.fc(out)
        return out
    

# class RandomResizeCrop(object):
#     """Randomly crops tensor then resizes uniformly between given bounds
#     Args:
#         size (sequence): Bounds of desired output sizes.
#         scale (sequence): Range of size of the origin size cropped
#         ratio (sequence): Range of aspect ratio of the origin aspect ratio cropped
#         interpolation (int, optional): Desired interpolation. Default is 'bilinear'
#     """

#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
#         #        assert (isinstance(size, Iterable) and len(size) == 2)
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio
#         self.interpolation = interpolation

#     def get_params(self, img, scale, ratio):
#         """Get parameters for ``crop`` for a random sized crop.
#         Args:
#             img (3-d tensor (C,H,W)): Tensor to be cropped.
#             scale (tuple): range of size of the origin size cropped
#             ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         area = img.size(1) * img.size(2)

#         for attempt in range(10):
#             target_area = random.uniform(*scale) * area
#             log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
#             aspect_ratio = math.exp(random.uniform(*log_ratio))

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if w <= img.size(1) and h <= img.size(2):
#                 i = random.randint(0, img.size(2) - h)
#                 j = random.randint(0, img.size(1) - w)
#                 return i, j, h, w

#         # Fallback to central crop
#         in_ratio = img.size(1) / img.size(2)
#         if (in_ratio < min(ratio)):
#             w = img.size(1)
#             h = int(w / min(ratio))
#         elif (in_ratio > max(ratio)):
#             h = img.size(2)
#             w = int(h * max(ratio))
#         else:  # whole image
#             w = img.size(1)
#             h = img.size(2)
#         i = int((img.size(2) - h) // 2)
#         j = int((img.size(1) - w) // 2)
#         return i, j, h, w

#     def __call__(self, img):
#         """
#         Args:
#             img (3-D tensor (C,H,W)): Tensor to be cropped and resized.
#         Returns:
#             Tensor: Randomly cropped and resized Tensor.
#         """
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         img = img[:, i:i + h, j:j + w]  ##crop
#         return torch.nn.functional.interpolate(img.unsqueeze(0), self.size, mode=self.interpolation,
#                                                align_corners=False).squeeze(0)

#     def __repr__(self):
#         interpolate_str = self.interpolation
#         return self.__class__.__name__ + '(size={0}, scale={1}, ratio={2}, interpolation={3})'.format(self.size,
#                                                                                                       self.scale,
#                                                                                                       self.ratio,
#                                                                                                       interpolate_str)