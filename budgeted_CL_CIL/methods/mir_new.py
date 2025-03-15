import logging
import copy

import torch
import torch.nn.functional as F

from methods.cl_manager import MemoryBase
from methods.er_new import ER
from utils.data_loader import MultiProcessLoader, StreamDataset

logger = logging.getLogger()


class MIR(ER):
    def __init__(
        self,  train_datalist, test_datalist, device, **kwargs
    ):
        self.cand_size = kwargs['mir_cands']
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )

    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = MemoryBase(self.memory_size)
        self.cand_loader = MultiProcessLoader(self.n_worker, self.cls_dict, self.test_transform, self.data_dir, transform_on_gpu=False, cpu_transform=None, device=self.device, use_kornia=False, transform_on_worker=False)
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

        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.cand_loader.add_new_class(self.memory.cls_dict)

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

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        cand_batch = self.cand_loader.get_batch()
        self.load_batch()
        return batch, cand_batch

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
            self.cand_loader.load_batch(self.waiting_batch[0][self.temp_batch_size:])
            del self.waiting_batch[0]

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.cand_size))

    def online_train(self, iterations=1):
        self.model.train()
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            data, cands = self.get_batch()
            memory_cands_test = cands
            str_x = data['image'][:self.temp_batch_size]
            str_y = data['label'][:self.temp_batch_size]
            x = str_x.to(self.device)
            y = str_y.to(self.device)

            logit = self.model(x)
            loss = self.criterion(logit, y)
            self.optimizer.zero_grad()
            loss.backward()
            grads = {}
            for name, param in self.model.named_parameters():
                grads[name] = param.grad.data

            if cands is not None:
                lr = self.optimizer.param_groups[0]['lr']
                new_model = copy.deepcopy(self.model)
                for name, param in new_model.named_parameters():
                    param.data = param.data - lr * grads[name]
                x = memory_cands_test['image']
                y = memory_cands_test['label']
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit_pre = self.model(x)
                            logit_post = new_model(x)
                            pre_loss = F.cross_entropy(logit_pre, y, reduction='none')
                            post_loss = F.cross_entropy(logit_post, y, reduction='none')
                            scores = post_loss - pre_loss
                    else:
                        logit_pre = self.model(x)
                        logit_post = new_model(x)
                        pre_loss = F.cross_entropy(logit_pre, y, reduction='none')
                        post_loss = F.cross_entropy(logit_post, y, reduction='none')
                        scores = post_loss - pre_loss
                    
                    
                    self.total_flops += (3 * len(logit_pre)) / 10e9
                    self.total_flops += (2 * len(x) * self.forward_flops)    
                selected_samples = torch.argsort(scores, descending=True)[:self.memory_batch_size]

                mem_x_cands = data["image"][self.temp_batch_size:]
                mem_y_cands = data["label"][self.temp_batch_size:]
                mem_x = mem_x_cands[selected_samples.cpu()]
                mem_y = mem_y_cands[selected_samples.cpu()]
                
                '''
                mem_indices = memory_cands['index'][selected_samples]
                self.memory.update_std(mem_indices)
                '''
                
                x = torch.cat([str_x, mem_x])
                y = torch.cat([str_y, mem_y])
                x = x.to(self.device)
                y = y.to(self.device)


            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x, y)
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
            self.total_flops += (self.batch_size * self.backward_flops)
            
        return total_loss / iterations, correct / num_data
