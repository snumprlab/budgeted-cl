# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class BASELINE(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def update_memory(self, sample):
        self.reservoir_memory(sample)

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
            
        self.update_memory(sample)
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if self.future_num_updates >= 1:
            self.temp_future_batch = []
            self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.memory.retrieval(self.memory_batch_size))

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)






