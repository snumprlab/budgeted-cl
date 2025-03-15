# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity

from methods.er_new import ER

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
# from torchdistill.losses.mid_level import SPKDLoss

class SPKDLoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_outputs, teacher_outputs):
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1) # slope : self.linear
        
    def forward(self, x):
        out = self.linear(x)
        return out


class EGERIA(ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.sp_loss = SPKDLoss()
        self.period_update_rm_model = 100
        self.target_layers = ['initial.block', \
                              'group1.blocks.block0', 'group1.blocks.block1', 'group1.blocks.block2', 'group1.blocks.block3.conv1.block.1', 'group1.blocks.block4', \
                              'group2.blocks.block0', 'group2.blocks.block1', 'group2.blocks.block2', 'group2.blocks.block3.conv1.block.1', 'group2.blocks.block4', \
                              'group3.blocks.block0', 'group3.blocks.block1', 'group3.blocks.block2', 'group3.blocks.block3.conv1.block.1', 'group3.blocks.block4', \
                              'fc']
        self.stable_counters = [0 for _ in range(len(self.target_layers))]
        self.plasticities = [[] for _ in range(len(self.target_layers))]
        self.slopes = [[] for _ in range(len(self.target_layers))]
        self.threshold = [float("-inf") for _ in range(len(self.target_layers))]
        self.W = 10
        self.threshold_determine_num = 5
        self.plasticity_buffersize = self.W
        self.reference_model = copy.deepcopy(self.model)
        # self.reference_model.eval()
        # self.reference_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        # self.reference_model = torch.ao.quantization.quantize_dynamic(
        #                         copy.deepcopy(self.reference_model),  # the original model
        #                         {'initial', 'group1', 'group2', 'group3', 'pool', 'fc'},
        #                         dtype=torch.qint8).to(self.device)
        # self.reference_model = torch.ao.quantization.prepare(self.reference_model)
        # self.reference_model = torch.ao.quantization.convert(self.reference_model)


    def get_slope(self, yN):
        linreg_model = LinearRegressionModel().to(self.device)
        xN = np.zeros(len(yN))
        yN = np.array(yN)
        lr = 0.005
        epochs = 70
        linreg_criterion = torch.nn.MSELoss().to(self.device)
        linreg_optimizer = torch.optim.SGD(linreg_model.parameters(), lr=lr)
        for epoch in range(epochs+1):
            # Convert data to torch tensors and move to device
            # inputs = Variable(torch.from_numpy(xN).to(self.device))
            # labels = Variable(torch.from_numpy(yN).to(self.device)).to(torch.float32)
            # inputs = inputs.to(torch.float32).squeeze().unsqueeze(dim=-1)
            inputs = torch.Tensor(xN).to(self.device).squeeze().unsqueeze(dim=-1)
            labels = torch.Tensor(yN).to(self.device)
            outputs = linreg_model(inputs)
            linreg_optimizer.zero_grad()
            loss = linreg_criterion(outputs, labels)
            loss.backward()
            linreg_optimizer.step()

            # Print statement for every 50 epochs
            # if epoch%50 == 0:
            #     print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        
        return linreg_model.linear.weight

    def check_plasticity(self, model_output, refer_output, group_num):
        plasticity = self.sp_loss(model_output, refer_output)
        self.plasticities[group_num].append(plasticity.item())
        if len(self.plasticities[group_num]) > self.plasticity_buffersize:
            del self.plasticities[group_num][0]
        
        slope = abs(self.get_slope(self.plasticities[group_num]).item())
        self.slopes[group_num].append(slope)
        if len(self.slopes[group_num]) == self.threshold_determine_num and self.threshold[group_num] == float("-inf"):
            new_threshold = np.mean(self.slopes[group_num]) * 0.2
            print("group_num", group_num, "new_threshold", new_threshold)
            print()
            self.threshold[group_num] = new_threshold
        
        print("group_num", group_num,"slope", slope, "self.threshold[group_num]", self.threshold[group_num])
        if slope < self.threshold[group_num]:
            self.stable_counters[group_num] += 1
        else:
            self.stable_counters[group_num] = 0
    
    def online_after_task(self):
        self.stable_counters = [0 for _ in range(len(self.target_layers))]
        self.plasticities = [[] for _ in range(len(self.target_layers))]
        # self.threshold = [float("-inf") for _ in range(len(self.target_layers))]

    def freeze_block(self, group_num):
        print("freeze group", group_num)
        # TODO 앞쪽 block들이 다 freeze 되어 있을때만?
        for name, param in self.model.named_parameters():
            if self.target_layers[group_num] in name:
                print("freezing", name)
                param.grad=False

    def model_forward(self, x, y):
        with torch.cuda.amp.autocast(self.use_amp):            
            model_logit, model_features = self.model(x, get_features=True, get_block_features=True)
            refer_logit, refer_features = self.reference_model(x, get_features=True, get_block_features=True)
            
            group_num = 0
            for model_feature, refer_feature in zip(model_features, refer_features): # initial 
                if self.stable_counters[group_num] >= self.W:
                    self.freeze_block(group_num)
                    continue

                if type(model_feature) == list:
                    for model_block_feature, refer_block_feature in zip(model_feature, refer_feature):
                        sp_loss = self.check_plasticity(model_block_feature, refer_block_feature, group_num)
                        group_num += 1
                else: # initial block
                    sp_loss = self.check_plasticity(model_feature, refer_feature, group_num)
                    group_num += 1

            loss = self.criterion(model_logit, y)

        self.total_flops += (len(y) * self.forward_flops)
        return model_logit, loss

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


        if self.sample_num % self.period_update_rm_model:
            # self.reference_model = torch.ao.quantization.quantize_dynamic(
            #                         copy.deepcopy(self.model),  # the original model
            #                         {'initial', 'group1', 'group2', 'group3', 'pool', 'fc'},
            #                         dtype=torch.qint8).to(self.device)
                                    # {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
            self.reference_model = copy.deepcopy(self.model)
        return total_loss / iterations, correct / num_data

