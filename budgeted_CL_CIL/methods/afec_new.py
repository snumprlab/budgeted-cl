import logging
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_new import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset
from utils.train_utils import cycle

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

import copy
from utils.train_utils import select_optimizer, select_scheduler


class AFEC(ER):
    def __init__(
        self, train_datalist, test_datalist, device, **kwargs
    ):
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        self.model_old = copy.deepcopy(self.model)
        self.model_emp = copy.deepcopy(self.model)
        self.model_emp_temp = copy.deepcopy(self.model)
        self.model_pt = None
        
        self.fisher = None
        self.fisher_emp = None
        self.epoch_fisher = {}
        
        self.parameters = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }
        self.parameters_emp = {
            n: p for n, p in list(self.model_emp.named_parameters())[:-2] if p.requires_grad
        }
        self.parameters_old = {
            n: p for n, p in list(self.model_old.named_parameters())[:-2] if p.requires_grad
        }
        
        for n, p in self.parameters.items():
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )
        
        self.optimizer_emp = select_optimizer(self.opt_name, self.lr, self.model_emp)
        self.scheduler_emp = select_scheduler(self.sched_name, self.optimizer_emp)
        
        self.lamb = 10000
        self.lamb_emp = 1
        self.task_count = 0
        
    def add_new_class(self, class_name):
        if hasattr(self.model, 'fc'):
            fc_name = 'fc'
        elif hasattr(self.model, 'head'):
            fc_name = 'head'
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        model_fc = getattr(self.model, fc_name)
        prev_weight = copy.deepcopy(model_fc.weight.data)
        prev_bias = copy.deepcopy(model_fc.bias.data)
        
        model_emp_fc = getattr(self.model_emp, fc_name)
        emp_prev_weight = copy.deepcopy(model_emp_fc.weight.data)
        emp_prev_bias = copy.deepcopy(model_emp_fc.bias.data)
        
        setattr(self.model, fc_name, nn.Linear(model_fc.in_features, self.num_learned_class).to(self.device))
        setattr(self.model_emp, fc_name, nn.Linear(model_emp_fc.in_features, self.num_learned_class).to(self.device))
        model_fc = getattr(self.model, fc_name)
        model_emp_fc = getattr(self.model_emp, fc_name)
        with torch.no_grad():
            if self.num_learned_class > 1:
                model_fc.weight[:self.num_learned_class - 1] = prev_weight
                model_fc.bias[:self.num_learned_class - 1] = prev_bias
                model_emp_fc.weight[:self.num_learned_class - 1] = emp_prev_weight
                model_emp_fc.bias[:self.num_learned_class - 1] = emp_prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        for param in self.optimizer_emp.param_groups[1]['params']:
            if param in self.optimizer_emp.state.keys():
                del self.optimizer_emp.state[param]
        del self.optimizer.param_groups[1]
        del self.optimizer_emp.param_groups[1]
        self.optimizer.add_param_group({'params': model_fc.parameters()})
        self.optimizer_emp.add_param_group({'params': model_emp_fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
    
    def online_train_emp(self, x, y):
        fisher={}
        for n,p in self.parameters_emp.items():
            fisher[n]=0*p.data
        
        self.optimizer_emp.zero_grad()
        self.model_emp.train()
        with torch.cuda.amp.autocast(self.use_amp):
            logit = self.model_emp(x)
            loss = self.criterion(logit, y)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_emp)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer_emp.step()
        
        self.scheduler_emp.step()
        
        for n,p in self.parameters_emp.items():
            fisher[n]+=p.data.pow(2)
        
        return fisher
    
    def regularization_loss(self):
        # Regularization for all previous tasks
        loss_reg=0
        loss_reg_emp = 0

        if self.task_count>0:
            for (name,param),(_,param_old) in zip(self.parameters.items(), self.parameters_old.items()):
                if 'last' not in name:
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

            self.parameters_temp = {
            n: p for n, p in list(self.model_emp_temp.named_parameters())[:-2] if p.requires_grad
        }
            for (name,param),(_,param_old) in zip(self.parameters.items(),self.parameters_temp.items()):
                if 'last' not in name:
                    loss_reg_emp+=torch.sum(self.fisher_emp[name]*(param_old-param).pow(2))/2

        return self.lamb*loss_reg + self.lamb_emp*loss_reg_emp

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return
    
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            self.x, self.y = x, y
            self.before_model_update()

            # Train model_emp and Calculate fisher
            self.fisher_emp = self.online_train_emp(x, y)
            self.model_emp_tmp = copy.deepcopy(self.model_emp)
            self.model_emp_tmp.train()
            self.freeze_model(self.model_emp_tmp)

            # Train model
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x,y)
            _, preds = logit.topk(self.topk, 1, True, True)
            with torch.cuda.amp.autocast(self.use_amp):
                reg_loss = self.regularization_loss()
                loss += reg_loss
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.update_fisher_and_score()

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
         # Checked: it is better than the other option

        return total_loss / iterations, correct / num_data
    
    @torch.no_grad()
    def update_fisher_and_score(self):
        new_grads = {
                n: p.grad.clone().detach() for n, p in self.model.named_parameters() if p.grad is not None
            }
        for n,p in self.parameters.items():
            if (self.epoch_fisher[n] == 0).all():  # First time
                self.epoch_fisher[n] = new_grads[n] ** 2
            else:
                self.epoch_fisher[n] += new_grads[n] ** 2
            
        
    @torch.no_grad()
    def online_after_task(self):
        print("end task")
        self.model_old = copy.deepcopy(self.model)
        self.model_old.train()
        self.freeze_model(self.model_old) # Freeze the weights
        self.parameters_old = {
            n: p for n, p in list(self.model_old.named_parameters())[:-2] if p.requires_grad
        }
        
        # Calculate model importance
        if self.task_count>0:
            fisher_old={}
            for n,_ in self.parameters.items():
                fisher_old[n]=self.fisher[n].clone()
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            # for n,_ in self.parameters:
                self.fisher[n]=(self.epoch_fisher[n]+fisher_old[n]*self.task_count)/(self.task_count+1)       # Checked: it is better than the other option
        else:
            self.fisher = copy.deepcopy(self.epoch_fisher)
                
         # Save the weight and importance of weights of current task
        self.task_count += 1
        
        for n, p in self.parameters.items():
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
        
        