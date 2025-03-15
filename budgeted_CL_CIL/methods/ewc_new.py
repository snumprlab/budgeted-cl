################################
# This code is referred by
# https://github.com/GT-RIPL/Continual-Learning-Benchmark
################################

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


class EWCpp(ER):

    def __init__(
        self, train_datalist, test_datalist, device, **kwargs
    ):
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        # except for last layers.
        self.parameters = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }
        # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        self.online_reg = True
        self.samples_per_task = kwargs["samples_per_task"]

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.parameters.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized

    def online_step(self, sample, sample_num, n_worker):
        super().online_step(sample, sample_num, n_worker)
        if sample_num % self.samples_per_task == 0:
            self.online_after_task(sample_num)

    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.parameters.items():
                    # print("importance[n]", importance[n].shape)
                    # print("p", p.shape)
                    # print("task_param[n]", task_param[n].shape)
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.parameters.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            self.before_model_update()
            self.optimizer.zero_grad()

            old_params = {n: p.clone().detach() for n, p in self.parameters.items()}
            old_grads = {n: p.grad.clone().detach() for n, p in self.parameters.items() if p.grad is not None}

            logit, loss = self.model_forward(x, y)

            self.total_flops += (len(x) * self.backward_flops)
            
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
            self.after_model_update()
            new_params = {n: p.clone().detach() for n, p in self.parameters.items()}
            new_grads = {
                n: p.grad.clone().detach() for n, p in self.parameters.items() if p.grad is not None
            }
            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    @torch.no_grad()
    def online_after_task(self, cur_iter):
        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.parameters.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance()

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")

    @torch.no_grad()
    def update_fisher_and_score(self, new_params, old_params, new_grads, old_grads, epsilon=0.001):
        for n, _ in self.parameters.items():
            if n in old_grads:
                #new_p = new_params[n]
                #old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                '''
                if torch.isinf(new_p).sum()+torch.isinf(old_p).sum()+torch.isinf(new_grad).sum()+torch.isinf(old_grad).sum():
                    continue
                if torch.isnan(new_p).sum()+torch.isnan(old_p).sum()+torch.isnan(new_grad).sum()+torch.isnan(old_grad).sum():
                    continue
                self.epoch_score[n] += (old_grad-new_grad) * (new_p - old_p) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            (old_grad-new_grad) * (new_p - old_p),
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                '''
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2

    @torch.no_grad()
    def calculate_importance(self):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.parameters.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.parameters.items():
            importance[n] = self.fisher[-1][n]
            self.epoch_score[n] = self.parameters[n].clone().detach().fill_(0)
        return importance