import logging.config
import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.my_augment import Kornia_Randaugment
from utils.train_utils import get_transform
from utils.method_manager import select_method
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor

'''
class DataAugmentation(nn.Module):   

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)



    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )

        ##### check transforms 
        # print("self.transform")
        # print(self.transforms)

        x_out = self.transforms(x)  # BxCxHxW
        #x_out, _ = self.cutmix(x_out)
        return x_out
'''



def main():
    args = config.base_parser()

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    #writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.rnd_seed}')

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    print("args.use_human_training", args.use_human_training)
    train_transform, test_transform = get_transform(args.dataset, args.transforms, args.gpu_transform, args.use_kornia)#, args.use_human_training)

    logger.info(f"Main using train-transforms {train_transform}")

    logger.info(f"Select a CIL method ({args.mode})")
    if args.mode == 'ours' and args.weight_option == 'loss':
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )
    
    print()
    print("###flops###")
    method.get_flops_parameter()
    print()

    eval_results = defaultdict(list)

    if args.online_fc_mode != 'none':
        eval_results_2 = defaultdict(list)
    samples_cnt = 0
    task_id = 0
    test_datalist = get_test_datalist(args.dataset)

    # get datalist
    train_datalist, cls_dict, cls_addition = get_train_datalist(args.dataset, args.sigma, args.repeat, args.init_cls, args.rnd_seed)

    method.n_samples(len(train_datalist))

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]

    #train_datalist = train_datalist[:5000] + train_datalist[10000:15000]

    for i, data in enumerate(train_datalist):

        # explicit task boundary for twf
        if samples_cnt % args.samples_per_task == 0 and (args.mode == "bic" or args.mode == "ewc++"):
            method.online_before_task(task_id)
            task_id += 1

        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        '''
        if args.max_validation_interval is not None and args.min_validation_interval is not None:
            if samples_cnt % method.get_validation_interval() == 0:
                method.online_validate(samples_cnt, 512, args.n_worker)
        else:
            if samples_cnt % args.val_period == 0:
                method.online_validate(samples_cnt, 512, args.n_worker)
        '''

        '''
        ### for using validation set ###
        if samples_cnt % args.val_period == 0:
            method.online_validate(samples_cnt, 512, args.n_worker)
        ''' 
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict, cls_addition, data["time"])
            eval_results["test_acc"].append(eval_dict['avg_acc'])
            eval_results["percls_acc"].append(eval_dict['cls_acc'])
            eval_results["online_acc"].append(eval_dict['online_acc'])
            eval_results["data_cnt"].append(samples_cnt)
            if method.f_calculated:
                eval_results["forgetting_acc"].append(eval_dict['cls_acc'])
            if args.online_fc_mode != 'none':
                eval_dict_2 = method.online_evaluate_2(test_datalist, samples_cnt, 512, args.n_worker, cls_dict,
                                                   cls_addition, data["time"])
                eval_results_2["test_acc"].append(eval_dict_2['avg_acc'])
                eval_results_2["percls_acc"].append(eval_dict_2['cls_acc'])
                eval_results_2["online_acc"].append(eval_dict_2['online_acc'])
                eval_results_2["data_cnt"].append(samples_cnt)
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict, cls_addition, data["time"])
        if args.online_fc_mode != 'none':
            eval_dict_2 = method.online_evaluate_2(test_datalist, samples_cnt, 512, args.n_worker, cls_dict,
                                                 cls_addition, data["time"])

    A_last = eval_dict['avg_acc']

    if args.mode == 'gdumb':
        eval_results = method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker, cls_dict, cls_addition)

    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])
    A_online = np.mean(eval_results["online_acc"])

    if args.online_fc_mode != 'none':
        A_last_2 = eval_dict_2['avg_acc']
        np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_2.npy', eval_results_2['test_acc'])
        np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time_2.npy', eval_results_2['data_cnt'])
        A_auc_2 = np.mean(eval_results_2["test_acc"])
        A_online_2 = np.mean(eval_results_2["online_acc"])

    cls_acc = np.array(eval_results["forgetting_acc"])
    acc_diff = []
    for j in range(n_classes):
        if np.max(cls_acc[:-1, j]) > 0:
            acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    F_last = np.mean(acc_diff)
    IF_avg = np.mean(method.forgetting[1:])
    KG_avg = np.mean(method.knowledge_gain[1:])
    Total_flops = method.get_total_flops()

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} | A_online {A_online} | A_last {A_last} | F_last {F_last} | IF_avg {IF_avg} | KG_avg {KG_avg} | Total_flops {Total_flops}")
    # logger.info(f"A_auc {A_auc} | A_online {A_online} | A_last {A_last} | F_last {F_last}")


    if args.online_fc_mode != 'none':
        logger.info(f"A_auc {A_auc_2} | A_online {A_online_2} | A_last {A_last_2}")

if __name__ == "__main__":
    main()
