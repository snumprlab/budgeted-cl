import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configuration import config
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist

from utils.method_manager_new import select_method

def main():
    args = config.base_parser()
    num_samples = {'cifar10': 50000, '5_dataset': 260103, 'cifar100': 50000, 'clear10':30000, 'clear100':100000, 'tinyimagenet': 100000, 'imagenet': 1281165}
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

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # get datalist
    print("args.dataset", args.dataset, "num_samples", num_samples[args.dataset])
    train_datalist, cls_dict, cls_addition = get_train_datalist(args.dataset, args.sigma, args.repeat, args.init_cls, args.rnd_seed)
    test_datalist = get_test_datalist(args.dataset)
    samples_cnt = 0

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]

    logger.info(f"Select a CIL method ({args.mode})")
    method = select_method(args, train_datalist, test_datalist, device)

    print("\n###flops###\n")
    #method.get_flops_parameter()

    eval_results = defaultdict(list)
    task_records = defaultdict(list)
    
    samples_cnt = 0
    task_id = 0

    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)

    for i, data in enumerate(train_datalist):

        # explicit task boundary for twf
        if samples_cnt % args.samples_per_task == 0 and args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l", "trire"]:
            method.online_before_task(task_id)
            task_id += 1

        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict,
                                               cls_addition, data["time"])
            eval_results["test_acc"].append(eval_dict['avg_acc'])
            eval_results["percls_acc"].append(eval_dict['cls_acc'])
            eval_results["data_cnt"].append(samples_cnt)
        
        if (args.mode in ["remind"]) and samples_cnt == args.baseinit_samples:
            method.finalize_baseinit()
        
        if samples_cnt % args.samples_per_task == 0 and (args.mode in ["memo", "xder", "afec", "sparcl", "trire"]) and samples_cnt != num_samples[args.dataset]:
            method.online_after_task()
        
        if samples_cnt % args.samples_per_task == 0:
            task_records["cls_acc"].append(eval_dict["cls_acc"])

        
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, cls_dict, cls_addition,
                                           data["time"])

    A_last = eval_dict['avg_acc']

    if args.mode == 'gdumb':
        eval_results = method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker, cls_dict, cls_addition)

    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_per_cls.npy', eval_results['percls_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])

    # Forgetting (F)
    cls_acc = np.array(task_records["cls_acc"])
    acc_diff = []
    for j in range(n_classes):
        if np.max(cls_acc[:-1, j]) > 0:
            acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    F_last = np.mean(acc_diff)

    KLR_avg = np.mean(method.knowledge_loss_rate[1:])
    KGR_avg = np.mean(method.knowledge_gain_rate)

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc:6f} | A_last {A_last:6f} | KLR_avg {KLR_avg:6f} | KGR_avg {KGR_avg:6f} | Total FLOPs {method.total_flops:4f} | F_last {F_last}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
