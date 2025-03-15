import os
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')
dir = 'cifar10'

def print_from_log(exp_name, seeds=(1, 2, 3)):
    A_auc = []
    A_last = []
    A_online = []
    F_last = []
    IF_avg = []
    KG_avg = []
    FLOPS = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        lines = f.readlines()
        try:
            for line in lines:
                if 'gdumb' in exp_name:
                    if 'Test' in line:
                        list = line.split(' ')
                        a_last = float(list[13])*100
                if 'A_auc' in line:
                    list = line.split(' ')
                    A_auc.append(float(list[4])*100)
                    if 'gdumb' in exp_name:
                        A_last.append(a_last)
                    else:
                        A_last.append(float(list[7])*100)
                        IF_avg.append(float(list[10])*100)
                        KG_avg.append(float(list[13])*100)
                        F_last.append(float(list[-1])*100)
                    FLOPS.append(float(list[-4])/100)
                    break
        except Exception as e:
            pass
            #print(exp_name, e)
    if np.isnan(np.mean(A_auc)):
        pass
    else:
        # print(IF_avg[1:])
        print(f'Exp:{exp_name:<50} \t\t\t {np.mean(A_auc):.2f}/{np.std(A_auc):.2f} \t {np.mean(A_last):.2f}/{np.std(A_last):.2f} \t  {np.mean(IF_avg):.2f}/{np.std(IF_avg):.2f}  \t  {np.mean(KG_avg):.2f}/{np.std(KG_avg):.2f} \t {np.mean(F_last):.2f}/{np.std(F_last):.2f}  \t  {np.mean(FLOPS):.2f}/{np.std(FLOPS):.2f}|')

print(" " * 72, f"A_auc  \t A_last  \t IF_avg \t KG_avg \t F_last \t FLOPS")

exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])

for exp in exp_list:
    try:
        print_from_log(exp)
    except:
        pass

