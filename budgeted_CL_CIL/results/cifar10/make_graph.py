import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter1d
import os

mpl.rcParams.update({
    'font.size'           : 10.0        ,
    'font.sans-serif'     : 'Arial'    ,
    'xtick.major.size'    : 2          ,
    'xtick.minor.size'    : 1.5        ,
    'xtick.major.width'   : 0.75       ,
    'xtick.minor.width'   : 0.75       ,
    'xtick.labelsize'     : 8.0        ,
    'xtick.direction'     : 'in'       ,
    'xtick.top'           : True       ,
    'ytick.major.size'    : 2          ,
    'ytick.minor.size'    : 1.5        ,
    'ytick.major.width'   : 0.75       ,
    'ytick.minor.width'   : 0.75       ,
    'ytick.labelsize'     : 8.0        ,
    'ytick.direction'     : 'in'       ,
    'xtick.major.pad'     : 2          ,
    'xtick.minor.pad'     : 2          ,
    'ytick.major.pad'     : 2          ,
    'ytick.minor.pad'     : 2          ,
    'ytick.right'         : True       ,
    'savefig.dpi'         : 600        ,
    'savefig.transparent' : True       ,
    'axes.linewidth'      : 0.75       ,
    'lines.linewidth'     : 1.0,
    'axes.prop_cycle': mpl.cycler('color',
                                  ['195190', 'DD4132', 'E3B448', '1B6535', 'B1624E', '595959', '724685', 'FF8C00',
                                   'CBCE91', '3A6B35', 'EF9DAF'])
})

def plot_from_result(ax, exp_name, label, lw=1.0, alpha=1.0, seeds=(1, 2, 3), smooth=False, zorder=1):
    seed_data = []
    for i in seeds:
        seed_data.append(np.load(exp_name + f'/seed_{i}_eval.npy'))
    data = sum(seed_data) / len(seed_data)

    data_time = np.load(exp_name + f'/seed_{seeds[0]}_eval_time.npy')
    dup = []
    for i in range(len(data)-1):
        if data_time[i] == data_time[i+1]:
            dup.append(i)
    data = np.delete(data, dup)
    data_time = np.delete(data_time, dup)
    if smooth:
        data = gaussian_filter1d(data, sigma=2)
    ax.plot(data_time, data, label=label, lw=lw, alpha=alpha, zorder=zorder)

def plot_loss_decrease(ax, exp_name, label, lw=1.0, alpha=1.0, seeds=(1,), smooth=False, zorder=1, train=True):
    seed_data = []
    for i in seeds:
        if train:
            filename = f'/seed_{i}_bn_train.npy'
            timename = f'/seed_{i}_bn_train_time.npy'
        else:
            filename = f'/seed_{i}_bn_eval.npy'
            timename = f'/seed_{i}_bn_eval_time.npy'
        seed_data.append(np.load(exp_name + filename))
    data = sum(seed_data) / len(seed_data)

    data_time = np.load(exp_name + timename)
    if 'only' in exp_name:
        data = data[:2:]
        data_time = data_time[:2:]
    if smooth:
        data = gaussian_filter1d(data, sigma=5)
    ax.plot(data_time, data, label=label, lw=lw, alpha=alpha, zorder=zorder)

def plot_forgetting(ax, exp_name, label, type, lw=1.0, alpha=1.0, seeds=(1, 2, 3), smooth=False, zorder=1):
    seed_data = []
    for i in seeds:
        seed_data.append(np.load(exp_name + f'/seed_{i}_{type}.npy'))
    data = sum(seed_data) / len(seed_data)
    data_time = np.load(exp_name + f'/seed_{seeds[0]}_forgetting_time.npy')
    dup = []
    for i in range(len(data) - 1):
        if data_time[i] == data_time[i + 1]:
            dup.append(i)
    data = np.delete(data, dup)
    data_time = np.delete(data_time, dup)
    if smooth:
        data = gaussian_filter1d(data, sigma=2)
    ax.plot(data_time, data, label=label, lw=lw, alpha=alpha, zorder=zorder)

def plot_from_log(ax, exp_name, label, lw=1.0, alpha=1.0, seeds=(1, 2, 3), smooth=False, zorder=1):
    # smooth=False
    all_data = []
    all_online = []
    all_time = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        seed_time = []
        seed_data = []
        online_data = []
        lines = f.readlines()
        for line in lines:
            if ' Test ' in line:
                list = line.split(' ')
                if len(seed_time) == 0:
                    seed_time.append(int(list[7]))
                    seed_data.append(float(list[13]))
                    online_data.append(float(list[16]))
                elif int(list[7]) != seed_time[-1]:
                    seed_time.append(int(list[7]))
                    seed_data.append(float(list[13]))
                    online_data.append(float(list[16]))
        all_data.append(np.array(seed_data))
        all_online.append(np.array(online_data))
        all_time.append(np.array(seed_time))
    length = min([len(data) for data in all_data])
    all_data = [seed_data[:length] for seed_data in all_data]
    all_online = [seed_online[:length] for seed_online in all_online]
    A_auc = [np.mean(seed_data) for seed_data in all_data]
    A_last = [seed_data[-1] for seed_data in all_data]
    A_online = [np.mean(seed_online) for seed_online in all_online]
    data = sum(all_data) / len(all_data)
    if smooth:
        data = gaussian_filter1d(data, sigma=2)
    data_time = all_time[0][:length]
    # label += f'(A_auc: {np.mean(A_auc):.4f}/{np.std(A_auc):.4f} | A_last: {np.mean(A_last):.4f}/{np.std(A_last):.4f} | A_online: {np.mean(A_online):.4f}/{np.std(A_online):.4f})'
    ax.plot(data_time, data, label=label, lw=lw, alpha=alpha, zorder=zorder)


def plot_flops_from_log(ax, exp_name, label, lw=1.0, alpha=1.0, seeds=(1, 2, 3), smooth=False, zorder=1):
    all_data = []
    all_time = []
    for i in seeds:
        f = open(f'{exp_name}/seed_{i}.log', 'r')
        seed_time = []
        seed_data = []
        lines = f.readlines()
        for line in lines:
            if ' Test ' in line:
                list = line.split(' ')
                if len(seed_time) == 0:
                    seed_time.append(int(list[7]))
                    seed_data.append(float(list[16]))
                elif int(list[7]) != seed_time[-1]:
                    seed_time.append(int(list[7]))
                    seed_data.append(float(list[16]))
        all_data.append(np.array(seed_data))
        all_time.append(np.array(seed_time))
    length = min([len(data) for data in all_data])
    all_data = [seed_data[:length] for seed_data in all_data]
    data = sum(all_data) / len(all_data)
    if smooth:
        data = gaussian_filter1d(data, sigma=2)
    data_time = all_time[0][:length]
    ax.plot(data_time, data, label=label, lw=lw, alpha=alpha, zorder=zorder)

def plot_acc_and_flops(ax1, ax2, exp_name, label, seeds=(1,), zorder=1):
    plot_from_log(ax1, exp_name, label, 1, 1, seeds=seeds, smooth=True, zorder=zorder)
    plot_flops_from_log(ax2, exp_name, label, 1, 1, seeds=seeds, smooth=False, zorder=zorder)
width = 4.5
height = width * 0.9


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), sharex=True)

exp_list = os.listdir()
for exp in exp_list:
    if exp != 'make_graph.py' and ".pdf" not in exp:
        k_value = exp.split('_')[-6]
        t_value = exp.split('_')[-6]
        if k_value == "2":
            print(exp)
            plot_acc_and_flops(ax1, ax2, exp, exp)

#plot_acc_and_flops(ax1, ax2, 'test', 'ours_full')


# plot_from_log(ax1, 'corr_relu_layer_10_4_100', 'corr_layer', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'corr_with_relu_10_4_100', 'corr', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'ours_distill_10_4_100', 'normal', 1, 1, seeds=(1, 2, 3), smooth=True)

# plot_forgetting(ax1, 'ours_distill_10_4_100', 'Forgetting', 'forgetting', 1, 1, seeds=(1,), smooth=True)
# plot_forgetting(ax1, 'ours_distill_10_4_100', 'Knowledge Gain', 'knowledge_gain', 1, 1, seeds=(1,), smooth=True)
# plot_forgetting(ax1, 'ours_distill_10_4_100', 'Retained Knowledge', 'retained_knowledge', 1, 1, seeds=(1,), smooth=True)
# plot_forgetting(ax1, 'ours_distill_10_4_100', 'Total Knowledge', 'total_knowledge', 1, 1, seeds=(1,), smooth=True)
# plot_loss_decrease(ax1, 'bn_alter_analysis_4r', 'alter_eval', 1, 1, seeds=(1,), smooth=True, train=False)
# plot_loss_decrease(ax1, 'bn_eval_analysis_4r', 'eval_only', 1, 1, seeds=(1,), smooth=True, train=False)
#
# plot_from_log(ax1, 'test_batchstat_modified', 'ours_1', 1, 1, seeds=(1,), smooth=True, zorder=1)
# plot_from_log(ax1, 'nofreeze', 'nofreeze', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'our_1.3', 'ours_1.3', 1, 1, seeds=(1,), smooth=True, zorder=1)
# plot_flops_from_log(ax2, 'test_batchstat_modified', 'ours_1', 1, 1, seeds=(1,), smooth=True, zorder=1)
# plot_flops_from_log(ax2, 'nofreeze', 'nofreeze', 1, 1, seeds=(1,), smooth=True)
# plot_flops_from_log(ax2, 'our_1.3', 'ours_1.3', 1, 1, seeds=(1,), smooth=True, zorder=1)


# plot_from_log(ax1, 'ours_1.0_uf0.05', 'ours_0.05', 1, 1, seeds=(1, ), smooth=True, zorder=3)
# plot_from_log(ax1, 'ours_1.0_uf0.1', 'ours_0.1', 1, 1, seeds=(1, ), smooth=True, zorder=3)
# ax2.plot(np.arange(0, 50000), np.arange(0, 50000)*110/50000, c='k', ls='--', lw=0.5)

# plot_from_log(ax1, '_prev_220822/ours_distill_10_4_100', 'distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, '_prev_220822/baseline_10_4_100', 'base', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'test', 'test', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'test_2', 'test_2', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'difflrdistill', 'test_3', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'difflrdistill_16', 'test_3', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, '../../temp/er_10_4_100', 'ER', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp/der_10_4_100', 'DER', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp/baseline_10_4_100', 'Baseline', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'distill_baseline', 'EMA distill', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp/distill_baseline', 'EMA_Distill', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'distill_baseline', 'distill', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp/ours_no_normalize', 'DMA_Distill', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp/ours', 'Ours (DMA_Distill + Normalize)', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '0prev_220822/base_norm_distill', 'norm_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'base_norm_distill_train', 'norm_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'base_norm_distill_eval', 'norm_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'base_norm_distill_pred', 'norm_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'pma_model_distill_9999_99976', 'period_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'pma_model_distill_9999_99976_train', 'period_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'dma_9999_99976_normloss', 'normloss', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'dma_9999_99976_normloss_clspred', 'clspred', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'dma_9999_99976_normloss_clspred2', 'clspred2', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'dma_9999_99976_normloss2_clspred', 'clspred', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours', 'Ours', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'er_10_5_100_none', 'Ours', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'der_10_4_100_none', 'Ours', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'er_10_4_100_autoaug', 'Ours', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'der_10_4_100_autoaug', 'Ours', 1, 1, seeds=(1,), smooth=True)

# plot_from_log(ax1, 'ours_1r_sample_fc_none', 'base', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_1r_batch_fc_none', 'Ours', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_1r_cumul_fc_train', 'Ours', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_1r_lr', 'Ours_lin', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, "ours_1r_linear_05_125", 'Ours_lin', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, "ours_1r_linear_05_125_", 'Ours_lin', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_1r_10_lr', 'Ours', 1, 1, seeds=(1, 2, 3), smooth=True)

# plot_from_log(ax1, 'ours_inc', 'clspred2', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'pma_model_distill_9999_99976_eval', 'period_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'pma_first_period_distil', 'period_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'pma_model_distill_999_99', 'dma_distill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'norm_pred_based', 'norm_distill', 1, 1, seeds=(1,), smooth=True)

# plot_from_log(ax1, '../../temp 100/baseline_10_4_100', 'Baseline', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, '../../temp 100/clib_10_4_100', 'CLIB', 1, 1, seeds=(1, 2, 3), smooth=True)

# plot_from_log(ax1, '../../temp 100/ours_100', 'Ours', 1, 1, seeds=(1, 2, 3), smooth=True)


# plot_from_result(ax1, 'bntest_4r_random100', 'train', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_result(ax1, 'bntest_4r_random0', 'eval', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_result(ax1, 'bntest_4r_random50', 'random', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'sbp_eval', 'Eval', 1, 1, seeds=(1, 2, 3), smooth=True, zorder=6)
# plot_from_log(ax1, 'sbp_train', 'rbp', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'corr_importance', 'corr', 1, 1, seeds=(1, 2, 3), smooth=True, zorder=6)
# plot_from_log(ax1, 'sbp_no_bpcut_train', 'none', 1, 1, seeds=(1, 2, 3), smooth=True)
# plot_from_log(ax1, 'corr_importance_rbp', 'corr_rbp', 1, 1, seeds=(1, 2, 3), smooth=True, zorder=6)
# plot_from_result(ax1, 'baseline', 'nd', 1, 1, seeds=(1, 2, 3), smooth=True)


# plot_from_log(ax1, 'modeldis_b1_e999_w', 'B1_W', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_50_00', 'B1_P_50_00', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_50_25', 'B1_P_50_25', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_all', 'gradweight', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_lastonly', 'gradweight_fc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_lastmap', 'gradweight_fcmap', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'symmatfc', 'symmatfc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_truncpred_bntrain', 'gradweight_fc_bntr', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_truncpred_bneval', 'gradweight_fc_bnev', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_truncpred_bntraincopy', 'gradweight_fc_bntrcp', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_truncpred_bnevalcopy', 'gradweight_fc_bnevcp', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_truncpred_bnevalcopystat', 'gradweight_fc_bnevcpst', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'gradweight_last_bnevalcopystat', 'gradweight_fc_bnevcpst_', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_stdnorm', 'norm_pred', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_statnorm', 'norm_pred', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncpred_statnorm', 'norm_pred_trunc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncpred_statnorm_9', 'norm_pred_trunc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncloss_statnorm', 'norm_loss_trunc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncpred_statnorm', 'norm_pred_trunc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncloss_statnorm_correct_9', 'norm_loss_trunc_1', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_truncpred_statnorm_correct_9', 'norm_pred_trunc_1', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_norm_50_00', 'norm_50_00', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldistill_predbased_norm_50_25', 'norm_50_25', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'modeldis_b1_e999_w_b', 'B10_B_W', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b1_trans', 'B1_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b50_trans', 'B50_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b10_pred_trans', 'B10_pred_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b10_pred_trans', 'B10_pred_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b50_pred_trans', 'B50_pred_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'nodistill', 'NoDistill', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b10_weight_pred', 'B10_weight_pred', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b1_pred_trans', 'B1_pred_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b10_pred_trans_fc', 'B10_pred_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'ours_b20_weight_pred_trans', 'B20_pred_weight_trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'newdistill_10_1_b10_weighted', 'B1', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'newdistill_10_1_b1_weighted_trans', 'B1_Trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'newdistill_10_1_b10_weighted_trans', 'B10_Trans', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'cwd_10_1', 'B1_fc', 1, 1, seeds=(1,), smooth=True)
# plot_from_log(ax1, 'cwd_10_1b', 'B10_fc', 1, 1, seeds=(1,), smooth=True)
# plot_from_result(ax1, 'outdated/feat_distill_10_1', 'FeatureDistill', 1, 1, seeds=(1,), smooth=True)

plt.xlabel('# samples')

# ax.grid()
# ax.set_ylabel('Accuracy')
# ax.set_xlim(0, 50000)
# ax.legend()

ax1.set_ylabel('Accuracy')
ax1.set_xlim(0, 50100)
ax2.set_ylabel('TFLOPs')
ax1.set_ylim(0, 1)
ax1.legend(facecolor=(0.97, 0.97, 0.97), fontsize=8)

ax1.grid()
ax2.grid()
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
# ax2.set_ylabel('LR')
# ax2.set_xlim(0, 50000)
ax2.set_ylim(0, 140)
# ax2.legend()

plt.tight_layout()
plt.savefig('k2_result.pdf')


