import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    ### for TriRE ###
    parser.add_argument("--kwinner", action='store_true')
    parser.add_argument('--het_drop', type=float, default=1, help='heterogenous dropout weight')
    parser.add_argument("--use_het_drop", action='store_true', default=False, help='use heterogenous dropout or not')
    parser.add_argument("--forget_perc", type=float, default=0.9, help="Epoch to where the model is rewinded")
    parser.add_argument('--tri_beta', type=float, default=1, help='Percentage of top-k weights from kwinner neurons/filters')
    parser.add_argument('--sparsity', type=float, default=0.2, help='Percentage of top-k weights from kwinner neurons/filters')
    parser.add_argument('--kwinner_sparsity', type=float, default=0.2, help='Percentage of top-k neurons/ filters to be selected after each task')
    parser.add_argument('--slow_lr_multiplier', type=float, default=1, help='Learning rate multiplier for sparse set in RR (o-ewc)')
    parser.add_argument('--slow_lr_multiplier_decay', action='store_true', help='Decay slow LR multiplier over the course of training')
    parser.add_argument('--train_budget_1', type=float, default=0.6, help='Training budget for first for loop in retain and revise')
    parser.add_argument('--train_budget_2', type=float, default=0.2, help='Training budget for second for loop in retain and revise')
    parser.add_argument('--pruning_technique', type=str, default="magnitude_pruning", help='Pruning technique for sparsifying the network')
    parser.add_argument('--non_overlaping_kwinner', action='store_true', help='During structured pruning stage, do not use any '
                             'overlapping kwinner masks. Be sure that the net has enough capacity.''set sparsity < (1 / num_tasks). ')
    parser.add_argument('--reinit_technique', type=str, default="xavier", help='Technique for re-initializing the part of the network')
    parser.add_argument('--reparameterize', action='store_true', help='Whether to reparameterize the weights after each task')
    parser.add_argument('--mask_non_sparse_weights', action='store_true', help='Whether to mask non sparse weights during forward pass of second loop')
    parser.add_argument('--mask_cum_sparse_weights', action='store_true', help='Whether to mask cumulative sparse weights during forward pass of third loop')
    parser.add_argument('--reservoir_buffer', action='store_true', help='Updates buffer in each iteration. If False, updates at task boundary. ')
    parser.add_argument('--reset_act_counters', action='store_true', help='Reset activation counters after each task')
    parser.add_argument('--reg_weight', type=float, default=0.10, help='EMA regularization weight')
    parser.add_argument('--stable_model_update_freq', type=float, default=0.05, help='EMA update frequency')
    parser.add_argument('--stable_model_alpha', type=float, default=0.999, help='EMA alpha')
    parser.add_argument('--rewind_tuning_incl', action='store_true', help='whether to finetune the observe_3')
    parser.add_argument('--sparse_model_finetuning', action='store_true', help='whether to finetune the sparse model '
                                                                               'exclusively')

    ### for SparCL ###
    parser.add_argument('--buffer_weight', type=float, default=1.0, help="weight of ce loss of buffered samples")
    parser.add_argument('--sample_frequency', type=int, default=50, help="sample frequency for gradient mask")
    parser.add_argument('--sp-retrain', action='store_true', help="Retrain a pruned model")
    parser.add_argument('--channel_constant', type=float, default=1, help="coeff of expanding channel")
    parser.add_argument('--sp-config-file', type=str, help="define config file")
    parser.add_argument('--sp-no-harden', action='store_true', help="Do not harden the pruned matrix")
    parser.add_argument('--sp-admm-sparsity-type', type=str, default='gather_scatter', help="define sp_admm_sparsity_type: [irregular, irregular_global, column,filter]")
    parser.add_argument('--sp-load-frozen-weights', type=str, help='the weights that are frozen throughout the pruning process')
    parser.add_argument('--retrain-mask-pattern', type=str, default='weight', help="retrain mask pattern")
    parser.add_argument('--sp-update-init-method', type=str, default='zero', help="mask update initialization method")
    parser.add_argument('--sp-mask-update-freq', type=int, default=5, help="how many epochs to update sparse mask")
    parser.add_argument('--sp-lmd', type=float, default=0.5, help="importance coefficient lambda")
    parser.add_argument('--retrain-mask-sparsity', type=float, default=-1.0, help="sparsity of a retrain mask, used when retrain-mask-pattern is set to NOT being 'weight' ")
    parser.add_argument('--retrain-mask-seed', type=int, default=None, help="seed to generate a random mask")
    parser.add_argument('--sp-prune-before-retrain', action='store_true', help="Prune the loaded model before retrain, in case of loading a dense model")
    parser.add_argument('--output-compressed-format', action='store_true', help="output compressed format")
    parser.add_argument("--sp-grad-update", action="store_true", help="enable grad update when training in random GaP")
    parser.add_argument("--sp-grad-decay", type=float, default=0.98, help="The decay number for gradient")
    parser.add_argument("--sp-grad-restore-threshold", type=float, default=-1, help="When the decay")
    parser.add_argument("--sp-global-magnitude", action="store_true", help="Use global magnitude to prune models")
    parser.add_argument('--sp-pre-defined-mask-dir', type=str, default=None, help="using another sparse model to init sparse mask")
    parser.add_argument('--upper-bound', type=str, default=None, help="using another sparse model to init sparse mask")
    parser.add_argument('--lower-bound', type=str, default=None, help="using another sparse model to init sparse mask")
    parser.add_argument('--mask-update-decay-epoch', type=str, default=None, help="using another sparse model to init sparse mask")
    parser.add_argument('--gradient_efficient', action='store_true', default=False, help='add gradient efficiency')   
    parser.add_argument('--gradient_efficient_mix', action='store_true', default=False, help='add gradient efficiency (mix method)')     
    parser.add_argument('--gradient_remove', type=float, default=0.1, help="extra removal for gradient efficiency")
    parser.add_argument('--gradient_sparse', type=float, default=0.75, help="total gradient_sparse for training")
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--rho', type=float, default = 0.0001, help ="Just for initialization")
    parser.add_argument('--pretrain-epochs', type=int, default=0, metavar='M', help='number of epochs for pretrain')
    parser.add_argument('--pruning-epochs', type=int, default=0, metavar='M', help='number of epochs for pruning')
    parser.add_argument('--remark', type=str, default=None, help='optimizer used (default: adam)')
    parser.add_argument('--save-model', type=str, default='model/', help='optimizer used (default: adam)')
    parser.add_argument('--sparsity-type', type=str, default='random-pattern', help="define sparsity_type: [irregular,column,filter,pattern]")
    parser.add_argument('--config-file', type=str, default='config_vgg16', help="config file name")
    parser.add_argument('--use_cl_mask', action='store_true', default=False, help='use CL mask or not')
    parser.add_argument('--remove-n', type=int, default=0, help='number of sorted examples to remove from training')
    parser.add_argument('--remove-data-epoch', type=int, default=200, help='the epoch to remove partial training dataset')
    parser.add_argument('--output-name', type=str)
    parser.add_argument('--output-dir', help='directory where to save results')
    parser.add_argument('--keep-lowest-n', type=int, default=0, help='number of sorted examples to keep that have the lowest score, equivalent to start index of removal, if a negative number given, remove random draw of examples')

    # for baseline
    parser.add_argument("--recent_ratio", type=float, default=0.5, help="sampling ratio between recent and past")
    parser.add_argument("--cls_weight_decay", type=float, default=0.999)
    parser.add_argument("--weight_option", type=str, default="loss", help="weightsum softmax loss")
    parser.add_argument("--weight_method", type=str, default="recent_important", help="recent_important / count_important")
    parser.add_argument("--weight_ema_ratio", type=float, default=0.1, help="ema smoothing ratio of loss sum")


    # for twf
    parser.add_argument("--optim_wd", type=float, default=0, help="")
    parser.add_argument("--optim_mom", type=float, default=0, help="")
    parser.add_argument("--pre_epoch", type=int, default=1000, help="pre_train epoch")
    parser.add_argument("--samples_per_task", type=int, default=12500, help="explicit task boundary for twf")
    parser.add_argument("--sigma", type=int, default=10, help="Sigma of gaussian*100")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat period")
    parser.add_argument("--init_cls", type=int, default=100, help="Percentage of classes already present in first period")
    parser.add_argument("--min_resize_threshold", type=int, default=16, help="")
    parser.add_argument("--resize_maps", type=int, choices=[0,1], default=0, help="")

    # 얘네는 default가 없음
    parser.add_argument("--lambda_diverse_loss", type=float, help="")
    parser.add_argument("--lambda_fp_replay", type=float, help="")
    parser.add_argument("--lambda_fp", type=float, help="")
    parser.add_argument("--der_alpha", type=float, help="")
    parser.add_argument("--der_beta", type=float, help="")

    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )
    
    # Train
    parser.add_argument("--klass_train_warmup", type=int, default=30)
    parser.add_argument("--loss_balancing_option", type=str, default="reverse_class_weight", help="reverse_class_weight, class_weight")
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )

    parser.add_argument(
        "--vit_pretrain", action="store_true", help="Use automatic mixed precision."
    )

    parser.add_argument(
        "--f_period", type=int, default=10000, help="Period for measuring forgetting"
    )

    # CL_Loader
    parser.add_argument("--n_worker", type=int, default=4, help="The number of workers")
    parser.add_argument("--future_steps", type=int, default=4, help="The number of future batches loaded.")

    parser.add_argument("--eval_n_worker", type=int, default=4, help="The number of workers for eval.")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="batchsize for eval.")


    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=['randaug'],
        help="Additional train transforms [cutmix, cutout, randaug]",
    )

    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")

    parser.add_argument("--transform_on_gpu", action="store_true", help="Use data augmentation on GPU.")

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--use_kornia", action="store_true", help="enable kornia")
    parser.add_argument("--transform_on_worker", action="store_false", help="transform_on_worker")
    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    # Ours
    parser.add_argument("--count_decay_ratio", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--k_coeff", type=float, default=2)
    parser.add_argument("--corr_warm_up", type=int, default=50)
    parser.add_argument("--unfreeze_rate", type=float, default=0.25)
    parser.add_argument("--freeze_warmup", type=int, default=100)
    parser.add_argument("--target_layer", type=str, default="whole_conv2")
    parser.add_argument("--use_weight", type=str, default="classwise")
    parser.add_argument("--version", type=str, default="ver2")
    parser.add_argument("--klass_warmup", type=int, default=300)
    parser.add_argument("--use_class_balancing", type=bool, default=False)
    parser.add_argument("--use_batch_cutmix", type=bool, default=False)
    parser.add_argument("--use_human_training", type=bool, default=False, help="disable kornia")
    parser.add_argument("--curriculum_option", type=str, default="class_loss", choices=['class_loss', 'class_acc'])

    # ASER
    parser.add_argument('--k', dest='k', default=5,
                        type=int,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')

    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')

    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=2.0,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')
    parser.add_argument('--aser_cands', type=int, default=50, help='# candidates to use for MIR')

    # Ours
    parser.add_argument('--avg_prob', type=float, default=0.2, help='number of GPUs, for GDumb eval')

    # GDumb
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1,
                        help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # RM & GDumb
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")

    # BiC
    parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")
    parser.add_argument("--n_tasks", type=int, default=4, help="use distillation for BiC.")
    parser.add_argument("--val_memory_size", type=int, default=50)

    # AGEM
    parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    parser.add_argument('--beta', type=float, default=10.0, help='distillation strength')
    parser.add_argument('--ema_ratio', type=float, default=0.999, help='ema_ratio')
    parser.add_argument('--ema_ratio_2', type=float, default=0.998, help='ema_ratio_2')
    parser.add_argument('--cls_dim', type=int, default=10, help='Number of output dim reserved for each_class')
    parser.add_argument('--weighted', action="store_true", help='Use class-weighted distillation')
    parser.add_argument('--pred_based', action="store_true", help='Use pred based weighting')
    parser.add_argument('--trans_feature', action="store_true", help='Use train transformed sample for distillation')
    parser.add_argument('--feature_only', action="store_true", help='Use only the features of last layer')
    parser.add_argument('--loss_ema', type=float, default=0.999, help='ema_ratio for updating cls loss')
    parser.add_argument('--norm_loss', type=str, default='none', help='Use normalized cls and distill loss')
    parser.add_argument('--loss_ratio', type=str, default='none', help='Dynamic ratio strategy for cls and distill loss')

    parser.add_argument('--sdp_mean', type=float, default=10000, help='mean of sdp weights, in period')
    parser.add_argument('--sdp_var', type=float, default=0.75, help='variance ratio (var/mean^2) of sdp weights')
    parser.add_argument('--fc_train', type=str, default='none', help='train mode of fc layer')
    parser.add_argument('--online_fc_mode', type=str, default='none', help='train mode of online fc layer')

    parser.add_argument('--reduce_bpdepth', action="store_true", help='Reduce backpropagation depth for distill loss')
    parser.add_argument('--importance', type=str, default='none', help='feature map importance type')
    parser.add_argument('--imp_ema', type=float, default=0.99, help='ema_ratio for updating importance')

    # REMIND
    parser.add_argument('--codebook_size', type=int, default=256, help='size of codebook for REMIND memory')
    parser.add_argument('--n_codebooks', type=int, default=32, help='# of codebooks')
    parser.add_argument('--mixup_alpha', type=float, default=0.1, help='parameter for replay sample mixup')
    parser.add_argument('--baseinit_samples', type=int, default=30000, help='# epochs for baseinit')
    parser.add_argument('--spatial_feat_dim', type=int, default=8, help='dimension of G model output')
    parser.add_argument('--feat_memsize', type=int, default=3000, help='memory size for features')


    args = parser.parse_args()
    return args
