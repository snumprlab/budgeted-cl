#/bin/bash

# CIL CONFIG
# NOTE="er_ccldc_cifar100_sigma0_mem_8000_0.75_constraint_ensemble"
NOTE="ours_5dataset_mem10000_iter_0.015625"
MODE="ours"

K_COEFF="4"
TEMPERATURE="0.125"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=3
FUTURE_STEPS=4
EVAL_N_WORKER=3
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.5
SEEDS="1 2 3"
DATA_DIR=""


DATASET="5_dataset" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=0.015625
SIGMA=0
REPEAT=1
INIT_CLS=100
USE_AMP="--use_amp"
CHANNEL_CONSTANT=1

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=8000
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet32" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=10000

    # for SparCL
    PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
    SPARSITY_TYPE="irregular"
    MASK_UPDATE_DECAY_EPOCH="5-45"
    SP_MASK_UPDATE_FREQ=500
    REMOVE_N=3000
    RM_EPOCH=20
    GRADIENT=0.80
    LOWER_BOUND="0.75-0.76-0.75"
    UPPER_BOUND="0.74-0.75-0.75"
    SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"
    CONFIG_FILE="./profiles/resnet32/resnet32_0.75.yaml"
    REMARK="irr_0.75_mut"
    LOG_NAME="75_derpp_${GRADIENT}"
    PKL_NAME="irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=8000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet32" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=10000

    # for SparCL
    PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
    SPARSITY_TYPE="irregular"
    MASK_UPDATE_DECAY_EPOCH="5-45"
    SP_MASK_UPDATE_FREQ=500
    REMOVE_N=3000
    RM_EPOCH=20
    GRADIENT=0.80
    LOWER_BOUND="0.75-0.76-0.75"
    UPPER_BOUND="0.74-0.75-0.75"
    SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"
    CONFIG_FILE="./profiles/resnet32/resnet32_0.75.yaml"
    REMARK="irr_0.75_mut"
    LOG_NAME="75_derpp_${GRADIENT}"
    PKL_NAME="irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"


elif [ "$DATASET" == "5_dataset" ]; then
    MEM_SIZE=10000
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet32" VAL_PERIOD=500 EVAL_PERIOD=5000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30000 FEAT_DIM=8 FEAT_MEM_SIZE=3000
    SAMPLES_PER_TASK=10000

    # for SparCL
    PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
    SPARSITY_TYPE="irregular"
    MASK_UPDATE_DECAY_EPOCH="5-45"
    SP_MASK_UPDATE_FREQ=500
    REMOVE_N=3000
    RM_EPOCH=20
    GRADIENT=0.80
    LOWER_BOUND="0.75-0.76-0.75"
    UPPER_BOUND="0.74-0.75-0.75"
    SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"
    CONFIG_FILE="./profiles/resnet32/resnet32_0.75.yaml"
    REMARK="irr_0.75_mut"
    LOG_NAME="75_derpp_${GRADIENT}"
    PKL_NAME="irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"

elif [ "$DATASET" == "clear10" ]; then
    MEM_SIZE=4000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=18000 FEAT_DIM=14 FEAT_MEM_SIZE=96000 #resnet18
    SAMPLES_PER_TASK=3000

elif [ "$DATASET" == "clear100" ]; then
    MEM_SIZE=8000
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=500 
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=60000 FEAT_DIM=14 FEAT_MEM_SIZE=96000 #resnet18
    SAMPLES_PER_TASK=10000

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=2000
    N_SMP_CLS="3" K="3" MIR_CANDS=100
    CANDIDATE_SIZE=200 VAL_SIZE=2
    MODEL_NAME="resnet32" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    # BASEINIT_SAMPLES=60000 FEAT_DIM=4 FEAT_MEM_SIZE=48000
    SAMPLES_PER_TASK=20000

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=10000000
    N_SMP_CLS="3" K="3" MIR_CANDS=500
    CANDIDATE_SIZE=1000 VAL_SIZE=2 DATA_DIR="--data_dir /disk1/jihun/dataset/imagenet"
    MODEL_NAME="resnet18" EVAL_PERIOD=8000 F_PERIOD=200000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10
    BASEINIT_SAMPLES=769000 FEAT_DIM=14 FEAT_MEM_SIZE=9600000000 #60% baseinit
    # BASEINIT_SAMPLES=512500 FEAT_DIM=14 FEAT_MEM_SIZE=960000 #40% baseinit
    SAMPLES_PER_TASK=256233 # Number of Tasks: 5

    # for SparCL
    PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
    SPARSITY_TYPE="irregular"
    MASK_UPDATE_DECAY_EPOCH="5-45"
    SP_MASK_UPDATE_FREQ=500
    REMOVE_N=3000
    RM_EPOCH=20
    GRADIENT=0.80
    LOWER_BOUND="0.75-0.76-0.75"
    UPPER_BOUND="0.74-0.75-0.75"
    SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"
    CONFIG_FILE="./profiles/resnet32/resnet32_0.75.yaml"
    REMARK="irr_0.75_mut"
    LOG_NAME="75_derpp_${GRADIENT}"
    PKL_NAME="irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"
else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=2 nohup python main_new.py --mode $MODE $DATA_DIR \
    --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --samples_per_task $SAMPLES_PER_TASK \
    --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE --channel_constant $CHANNEL_CONSTANT \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --mir_cands $MIR_CANDS \
    --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE \
    --baseinit_samples $BASEINIT_SAMPLES --spatial_feat_dim $FEAT_DIM --feat_memsize $FEAT_MEM_SIZE \
    --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
    --gradient_efficient_mix --gradient_sparse=$GRADIENT --remove-n=$REMOVE_N --keep-lowest-n 0 --remove-data-epoch=$RM_EPOCH --output-dir ${SAVE_FOLDER} --output-name=${PKL_NAME} \
    --kwinner_sparsity 0.3 --pruning_technique CWI --sparsity 0.2 --reset_act_counters --train_budget_1 0.6 --train_budget_2 0.2 --reparameterize --reinit_technique rewind --use_cl_mask --reg_weight 0.2 --stable_model_update_freq 0.5 --rewind_tuning_incl --use_het_drop  &
done
