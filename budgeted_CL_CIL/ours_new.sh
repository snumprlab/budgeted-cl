#/bin/bash

# CIL CONFIG
NOTE="aser_sigma10_tiny_iter0.75" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="aser"
DATASET="tinyimagenet" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
HUMAN_TRAINING="False"
USE_AMP="--use_amp"
SEEDS="1"
AVG_PROB="0.4"
RECENT_RATIO="0.8"
LOSS_BALANCING_OPTION="reverse_class_weight" #none
WEIGHT_METHOD="count_important"
WEIGHT_OPTION="loss"
USE_WEIGHT="similarity"
KLASS_WARMUP="300"
KLASS_TRAIN_WARMUP="50"
CURRICULUM_OPTION="class_acc"
VERSION="ver8"
INTERVAL=5
UNFREEZE_COEFF=100
FREEZE_WARMUP=1000
MAX_P="1.0"
MIN_P="0.1"
TARGET_LAYER="last_conv2" # whole_conv2, last_conv2
COUNT_DECAY_RATIO=0.9
CORR_WARM_UP=50
TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=2
FUTURE_STEPS=2
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=512
K_COEFF=4
TEMPERATURE=1


if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    N_SMP_CLS="9" K="3"
    CANDIDATE_SIZE=50
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 ONLINE_ITER=0.75
    N_SMP_CLS="2" K="3"
    CANDIDATE_SIZE=100
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000 ONLINE_ITER=0.75
    N_SMP_CLS="3" K="3"
    CANDIDATE_SIZE=200
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167 ONLINE_ITER=0.0625
    N_SMP_CLS="3" K="3"
    CANDIDATE_SIZE=1000
    MODEL_NAME="resnet18" EVAL_PERIOD=8000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=2 nohup python main_new.py --mode $MODE --loss_balancing_option $LOSS_BALANCING_OPTION \
    --dataset $DATASET --use_weight $USE_WEIGHT --klass_train_warmup $KLASS_TRAIN_WARMUP --freeze_warmup $FREEZE_WARMUP\
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --weight_method $WEIGHT_METHOD --target_layer $TARGET_LAYER \
    --rnd_seed $RND_SEED --weight_option $WEIGHT_OPTION --klass_warmup $KLASS_WARMUP --temperature $TEMPERATURE --n_smp_cls $N_SMP_CLS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME --version $VERSION --transform_on_gpu \
    --lr $LR --batchsize $BATCHSIZE --recent_ratio $RECENT_RATIO --avg_prob $AVG_PROB --corr_warm_up $CORR_WARM_UP --k_coeff $K_COEFF \
    --memory_size $MEM_SIZE --online_iter $ONLINE_ITER --curriculum_option $CURRICULUM_OPTION --count_decay_ratio $COUNT_DECAY_RATIO --use_kornia \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --aser_cands $CANDIDATE_SIZE \
    --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE &
done
