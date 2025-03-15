#/bin/bash

# CIL CONFIG
NOTE="ver10_flops_check" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ours"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
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
VERSION="ver10"
INTERVAL=5
THRESHOLD="5e-2"
UNFREEZE_THRESHOLD="1e-1"
THRESHOLD_COEFF=0.1
THRESHOLD_POLICY="block"
UNFREEZE_COEFF=100
FREEZE_WARMUP=1000
MAX_P="1.0"
MIN_P="0.1"
TARGET_LAYER="last_conv2" # whole_conv2, last_conv2

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1.2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000 ONLINE_ITER=0.75
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet18" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=4 nohup python main.py --mode $MODE --loss_balancing_option $LOSS_BALANCING_OPTION \
    --dataset $DATASET --use_weight $USE_WEIGHT --klass_train_warmup $KLASS_TRAIN_WARMUP --freeze_warmup $FREEZE_WARMUP\
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --weight_method $WEIGHT_METHOD --target_layer $TARGET_LAYER \
    --rnd_seed $RND_SEED --weight_option $WEIGHT_OPTION --klass_warmup $KLASS_WARMUP --threshold $THRESHOLD --max_p $MAX_P --min_p $MIN_P \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME --version $VERSION --unfreeze_threshold $UNFREEZE_THRESHOLD \
    --lr $LR --batchsize $BATCHSIZE --recent_ratio $RECENT_RATIO --avg_prob $AVG_PROB --interval $INTERVAL --threshold_coeff $THRESHOLD_COEFF --unfreeze_coeff $UNFREEZE_COEFF \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --curriculum_option $CURRICULUM_OPTION --threshold_policy $THRESHOLD_POLICY \
    --note $NOTE --val_period $VAL_PERIOD --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP &
done
