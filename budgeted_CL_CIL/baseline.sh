#/bin/bash

# CIL CONFIG
NOTE="baseline_cifar10_std_check_real" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="baseline"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
HUMAN_TRAINING="True"
USE_AMP="--use_amp"
SEEDS="1"
RECENT_RATIO="0.8"
USE_CLASS_BALANCING="True"
LOSS_BALANCING_OPTION="reverse_class_weight"
WEIGHT_METHOD="count_important"
WEIGHT_OPTION="loss"
T="8"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 ONLINE_ITER=3
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet18" EVAL_PERIOD=100
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
    CUDA_VISIBLE_DEVICES=0 nohup python main.py --mode $MODE --loss_balancing_option $LOSS_BALANCING_OPTION \
    --dataset $DATASET --T $T --use_class_balancing $USE_CLASS_BALANCING \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --weight_method $WEIGHT_METHOD \
    --rnd_seed $RND_SEED --use_human_training $HUMAN_TRAINING --weight_option $WEIGHT_OPTION \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --recent_ratio $RECENT_RATIO \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP &
done
