#/bin/bash

# CIL CONFIG
NOTE="bic_cifar10_iter1_check" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="bic"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3"
N_TASKS="5"
SAMPLES_PER_TASK="10000"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 VAL_SIZE=50 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 VAL_SIZE=200 ONLINE_ITER=3
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000 VAL_SIZE=400 ONLINE_ITER=3
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167 VAL_SIZE=2000 ONLINE_ITER=0.25
    MODEL_NAME="resnet18" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=7 nohup python main.py --mode $MODE  \
    --dataset $DATASET --samples_per_task $SAMPLES_PER_TASK  \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS  \
    --rnd_seed $RND_SEED --n_tasks $N_TASKS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --val_memory_size $VAL_SIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP &
done
