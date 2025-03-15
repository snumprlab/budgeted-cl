#/bin/bash

# CIL CONFIG
NOTE="gdumb_1r_50s" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="gdumb"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=50
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3"
NUM_GPUS=8
WORKERS_PER_GPU=4


if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100 F_PERIOD=10000
    BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos" MEMORY_EPOCH=255

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100 F_PERIOD=10000
    BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos" MEMORY_EPOCH=255

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100 F_PERIOD=20000
    BATCHSIZE=32; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos" MEMORY_EPOCH=255

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=1000 F_PERIOD=100000
    BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep" MEMORY_EPOCH=100

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS\
    --rnd_seed $RND_SEED --f_period $F_PERIOD\
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --num_gpus $NUM_GPUS --workers_per_gpu $WORKERS_PER_GPU --memory_epoch $MEMORY_EPOCH \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP
done
