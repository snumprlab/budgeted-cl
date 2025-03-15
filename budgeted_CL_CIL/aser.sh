#/bin/bash

# CIL CONFIG
NOTE="aser_sigma0_cifar10_mem50000_iter32_match" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="aser"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=0
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3"
ASER_TYPE="asvm"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=7.872793029228608 #0.324649306
    N_SMP_CLS="9" K="3"
    CANDIDATE_SIZE=50
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=50000 ONLINE_ITER=0.102945247461735 #0.973947917
    N_SMP_CLS="2" K="3"
    CANDIDATE_SIZE=100
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000 ONLINE_ITER=3
    N_SMP_CLS="2" K="3"
    CANDIDATE_SIZE=200
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167 ONLINE_ITER=0.03125
    MODEL_NAME="resnet18" EVAL_PERIOD=1000
    BATCHSIZE=1024; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=7 nohup python main.py --mode $MODE \
    --dataset $DATASET --n_smp_cls $N_SMP_CLS \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS\
    --rnd_seed $RND_SEED --k $K --aser_type $ASER_TYPE \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --aser_cands $CANDIDATE_SIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP &
done
