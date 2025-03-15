#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH -c 6
#SBATCH --job-name=budgeted_base
#SBATCH --mem=30G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=unkillable
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/vpmamba-%j.out
#SBATCH --error=$SCRATCH/vpmamba-%j.err
pyfile=/home/mila/s/sparsha.mishra/scratch/smh/budgeted_CL_LLAVA/main_new_llava_trainer.py
module load anaconda/3
module load cudatoolkit
conda activate /home/mila/s/sparsha.mishra/scratch/budgeted_CL_llava

NOTE="Bongard-OpenWorld_ma_ver3_more_text_num5_iter0.5_infinite_base" #"Bongard-HOI_ma_real_num5_iter0.5_infinite_base" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="VLM"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=4
OURS=""
SAR="" #"--sar"

# fed args
DATASET="Bongard-OpenWorld"
DATA_TYPE="ma_ver3_more_text" #ma, generaetd, web
NUM_SET=5 # 5 - support set : 4 (2 positive, 2 negative) + 1 query, choice = [5, 7, 9]
MODEL_MAX_LEN=10000
NUM_ITER=0.5
BATCHSIZE=2
LR=5e-5
MM_PROJECTOR_LR=0
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="constant" #cosine
WARMUP_RATIO=0.03 # SHOULD BE 0.03 / NUM_ROUNDS

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="./llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="./clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16

elif [ "$MODEL_ARCH" == "bunny_3b" ]; then
    MODEL_NAME="BAAI/Bunny-v1_0-3B"
    VERSION="bunny"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="phi-2"
    BITS=16

elif [ "$MODEL_ARCH" == "bunny_8b" ]; then
    MODEL_NAME="BAAI/Bunny-Llama-3-8B-V"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    BITS=8
else
    echo "Undefined setting"
    exit 1
fi
# --master_port 29500
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    deepspeed --master_port 27006 \
    --include localhost:0 \
    main_new_llava_trainer.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --dataset $DATASET \
    --num_set $NUM_SET \
    --data_type $DATA_TYPE \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --num_iter $NUM_ITER \
    --note $NOTE $OURS $SAR \
    --output_dir "./results/test/" # > ./nohup/fedavg_llava_sc12_lr5e-5_bs16_itr100_constant_nodist.log 2>&1 &

# --eval_period $EVAL_PERIOD
#
