#/bin/bash
sudo sysctl -w vm.max_map_count=262144

# CIL CONFIG
NOTE="bongard_openworld_sample" # experiment name *****All the models are saved in client_states_$NOTE folder*******
MODE="VLM" # method name
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1
DATASET="Bongard-OpenWorld"

# CL args
FUTURE_STEPS=1
DATA_TYPE="ma" #ma, generaetd, web
NUM_SET=7 # 5 - support set : 4 (2 positive, 2 negative) + 1 query, choice = [5, 7, 9]
MEM_SIZE=500
ONLINE_ITER=1
BATCHSIZE=1
TEMP_BATCHSIZE=0
EVAL_PERIOD=10
EVAL_POINT="600_1200_1800_2400"

LR=5e-5
MM_PROJECTOR_LR=5e-5
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine_with_restarts" #cosine
WARMUP_RATIO=0.03 # SHOULD BE 0.03 / NUM_ROUNDS
MODEL_MAX_LEN=6000

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=8

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

CUDA_VISIBLE_DEVICES=0 nohup python main_new_llava.py \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --model_max_length $MODEL_MAX_LEN \
    --bits $BITS \
    --bf16 True \
    --zero_shot False \
    --future_steps $FUTURE_STEPS \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --memory_size $MEM_SIZE \
    --seed $RND_SEED \
    --dataset $DATASET \
    --num_set $NUM_SET \
    --data_type $DATA_TYPE \
    --eval_point $EVAL_POINT \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --temp_batchsize $TEMP_BATCHSIZE \
    --online_iter $ONLINE_ITER \
    --note $NOTE \
    --eval_period $EVAL_PERIOD \
    --output_dir "./nohup" & #> ./nohup/zs_LLaVA.log 2>&1 &