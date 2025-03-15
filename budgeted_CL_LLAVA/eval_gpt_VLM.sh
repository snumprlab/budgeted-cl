#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH -c 8
#SBATCH --job-name=budgeted_eval
#SBATCH --mem=60G
#SBATCH --gres=gpu:l40s:1
##SBATCH --partition=main
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/vpmamba-%j.out
#SBATCH --error=$SCRATCH/vpmamba-%j.err
pyfile=/home/mila/s/sparsha.mishra/scratch/smh/budgeted_CL_LLAVA/eval_VLM_CL.py
module load anaconda/3
module load cudatoolkit
conda activate /home/mila/s/sparsha.mishra/scratch/budgeted_CL_llava

RND_SEED=1
NUM_TASKS=5
PREFIX="eval_results/VLM"
NOTE="Bongard-OpenWorld_ma_ver3_more_text_num5_iter0.5_infinite_base"
KEY="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"

python gpt_eval.py \
    --prefix $PREFIX \
    --key $KEY \
    --num_tasks $NUM_TASKS \
    --note $NOTE \
    --seed $RND_SEED \
    


