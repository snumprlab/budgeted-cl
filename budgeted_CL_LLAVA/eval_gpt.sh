#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH -c 8
#SBATCH --job-name=gpt_score
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
##SBATCH --partition=main
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/vpmamba-%j.out
#SBATCH --error=$SCRATCH/vpmamba-%j.err
pyfile=/home/mila/s/sparsha.mishra/scratch/smh/budgeted_CL_LLAVA/eval_gpt_explainfirst.py
module load anaconda/3
module load cudatoolkit
conda activate /home/mila/s/sparsha.mishra/scratch/budgeted_CL_llava

NOTE='Bongard-OpenWorld_ma_ver3_more_text_num5_iter0.5_infinite_ours'
# Associate each integer value with a list of corresponding dataset names

NUM_TASKS="5"
SEEDS="3"

for SEED in $SEEDS; do
    for curr_task in $(seq 1 $NUM_TASKS); do
        for eval_task in $(seq 1 $curr_task); do
            input_file="./eval_results/VLM/${NOTE}/seed${SEED}/task${curr_task}_evaltask${eval_task}_Bongard-OpenWorld.json"
            output_file="./eval_results/VLM/${NOTE}/seed${SEED}/gpt_task${curr_task}_evaltask${eval_task}_Bongard-OpenWorld.json"

            echo "Processing ${curr_task} ${eval_task}"
            OPENAI_API_KEY="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" python eval_gpt_explainfirst.py -r "$input_file" -o "$output_file"
        done
    done
done

echo "All datasets processed."