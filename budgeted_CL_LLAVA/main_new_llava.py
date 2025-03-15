import logging.config
import os
import random
import pickle
import numpy as np
import torch
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, load_deepspeed
from collections import defaultdict

# from utils.method_manager_VLM import select_method
from utils.method_manager_new import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict, Optional, Sequence, List
from utils.train_utils import load_deepspeed
from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig
from models.llava.llava_trainer import LLaVATrainer
from collections import OrderedDict
from deepspeed import zero
import time
import datetime
# import warnings
# warnings.filterwarnings('ignore')

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    ### Load Train & Test datalists ###
    with open(f"collections/{data_args.dataset}/{data_args.data_type}/{data_args.num_set}_set/{data_args.dataset}_train_seed{training_args.seed}.json") as fp:
        train_datalists = json.load(fp)

    with open(f"collections/{data_args.dataset}/ma/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
        test_datalists = json.load(fp)    
    
    print("num_train_samples", len(train_datalists), "num_test_samples", len(test_datalists)) #num_samples[args.dataset]
    
    ### Load Training Eval points ###
    #eval_point = [int(point) for point in training_args.eval_point.split("_")]
    with open(file=f'collections/{data_args.dataset}/ma_splits/{data_args.dataset}_split_record.pkl', mode='rb') as f:
        split_config = pickle.load(f)
    eval_point = split_config[training_args.seed]["train_eval_point"]
    print("eval_point")
    print(eval_point)

    logger.info(f"Select a CIL method ({training_args.mode})")
    method = select_method(training_args, train_datalists, test_datalists, device, model_args=model_args,data_args=data_args, bnb_model_from_pretrained_args=bnb_model_from_pretrained_args)
    eval_results = defaultdict(list)

    samples_cnt = 0
    task_id = 0
    cur_task=0
    
    print("datalists", len(train_datalists))
    for i, data in enumerate(train_datalists):
        # explicit task boundary for twf
        if samples_cnt in [0] + eval_point and training_args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l"]:
            task_id += 1
        
        samples_cnt += 1
        method.online_step(data, samples_cnt)

        if samples_cnt in eval_point or samples_cnt % training_args.eval_period == 0:
            method.online_evaluate(test_datalists, samples_cnt)

    if eval_results["data_cnt"][-1] != samples_cnt:
        method.online_evaluate(test_datalists, samples_cnt)


def load_state_dict(model, model_state_dict_list, training_args):
    model_to_load = local_state_dict_list[client_id]
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(model_to_load, model, strict=False)
        else:
            model.load_state_dict(model_to_load, strict=False)  


def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

if __name__ == "__main__":
    main()