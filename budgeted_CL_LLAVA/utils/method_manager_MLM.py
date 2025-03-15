from typing import Callable, Tuple, Type, Dict
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
import torch

def sft_load_state_dict(model, local_state_dict_list, client_id, training_args):
    model_to_load = local_state_dict_list[client_id]
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(model_to_load, model, strict=False)
        else:
            model.load_state_dict(model_to_load, strict=False)  

def create_trainer(model, tokenizer, training_args, data_module):
    trainer = LLaVATrainer(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        )
    return trainer

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    if mode == 'VLM':
        load_state_dict, create_trainer = sft_load_state_dict, create_trainer
    else:
        raise NotImplementedError(mode)
    return load_state_dict, create_trainer