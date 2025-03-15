from dataclasses import dataclass, field
import transformers
from typing import Dict, Optional, Sequence, List, Union

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None) #"liuhaotian/llava-v1.5-7b"
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    max_new_tokens: Optional[int] = field(default=512)
    ours: bool = field(default=False)
    sar: bool = field(default=False)



@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    model_name_for_dataarg: Optional[str] = field(default=None)
    num_set: int = "5"
    data_type: str = "ma"
    dataset: str = field(default="bongard-HOI")

import torch
import os

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    is_eval: bool = False
    round_to_eval: int = None
    eval_temp: float = 0.2
    eval_server: bool = True
    num_iter:float = field(default=100)

    # cl config
    future_steps:int = field(default=4)
    eval_point: str = field(default="100_200_300")
    mode: str = field(default="er")
    # dataset: str = field(default="cifar10")
    scenario: int = field(default=1)
    note: str = field(default=None)
    eval_period: int = field(default=100)
    online_iter: float = field(default=1.0)
    use_kornia: bool = True
    transform_on_gpu: bool = True
    transform_on_worker: bool = False
    topk: int = field(default=1)
    f_period: int = field(default=None)
    transforms: str= field(default='randaug')
    memory_size: int = 500

    # federated learning
    num_clients: int = 5
    num_rounds: int = 20
    iter_per_round: int = 1
    state_dir: str = field(default="./checkpoints")
    final_lr: float = field(default=1e-6)
    mm_final_lr: float = field(default=1e-6)
    # prompt tuning args
    prompt_num: int = field(default=100)
    # dataloader_num_workers
    optim: str = field(default="adamw_torch")
    is_wsd: str = field(default=None)
    decay_ratio: float = field(default=1.0)
    # per_gpu_train_batch_size
    # per_device_eval_batch_size
    # per_device_eval_batch_size
    # learning_rate
    temp_batchsize: int = field(default=2)
    # seed
    # tf32

    cache_dir: Optional[str] = field(default=None)
    
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )


    # lora config
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 2e-5
    group_by_modality_length: bool = field(default=True)