import torch
import os
import logging
import transformers
from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
from models.llava.language_model.llava_mpt import LlavaMptForCausalLM
from models.bunny import BunnyPhiForCausalLM, BunnyStableLMForCausalLM, BunnyQwen2ForCausalLM, BunnyMiniCPMForCausalLM, BunnyLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import models.llava.conversation as conversation_lib_llava
import models.bunny.conversation as conversation_lib_bunny
from transformers import Trainer
from peft.tuners.lora import LoraLayer
from models.bunny.prompt_tuning_model import Bunny_PT
from models.llava.prompt_tuning_model import Llava_PT
from models.llava.llama_feddat import LlavaLlamaAdapterForCausalLM
from models.duallora.dualloralayer import DualLoraLayer
from models.feddat_lora.tripleloralayer import TripleLoraLayer
# from models.llava.llava_fedsim import FEDSIMLlavaLlamaForCausalLM
import copy
ACCESS_TOKEN = "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"

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"
    assert model_args.vision_tower is not None
    
    if training_args.mode == 'pfedpg' or training_args.mode == 'fedadapter':
        assert training_args.lora_enable == False, "no lora in pFedPG and feddat  and fedadapter"
    if training_args.mode == 'feddat' or training_args.mode == 'fedadapter':
        assert training_args.gradient_accumulation_steps == 1
    
    # load tokenizer
    # for llava
    if model_args.model_type == "mpt":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif model_args.model_type == 'llama': 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    # for bunny
    elif (
        model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2'
            or model_args.model_type == 'qwen1.5-1.8b' or model_args.model_type == 'minicpm'):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'llama3-8b':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            token=ACCESS_TOKEN
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if model_args.model_type == 'llama3-8b':
        tokenizer.pad_token = tokenizer.eos_token
        
    if training_args.is_eval:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        # prompt tuning
        if training_args.mode == 'pfedpg':
            assert model_args.model_type != 'mpt'
            model = Llava_PT.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                prompt_num=training_args.prompt_num,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load pfedpg')
        # elif training_args.mode == 'feddat' or training_args.mode == 'fedadapter':
        #     assert model_args.model_type != 'mpt'
        #     model = LlavaLlamaAdapterForCausalLM.from_pretrained(
        #         model_args.model_name_or_path,
        #         cache_dir=training_args.cache_dir,
        #         attn_implementation=attn_implementation,
        #         torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        #         **bnb_model_from_pretrained_args
        #     )
        #     print('load feddat')
        # elif training_args.mode == 'fedsim' and training_args.is_eval:
        #     assert model_args.model_type != 'mpt'
        #     model = FEDSIMLlavaLlamaForCausalLM.from_pretrained(
        #         model_args.model_name_or_path,
        #         cache_dir=training_args.cache_dir,
        #         attn_implementation=attn_implementation,
        #         torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        #         **bnb_model_from_pretrained_args
        #     )
        elif 'mpt' == model_args.model_type:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    
    elif 'bunny' in model_args.model_name_or_path.lower():
        # prompt tuning
        if training_args.mode == 'pfedpg':
            assert model_args.model_type == 'phi-2'
            model = Bunny_PT.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
            print('load pfedpg')
        elif model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
            model = BunnyPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'stablelm-2':
            model = BunnyStableLMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'qwen1.5-1.8b':
            model = BunnyQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'minicpm':
            model = BunnyMiniCPMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'llama3-8b':
            model = BunnyLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                token = ACCESS_TOKEN,
                **bnb_model_from_pretrained_args
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")    

    model.config.use_cache = False
    model.model.requires_grad_(False)

    # FIXME
    if training_args.bits >= 16:
        # print(training_args.device)
        model = model.to(training_args.device)
    
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.mode in ['fedsim', 'apfl', 'ditto']:
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        elif training_args.mode in ['feddat']:
            from models.feddat_lora.tripleloramodel import TripleLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['TRIPLELORA'] = TripleLoraModel
            lora_config.peft_type = 'TRIPLELORA'
        
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'llava' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_llava.conv_templates:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
        else:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
            
    elif 'bunny' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_bunny.conv_templates:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates[model_args.version]
        else:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates["default"]

    # load vision tower
    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        # fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    # vision_tower.requires_grad_(True)
    
    # if not training_args.is_eval:
    #     data_args.img_mean = vision_tower.image_processor.image_mean
    #     data_args.img_std = vision_tower.image_processor.image_std
    #     vision_tower.image_processor.do_normalize=False
    # vision_tower.image_processor.do_rescale=False
    data_args.image_processor = vision_tower.image_processor
    
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    # FIXME: freeze mm_projector for feddat or not?
    if training_args.mode == 'pfedpg':
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
    elif training_args.mode == 'feddat':
        if training_args.is_eval:
            for name, module in model.named_modules():
                if isinstance(module, TripleLoraLayer):
                    module.set_state('gate')
                    module.activate_all()
        else:
            for name, module in model.named_modules():
                if isinstance(module, TripleLoraLayer):
                    module.set_state('lora1')
                    module.activate_all()
            model.lm_head.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
    elif training_args.mode == 'fedadapter':
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if 'adapter_1' in n:
                p.requires_grad = True
            elif 'adapter_0' in n or 'adapter_2' in n:
                p.requires_grad = False
        model.deactivate_gating()
        model.set_active_adapter('adapter_1')
    
    elif training_args.mode in [ 'fedsim', 'ditto', 'apfl']:
        model.get_model().global_mm_projector = model.get_model().mm_projector
        model.get_model().local_mm_projector = copy.deepcopy(model.get_model().mm_projector)
        model.get_model().mm_projector = None
        
        if training_args.is_eval:
            for name, module in model.named_modules():
                if isinstance(module, DualLoraLayer):
                    module.set_state('lora2')
        else:
            for name, module in model.named_modules():
                if isinstance(module, DualLoraLayer):
                    module.set_state('lora1')
                    module.activate_all()
            model.lm_head.requires_grad_(False)
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad_(False)
        # for n, p in model.named_parameters():
        #     if 'vision_model' not in n:
        #         p.requires_grad_(True)
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
    
    

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer)or isinstance(module, torch.nn.LayerNorm):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            # if 'norm' in name and 'vision_tower' not in name:
            #     module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    return model, tokenizer, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

from torch import nn

def load_deepspeed(state_dict, module: nn.Module, prefix="", strict=True):
    import deepspeed
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, strict, [], [], [])
            # module.load_state_dict(state_dict, strict=strict)

    for name, child in module._modules.items():
        if child is not None:
            load_deepspeed(state_dict, child, prefix + name + ".")