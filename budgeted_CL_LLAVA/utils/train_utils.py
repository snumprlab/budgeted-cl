import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import pandas as pd
from models import mnist, cifar, imagenet
from torch.utils.data import DataLoader
from onedrivedownloader import download as dn
from torch.optim import SGD
import timm
import copy
import requests
from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from tqdm import tqdm
from torchvision import models as torchvision_models
from torchvision import models
from models.open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from transformers import AutoFeatureExtractor, ResNetModel

### for LLaVA and bunny
import transformers
from transformers.optimization import get_scheduler
from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
from models.llava.language_model.llava_mpt import LlavaMptForCausalLM
# from models.bunny import BunnyPhiForCausalLM, BunnyStableLMForCausalLM, BunnyQwen2ForCausalLM, BunnyMiniCPMForCausalLM, BunnyLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import models.llava.conversation as conversation_lib_llava
# import models.bunny.conversation as conversation_lib_bunny
from transformers import Trainer
from peft.tuners.lora import LoraLayer
# from models.bunny.prompt_tuning_model import Bunny_PT
from models.llava.prompt_tuning_model import Llava_PT
from models.llava.llama_feddat import LlavaLlamaAdapterForCausalLM
from models.duallora.dualloralayer import DualLoraLayer
from models.feddat_lora.tripleloralayer import TripleLoraLayer
# from models.llava.llava_fedsim import FEDSIMLlavaLlamaForCausalLM
import copy
ACCESS_TOKEN = "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"
from transformers import StoppingCriteria, StoppingCriteriaList


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, repeat_len = 2):
      self.n = repeat_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        should_stop =False
        if input_ids.shape[1] > self.n*3:
            last_n_ids = input_ids[0][-self.n:]		# 마지막으로 생성한 n개의 토큰
            lastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            lastlastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            for i in range(self.n):
                if lastlastlast_n_ids[i] != lastlast_n_ids[i] or lastlast_n_ids[i] != last_n_ids[i]: # stop sequence와 비교
                    should_stop = False
                    break
                else :
                    should_stop = True
        return should_stop

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def send_message(url, message):
    payload = {
        'text': message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Message post success")
        else:
            print("Fail to send message:", response.status_code)
    except Exception as e:
        print(e)

# from llava_traininer
def create_LLM_optimizer(model, mm_projector_lr, weight_decay):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """

    opt_model = model
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if mm_projector_lr is not None:
        projector_parameters = [name for name, _ in opt_model.named_parameters() if ("mm_projector" in name)]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                ],
                "weight_decay": weight_decay,
                "lr": mm_projector_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": mm_projector_lr,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    
    if optimizer_cls.__name__ == "Adam8bit":
        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                print(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                print(f"bitsandbytes: will optimize {module} in fp32")
        print(f"skipped: {skipped/2**20}M params")

    return optimizer

class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        #if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )
        x_out = self.transforms(x)
        return x_out


def get_transform(dataset, transform_list, gpu_transform, use_kornia=True):
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset)
    if use_kornia:
        train_transform = DataAugmentation(inp_size, mean, std)
    else:
        train_transform = []
        if "cutout" in transform_list:
            train_transform.append(Cutout(size=16))
            if gpu_transform:
                gpu_transform = False
                print("cutout not supported on GPU!")
        if "randaug" in transform_list:
            train_transform.append(transforms.RandAugment())
            
        if "autoaug" in transform_list:
            if hasattr(transform_list, 'AutoAugment'):
                if 'cifar' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
                elif 'imagenet' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            else:
                train_transform.append(select_autoaugment(dataset))
                gpu_transform = False
        if "trivaug" in transform_list:
            train_transform.append(transforms.TrivialAugmentWide())
        if gpu_transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform

def select_optimizer(opt_name, lr, model, kwargs=None):
    if opt_name in ["MLM_optimizer", "adam"]:
        if hasattr(model, 'fc'):
            fc_name = 'fc'
        elif hasattr(model, 'head'):
            fc_name = 'head'
        if "adam" in opt_name:
            params = [param for name, param in model.named_parameters() if fc_name not in name]
            opt = optim.Adam(params, lr=lr, weight_decay=0)
        elif "sgd" in opt_name:
            params = [param for name, param in model.named_parameters() if fc_name not in name]
            opt = optim.SGD(
                params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
            )
        else:
            raise NotImplementedError("Please select the opt_name [adam, sgd]")
        if 'freeze_fc' not in opt_name:
            opt.add_param_group({'params': getattr(model, fc_name).parameters()})
    else: 
        opt = create_LLM_optimizer(model, kwargs["mm_projector_lr"], kwargs["weight_decay"])
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    elif sched_name == "MLM_scheduler":
        scheduler = get_scheduler(
            kwargs["lr_scheduler_type"],
            optimizer=opt,
            num_warmup_steps=kwargs["num_training_steps"],
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={"num_cycles": num_cycles,}
        )
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)

    return scheduler

def create_scheduler(self, num_training_steps: int, num_cycles:int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    # if self.lr_scheduler is None:

    self._created_lr_scheduler = True
    return self.lr_scheduler

def get_ckpt_remote_url(pre_dataset):
    if pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"

    elif pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"

    elif pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

    else:
        raise ValueError("Unknown auxiliary dataset")


def load_initial_checkpoint(pre_dataset, model, device, load_cp_path = None):
    url, ckpt_name = get_ckpt_remote_url(pre_dataset)
    load_cp_path = load_cp_path if load_cp_path is not None else './checkpoints/'
    print("Downloading checkpoint file...")
    dn(url, load_cp_path)
    print(f"Downloaded in: {load_cp}")
    net = load_cp(load_cp_path, model, device, moco=True)
    print("Loaded!")
    return net

def generate_initial_checkpoint(net, pre_dataset, pre_epochs, num_aux_classes, device, opt_args):
    aux_dset, aux_test_dset = get_aux_dataset()
    net.fc = torch.nn.Linear(net.fc.in_features, num_aux_classes).to(device)
    net.train()
    opt = SGD(net.parameters(), lr=opt_args["lr"], weight_decay=opt_args["optim_wd"], momentum=opt_args["optim_mom"])
    sched = None
    if self.args.pre_dataset.startswith('cub'):
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[80, 150, 250], gamma=0.5)
    elif 'tinyimg' in self.args.pre_dataset.lower():
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[20, 30, 40, 45], gamma=0.5)

    for e in range(pre_epochs):
        for i, (x, y, _) in tqdm(enumerate(aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(aux_dl)):
            y = y.long()
            opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            aux_out = net(x)
            aux_loss = loss(aux_out, y)
            aux_loss.backward()
            opt.step()

        if sched is not None:
            sched.step()
        if e % 5 == 4:
            print(e, f"{self.mini_eval()*100:.2f}%")
    from datetime import datetime
    # savwe the model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelpath = "my_checkpoint" + '_' + now + '.pth'
    torch.save(net.state_dict(), modelpath)
    print(modelpath)

def load_cp(cp_path, net, device, moco=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    print("net")
    print([name for name, _ in net.named_parameters()])
    s = torch.load(cp_path, map_location=device)
    print("s keys", s.keys())
    '''
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
             i in s['state_dict'].items() if 'encoder_q' in k}
    '''

    #if not ignore_classifier: # online CL이므로 fc out-dim을 1부터 시작
    net.fc = torch.nn.Linear(
        net.fc.in_features, 1).to(device) # online이므로 num_aux_classes => 1

    for k in list(s):
        if 'fc' in k:
            s.pop(k)
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
    except:
        _, unm = net.load_state_dict(s, strict=False)
        print("unm")
        print(unm)
        '''
        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"
        '''

    return net
'''
def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(self.device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(self.device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps
'''
def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )

    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"
    assert model_args.vision_tower is not None
    
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
        
        if 'mpt' == model_args.model_type:
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
    
    # elif 'bunny' in model_args.model_name_or_path.lower():
    #     # prompt tuning
        
    #     if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
    #         model = BunnyPhiForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #             bos_token_id=tokenizer.bos_token_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #             **bnb_model_from_pretrained_args
    #         )
    #     elif model_args.model_type == 'stablelm-2':
    #         model = BunnyStableLMForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #             **bnb_model_from_pretrained_args
    #         )
    #     elif model_args.model_type == 'qwen1.5-1.8b':
    #         model = BunnyQwen2ForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #             **bnb_model_from_pretrained_args
    #         )
    #     elif model_args.model_type == 'minicpm':
    #         model = BunnyMiniCPMForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #             **bnb_model_from_pretrained_args
    #         )
    #     elif model_args.model_type == 'llama3-8b':
    #         model = BunnyLlamaForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #             attn_implementation=attn_implementation,
    #             token = ACCESS_TOKEN,
    #             **bnb_model_from_pretrained_args
    #         )
    #     else:
    #         raise ValueError(f"Unknown Model Type {model_args.model_type}")    

    model.config.use_cache = False
    model.model.requires_grad_(False)

    # FIXME
    if training_args.bits >= 16:
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
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'llava' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_llava.conv_templates:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
        else:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
            
    # elif 'bunny' in model_args.model_name_or_path.lower():
    #     if model_args.version in conversation_lib_bunny.conv_templates:
    #         conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates[model_args.version]
    #     else:
    #         conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates["default"]

    # load vision tower
    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        # fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    data_args.image_processor = vision_tower.image_processor
    
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    # FIXME: freeze mm_projector for feddat or not?
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True
    model.lm_head.requires_grad_(False)
    
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
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    return model, tokenizer, data_args

def select_model(model_name, dataset=None, num_classes=None, opt_dict=None, G=False, F=False, ver2=False, args=None):
    if model_name == 'vit':
        if G:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            layers=[model.patch_embed]
            layers.extend(model.blocks[:8])
            # model.cls_token, model.pos_embed, nn.Sequential(*layers)
            return nn.Sequential(*layers)
        elif F:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            layers = model.blocks[8:]
            layers.append(model.head)
            
            return model.norm, nn.Sequential(*layers)
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            return ViTModel(model)
    
    elif model_name == "clip_resnet":
        model, preprocess_train, preprocess_val = create_model_and_transforms('RN50', pretrained="yfcc15m")
        tokenizer = get_tokenizer('RN50')
        criterion = create_loss()
        
        return model, preprocess_train, preprocess_val, tokenizer, criterion
    elif model_name == "clip_vit": 
        model, preprocess_train, preprocess_val = create_model_and_transforms('ViT-B-32', pretrained="openai")
        tokenizer = get_tokenizer('ViT-B-16')
        criterion = create_loss()
        return model, preprocess_train, preprocess_val, tokenizer, criterion

    elif model_name == "resnet50":
        dino_model = torchvision_models.__dict__['resnet50'](num_classes=0)
        dino_model.fc = nn.Linear(2048, num_classes)
        dino_model.cuda()
        dino_model = load_pretrained_weights(dino_model, "dino_resnet50_pretrain.pth")
        return dino_model
    
    model_imagenet = False
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )
    
    #model_class = getattr(imagenet, "ResNet")
    model_imagenet=True
    model_class = getattr(cifar, "ResNet")
    if G:
        model_class = getattr(cifar, "ResNet_G")
        opt["ver2"] = ver2
    elif F:
        model_class = getattr(cifar, "ResNet_F")
        opt["ver2"] = ver2
    
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt, model_imagenet)
    return model

##### for ASER #####
def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


#### For x_der ###
def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)


def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


class strong_aug():
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.ToTensor()
        ])
        self.mean = mean
        self.std = std

    def __call__(self, x):
        flip = random_flip(x)
        tmp = torch.stack(
                [self.transform(a) for a in flip]
            )
        tmp2 = random_grayscale(
            tmp)
        y = normalize(tmp2, self.mean, self.std)
        return y


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean(0)

        return loss.mean() if self.reduction == 'mean' else loss.sum()
    
    
class ViTModel(nn.Module):
    def __init__(self, vit_model, layer_name=None, G=False):
        super(ViTModel, self).__init__()
        self.vit_model = vit_model
        self.layer_name = layer_name
        # self.intermediate_output = None
        # self.hook_handle = self.vit_model.norm.register_forward_hook(self.hook_fn)
        self.head = copy.deepcopy(self.vit_model.head)
        self.vit_model.head = nn.Identity()
        self.G = G

    def hook_fn(self, module, input, output):
        # print("module name", module._get_name())
        self.intermediate_output = output

    def forward(self, x, get_feature=False):
        features = self.vit_model(x) 
        output = self.head(features)
        if get_feature:
            return output, features
        return output
    
class ResNet50(nn.Module):
    def __init__(self, resnet_model, num_classes=None, layer_name=None, G=False):
        super(ResNet50, self).__init__()
        self.encoder = resnet_model
        self.fc = nn.Linear(100352, num_classes)

    def forward(self, x, get_feature=False):
        # inputs = self.feature_extractor(x)
        features = self.encoder(x) 
        features = features.last_hidden_state
        features = features.view(x.size(0), -1)
        output = self.fc(features)
        if get_feature:
            return output, features
        return output
    
def load_pretrained_weights(model, pretrained_weights):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return model


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