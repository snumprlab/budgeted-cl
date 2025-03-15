
import torch
import copy
import os
from torch.utils.data import Dataset
import transformers
from configuration.VLM_config import DataArguments
from typing import Dict, Sequence
from PIL import Image
from dataclasses import dataclass
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava import conversation as conversation_lib_llava
# from models.bunny import conversation as conversation_lib_bunny
from packaging import version
from utils.augment import DataAugmentation
# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
IS_TOKENIZER_GREATER_THAN_0_14 = True
Image.MAX_IMAGE_PIXELS = None

class GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        if 'llava' in self.data_args.model_name_for_dataarg.lower():
            conv = conversation_lib_llava.default_conversation.copy()
        # elif 'bunny' in self.data_args.model_name_for_dataarg.lower():
        #     conv = conversation_lib_bunny.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if "image" in source.keys():
            image_file = source["image"]
            if isinstance(image_file, list):
                image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            else:
                image = [Image.open(image_file).convert('RGB')]
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = torch.stack([self.data_args.image_processor.preprocess(expand2square(img, tuple(int(x*255) for x in self.data_args.image_processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
        else: 
            image = torch.zeros(0)
            image_file = []
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        # return input_ids, image, gold, prompt, image_file
        return {
            'input_ids':input_ids,
            'image':image,
            'gold':gold,
            'prompt':prompt,
            'image_file':image_file
        }

    def __len__(self):
        return len(self.datalist)

@dataclass
class DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_file = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_file'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            gold=gold,
            prompt=prompt,
            image_file=image_file
        )
        images = [instance['image'] for instance in instances]
        # if all(x is not None and x.shape == images[0].shape for x in images):
        if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
            images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
            # b, n, c, h, w = images.shape
            # images = images.reshape(b*n,c,h,w)
            # images = self.transform(images).to(dtype=torch.bfloat16)
            batch['images'] = images#.reshape(b,n,c,h,w)
        else:
            # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
            batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        # if preprocess:
        #     self.images = []
        #     for data in self.datalist:
        #         image_file = data['image']
        #         image = Image.open(image_file).convert('RGB')
        #         self.images.append(image)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        sources = self.datalist[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            processor = self.data_args.image_processor
            if self.preprocess:
                image = self.images[i]
            else:
                image_file = self.datalist[i]['image']
                # if ' |sep| ' in image_file:
                if isinstance(image_file, list):
                    image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
                else:
                    image = [Image.open(image_file).convert('RGB')]
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = torch.stack([processor.preprocess(expand2square(img, tuple(int(x*255) for x in processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
                # image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = torch.stack([processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            image=torch.zeros(0)
        if 'llava' in self.data_args.model_name_for_dataarg.lower():
            data_dict = preprocess_text_llava(
                sources,
                self.tokenizer,
                has_image=('image' in self.datalist[i]))
        # elif 'bunny' in self.data_args.model_name_for_dataarg.lower():
        #     data_dict = preprocess_text_bunny(
        #         sources,
        #         self.tokenizer,
        #         has_image=('image' in self.datalist[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        # if 'image' in self.datalist[i]:
        data_dict['image'] = image
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['images'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # if DEFAULT_IMAGE_TOKEN in sentence['value']:
            #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            #     sentence['value'] = sentence['value'].strip()
            if "mmtag" in conversation_lib_llava.default_conversation.version:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


# bunny preprocess

# def preprocess_text_bunny(
#         sources: Sequence[str],
#         tokenizer: transformers.PreTrainedTokenizer,
#         has_image: bool = False
# ) -> Dict:
#     if conversation_lib_bunny.default_conversation.sep_style == conversation_lib_bunny.SeparatorStyle.PLAIN:
#         return preprocess_plain_bunny(sources, tokenizer)

#     if conversation_lib_bunny.default_conversation.version == "bunny":
#         return preprocess_bunny(sources, tokenizer, has_image=has_image)
#     elif conversation_lib_bunny.default_conversation.version in {"minicpm", "llama"}:
#         return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)


# def preprocess_plain_bunny(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         assert len(source) == 2
#         assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#         source[0]['value'] = DEFAULT_IMAGE_TOKEN
#         conversation = source[0]['value'] + source[1]['value'] + conversation_lib_bunny.default_conversation.sep
#         conversations.append(conversation)
#     # tokenize conversations
#     input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
#         target[:tokenized_len] = IGNORE_INDEX

#     return dict(input_ids=input_ids, labels=targets)

# def preprocess_bunny(
#         sources,
#         tokenizer: transformers.PreTrainedTokenizer,
#         has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib_bunny.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack(
#             [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib_bunny.SeparatorStyle.TWO

#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 0
#         end_token_cnt = 0

#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 1

#             round_len += 1
#             end_token_cnt += 1

#             target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if tokenizer.pad_token_id == tokenizer.eos_token_id:
#             cur_len -= end_token_cnt
#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_bunny_with_bos(
#         sources,
#         tokenizer: transformers.PreTrainedTokenizer,
#         has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib_bunny.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack(
#             [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib_bunny.SeparatorStyle.TWO

#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         end_token_cnt = 0
#         target[:cur_len] = IGNORE_INDEX

#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

#             end_token_cnt += 1
#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if tokenizer.pad_token_id == tokenizer.eos_token_id:
#             cur_len -= end_token_cnt
#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# llava preprocess
def preprocess_text_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.PLAIN:
        return preprocess_plain_llava(sources, tokenizer)
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib_llava.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib_llava.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib_llava.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib_llava.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len