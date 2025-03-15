import logging.config
import os
import random
import re
import string
import pickle
import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel
import pandas as pd
from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num#, can_infer
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava
from models.bunny import conversation as conversation_lib_bunny
from models.duallora.dualloralayer import DualLoraLayer

import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

    
def evaluate(dataset, dataname, task, eval_task, model, tokenizer, device, model_args, training_args, logger):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=2, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
    if 'llava' in model_args.model_name_or_path.lower():
        conv = conversation_lib_llava.default_conversation
    elif 'bunny' in model_args.model_name_or_path.lower():
        conv = conversation_lib_bunny.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    n_word_total = 0
    n_generated_word_total = 1
    n_word_correct = 1
    cnt = 0
    with torch.no_grad():
        # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
        for i, batch in enumerate(tqdm(dataloader)):

            inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
            attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    images=imgs,
                    # image_sizes=image_sizes,
                    do_sample=True,# if args.temperature > 0 else False,
                    temperature=training_args.eval_temp,#args.temperature,
                    top_p=None,#args.top_p,
                    num_beams=1,#args.num_beams,
                    max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria = stopping_criteria
                )
            # if 'bunny' in model_args.model_name_or_path.lower():
            #     input_token_len = inputs.shape[1]
            #     output_ids = output_ids[:,input_token_len:]
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            # breakpoint()
            for pred_sentence, gold, prompt, img_file in zip(pred_sentences, golds, prompts, img_files):
                pred_sentence = pred_sentence.strip()
                print()
                print("gt", gold, "answer", pred_sentence)
                input_label = tokenizer.encode(gold)
                output_id = tokenizer.encode(pred_sentence)
                n_word = len(set(input_label))
                n_generated_word = len(set(output_id))
                n_correct = matching_token_num(output_id, input_label)
                # print(pred_sentence)
                predictions.append({"image_file":img_file, "input":prompt, "sentence":pred_sentence, "gt_sentence":gold.strip()})
                
                n_word_total += n_word
                n_generated_word_total += n_generated_word
                n_word_correct += n_correct
                cnt += 1
                
    logger.info(f"Test | Data {dataname} | curr_task {task} | eval_task {eval_task}| Data {dataname}")
    # logger.info(f"Test | Data {dataname} | curr_task {task} | eval_task {eval_task}| Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
    with open(f"./eval_results/{training_args.mode}/{training_args.note}/seed{training_args.seed}/task{task}_evaltask{eval_task}_{dataname}.json", 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()

    # scores = NLPEvaluator(predictions).evaluate()
    # scores["precision"] = n_word_correct / n_word_total
    # scores["recall"] = n_word_correct / n_generated_word_total
    
    # predictions.append(scores)
    # #save predictions
    # logger.info(f"Test | Data {dataname} | curr_task {task} | eval_task {eval_task}| Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
    # with open(f"./eval_results/{training_args.mode}/{training_args.note}/seed{training_args.seed}/task{task}_evaltask{eval_task}_{dataname}.json", 'w') as fp:
    #     json.dump(predictions, fp, indent=4)
    # torch.cuda.empty_cache()

def evaluate_choices(dataset, dataname, task, eval_task, model, tokenizer, device, model_args, training_args, logger):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=2, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))

    if 'llava' in model_args.model_name_or_path.lower():
        conv = conversation_lib_llava.default_conversation
    elif 'bunny' in model_args.model_name_or_path.lower():
        conv = conversation_lib_bunny.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
        for i, batch in enumerate(tqdm(dataloader)):
            inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
            attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    images=imgs,
                    # image_sizes=image_sizes,
                    do_sample=True,# if args.temperature > 0 else False,
                    temperature=training_args.eval_temp,#args.temperature,
                    top_p=None,#args.top_p,
                    num_beams=1,#args.num_beams,
                    max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria = stopping_criteria
                )
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()

            for pred_sentence, gold, prompt, img_file in zip(pred_sentences, golds, prompts, img_files):
                pred_sentence = pred_sentence.strip()
                choices = parse_choice_list(prompt)
                print()
                print("gt", gold, "answer", pred_sentence)
                pred_option = can_infer(pred_sentence, choices)
            
                if isinstance(pred_option, str):
                    # if gold == pred_option:
                    if gold.lower() == pred_option.lower():
                        correct += 1
                        status='correct'
                    else:
                        status='wrong'
                else:
                    status = 'unkown'
                total += 1
                predictions.append({"image_file":img_file, "input":prompt, "sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status})

    scores = {'accuracy': correct/total}
    
    predictions.append(scores)
    #save predictions
    logger.info(f"Test | Data {dataname} | curr_task {task} | eval_task {eval_task} | accuracy {scores['accuracy']} |")
    with open(f"./eval_results/{training_args.mode}/{training_args.note}/seed{training_args.seed}/task{task}_evaltask{eval_task}_{dataname}.json", 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()

def parse_choice_list(input_string):
    # Try to find the choice list in the format "Choice list:[...]"
    match = re.search(r'Choice list:\[(.*?)\]', input_string)
    if match:
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image ") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        return choices
    
    match = re.search(r'Choice List: \[(.*?)\]', input_string)
    if match:
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image ") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        return choices
    
    # If not found, try to find choices in the format "A. ... B. ... C. ... D. ..."
    match = re.findall(r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)', input_string, re.DOTALL)
    if match:
        return [letter for letter, _ in match]
    
    # If still not found, look for "Image A: ..., Image B: ..., Image C: ..., Image D: ..."
    match = re.findall(r'Image ([A-D]):', input_string)
    if match:
        return [f"Image {letter}" for letter in match]
    
    # If no choices found, return an empty list
    return []

def can_infer(answer, choices):
    answer = str(answer).lower()
    # # Special case for ['Positive', 'Negative']
    # if set(choices) == {'Positive', 'Negative'}:
    #     if 'yes' in answer or 'Yes' in answer:
    #         return 'Positive'
    #     elif 'no' in answer or 'No' in answer:
    #         return 'Negative'

    # First, look for exact matches if choices are not simple letters
    if not all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        possible_answer = []
        for choice in choices:
            if choice.lower() in answer or choice in answer:  # Allow for case-insensitive exact match
                possible_answer.append(choice)
        if len(possible_answer) == 1:
            # print("one", possible_answer[0])
            return possible_answer[0]
    
    # Then, look for simple letter matches (A, B, C, ...)
    letter_matches = re.findall(r'\b[A-Z]\b', answer.upper())
    for letter in letter_matches:
        index = string.ascii_uppercase.index(letter)
        if index < len(choices):
            # print("two", choices[index])
            return choices[index]
    
    # If choices are simple letters, look for those
    if all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        for choice in choices:
            if choice in answer.upper():
                # print("three", choice)
                return choice
            
    # remove underscore and try
    answer =  answer.strip().replace('_', ' ').lower()
    normalized_choices = [choice.replace('_', ' ').lower() for choice in choices]
    if answer in normalized_choices:
        # print("four", choices[normalized_choices.index(answer)])
        return choices[normalized_choices.index(answer)]
    
    # Check for partial matches
    possible_answer = []
    for i, choice in enumerate(normalized_choices):
        if answer in choice or choice in answer:
            possible_answer.append(choices[i])

    if len(possible_answer) == 1:
        # print("five", possible_answer[0])
        return possible_answer[0]
    
    # If no match found, return False
    return False

def main():
    ##################################
    # round_to_eval = 1
    ##################################    
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

    os.makedirs(f"eval_results/{training_args.mode}/{training_args.note}/seed{training_args.seed}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/{training_args.note}/seed{training_args.seed}/round_{training_args.round_to_eval}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    server_eval_key = []
    past_test_datalists = []
    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    if training_args.mode in ["VLM"]:
        test_datalists = get_datalists(data_args, training_args, training_args.seed)
    else:
        if training_args.mode == "VLM_ood":
            test_datalists = get_ood_datalists(data_args, training_args.seed, num_test = 5)
        elif training_args.mode == "VLM_zeroshot":
            test_datalists = get_ood_datalists(data_args, training_args.seed, num_test = 1)
    
    for task_num, test_datalist in enumerate(test_datalists):
        past_test_datalists.append(test_datalist)
        
        if training_args.mode == "VLM_zeroshot":
            logger.info(f'./zeroshot.pth')
            server_state_dict = torch.load(f'./zeroshot.pth', map_location='cpu')
        else:
            logger.info(f'load ./checkpoints_{training_args.note}/seed{training_args.seed}/{training_args.note}_task{task_num+1}.pth')
            server_state_dict = torch.load(f'./checkpoints_{training_args.note}/seed{training_args.seed}/{training_args.note}_task{task_num+1}.pth', map_location='cpu')
        model.load_state_dict(server_state_dict, strict=False)
        
        if training_args.mode == "VLM":
            for eval_task_num, past_test_datalist in enumerate(past_test_datalists):
                dataset = GenerationDataset(past_test_datalist, tokenizer, data_args)
                if "text" in training_args.note and data_args.dataset == "Bongard-OpenWorld":
                    evaluate(dataset, data_args.dataset, task_num + 1, eval_task_num + 1, model, tokenizer, device, model_args, training_args, logger)
                else:
                    evaluate_choices(dataset, data_args.dataset, task_num + 1, eval_task_num + 1, model, tokenizer, device, model_args, training_args, logger)
                server_eval_key.append(data_args.dataset)
        else:
            dataset = GenerationDataset(past_test_datalists[0], tokenizer, data_args)
            if "text" in training_args.note and data_args.dataset == "Bongard-OpenWorld":
                evaluate(dataset, data_args.dataset, task_num + 1, eval_task_num + 1, model, tokenizer, device, model_args, training_args, logger)
            else:
                evaluate_choices(dataset, data_args.dataset, task_num + 1, task_num + 1, model, tokenizer, device, model_args, training_args, logger)
            server_eval_key.append(data_args.dataset)    

def get_datalists(data_args, training_args, seed = 1):

    if data_args.dataset == "Bongard-HOI":        
        with open(f"seen_tasks/Bongard-HOI_seen_tasks.pkl", mode='rb') as f:
            task_splits=pickle.load(f)[seed][data_args.num_set]

    elif data_args.dataset == "Bongard-OpenWorld":    
        with open(f"collections/{data_args.dataset}/ma_splits/{data_args.dataset}_split_record.pkl", mode='rb') as f:
            task_splits=pickle.load(f)[seed]["task_splits"]   
    
    if "text" in training_args.note: 
        if data_args.dataset == "Bongard-OpenWorld":
            # ma_text_ver3_more
            with open(f"collections/{data_args.dataset}/ma_ver3_more_text/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
                whole_test_datalists = json.load(fp)
        else:
            # ma_text_ver3_more
            with open(f"collections/{data_args.dataset}/ma_text/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
                whole_test_datalists = json.load(fp)
    else:
        with open(f"collections/{data_args.dataset}/ma/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
            whole_test_datalists = json.load(fp)
        
    test_df = pd.DataFrame(whole_test_datalists)
    df_keys = list(test_df.keys())
    test_datalists = []
    for task_split in task_splits:
        temp_curr_test_datalists = []
        curr_test_datalists = []
        for curr_task in task_split:
            if data_args.dataset == "Bongard-OpenWorld":
                curr_test_datalist = test_df[test_df["commonSense"] == curr_task].values
            elif data_args.dataset == "Bongard-HOI":
                curr_test_datalist = test_df[(test_df['action_class'] == curr_task[0]) & (test_df['object_class'] == curr_task[1])].values
            temp_curr_test_datalists.extend(curr_test_datalist)

        for test_data in temp_curr_test_datalists:
            test_dict = {}
            for key, element in zip(df_keys, test_data):
                test_dict[key] = element
            curr_test_datalists.append(test_dict)

        test_datalists.append(curr_test_datalists)
            
    return test_datalists

def get_ood_datalists(data_args, seed = 1, num_test=1):

    with open(f"collections/{data_args.dataset}/ma/{data_args.num_set}_set/{data_args.dataset}_test.json") as fp:
        test_datalists = json.load(fp)
    
    return [test_datalists for _ in range(num_test)]

if __name__ == "__main__":
    main()
