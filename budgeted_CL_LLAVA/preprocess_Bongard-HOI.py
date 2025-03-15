import os
import json
import random
from collections import defaultdict

random.seed(42)

def save_dataset(dataset_name, input_folder, output_folder, subset_name, max_num=None):
    # subset_folder = os.path.join(output_folder, subset_name)
    # if not os.path.exists(subset_folder):
    #     os.makedirs(subset_folder)
        
    with open(f"{input_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    
    # Group items by type
    random.shuffle(datalist)
    type_groups = defaultdict(list)
    for item in datalist:
        type_groups[item['type']].append(item)
    
    all_samples = []
    #samples_per_type = max_num // (2 * len(type_groups)) if max_num else None
    
    for type_name, items in type_groups.items():
        positive_samples = []
        negative_samples = []
        
        for idx, item in enumerate(items):
            # if samples_per_type and idx >= samples_per_type:
            #     break
            
            positive_imgfiles = ["dataset/Bongard-HOI/" + path for path in item['image_files'][:7]]
            negative_imgfiles = ["dataset/Bongard-HOI/" + path for path in item['image_files'][7:]]
            
            positive_files = positive_imgfiles[:-1]
            positive_queryfile = positive_imgfiles[-1]
            negative_files = negative_imgfiles[:-1]
            negative_queryfile = negative_imgfiles[-1]
            
            # Randomly select images for positive and negative sets
            selected_positive = random.sample(positive_files, num_per_set)
            selected_negative = random.sample(negative_files, num_per_set)
            
            # Create positive sample
            positive_sample = {
                "id": f"{type_name}-{idx}-positive",
                "object_class": item['object_class'][0],
                "action_class": item['action_class'][0],
                "image": selected_positive + selected_negative + [positive_queryfile],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Positive: {'<image>'*num_per_set}\nNegative: {'<image>'*num_per_set}\nQuery: <image>\n{prompts}\nChoice list:[Positive, Negative]. Your answer is:"
                    },
                    {
                        "from": "gpt",
                        "value": "Positive"
                    }
                ]
            }
            positive_samples.append(positive_sample)
            
            # Randomly reselect images for positive and negative sets
            selected_positive = random.sample(positive_files, num_per_set)
            selected_negative = random.sample(negative_files, num_per_set)
                   
            # Create negative sample
            negative_sample = {
                "id": f"{type_name}-{idx}-negative",
                "object_class": item['object_class'][0],
                "action_class": item['action_class'][0],
                "image": selected_positive + selected_negative + [negative_queryfile],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Positive: {'<image>'*num_per_set}\nNegative: {'<image>'*num_per_set}\nQuery: <image>\n{prompts}\nChoice list:[Positive, Negative]. Your answer is:"
                    },
                    {
                        "from": "gpt",
                        "value": "Negative"
                    }
                ]
            }
            negative_samples.append(negative_sample)
        
        all_samples.extend(positive_samples)
        all_samples.extend(negative_samples)
    
    # Shuffle the samples
    random.shuffle(all_samples)
    
    # Trim to max_num if specified
    if max_num and len(all_samples) > max_num:
        all_samples = all_samples[:max_num]
    
    # Save the combined samples to a single JSON file
    json_output_path = os.path.join(output_folder, f'{subset_name}.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_samples, json_file, indent=4)
    
    print(f"Total samples: {len(all_samples)}")

# User inputs
num_per_sets = [2,3,4]
seeds = [1,2,3,4,5]
types = "ma"

for num_per_set in num_per_sets:
    for seed in seeds:
        input_folder = f'collections/Bongard-HOI/{types}_splits'
        output_folder = f'collections/Bongard-HOI/{types}/{num_per_set*2+1}_set'
        os.makedirs(output_folder, exist_ok=True)
        prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where both "positive" and "negative" images share a "common" object, and only "positive" images share a "common" action whereas "negative" images have different actions compared to the "positive" images, the "common" action is exclusively depicted by the "positive" images. And then given 1 "query" image, please determine whether it belongs to "positive" or "negative".'''

        ### Only for MA ###
        save_dataset('Bongard-HOI', input_folder, output_folder, 'Bongard-HOI_test')
        save_dataset('Bongard-HOI', input_folder, output_folder, f'Bongard-HOI_train_seed{seed}')