import os
import json
import random
from collections import Counter
from tqdm import tqdm

json_path = './train.json'
output_path = './train_generated.json'
samples_path = './images/generated'
with open(json_path, 'r') as f:
    data_dict = json.load(f)
    
result_list = []
for data in tqdm(data_dict):
    generated_dict = {}
    generated_dict['id'] = data['id']
    generated_dict['type'] = data['type']
    generated_dict['image_files'] = []
    generated_dict['object_class'] = data['object_class']
    generated_dict['action_class'] = data['action_class']
    
    # Count each images per action
    action_count_dict = Counter(data['action_class'])
    action_samples_dict = {}
    
    # Randomly sample images
    object_class = data['object_class'][0]
    for action, count in action_count_dict.items():
        sample_path = os.path.join(samples_path, object_class, action)
        samples_list = os.listdir(sample_path)
        samples_list = [os.path.join(sample_path, sample) for sample in samples_list]
        samples = random.sample(samples_list, count)
        action_samples_dict[action] = samples
    
    # Assign each sample to each action
    for action in generated_dict['action_class']:
        generated_dict['image_files'].append(action_samples_dict[action].pop(0))
    
    result_list.append(generated_dict)

with open(output_path, 'w') as f:
    json.dump(result_list, f)