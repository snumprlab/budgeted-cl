import os
import json
import random
from collections import Counter
from tqdm import tqdm

NUM_IMAGES = 14
json_path = './train_ma.jsonl' # FIX THIS!
output_path = './train_generated_RMD.jsonl'
base_path = './images/generated_RMD'

# Process jsonl file
data_list = []
with open(json_path, 'r') as f:
    for line in f:
        json_object = json.loads(line)
        data_list.append(json_object)

result_list = []
for data in data_list:
    data_dict = {}
    data_dict['uid'] = data['uid']
    data_dict['commonSense'] = data['commonSense']
    data_dict['concept'] = data['concept']
    data_dict['caption'] = data['caption']
    data_dict['imageFiles'] = []
    data_dict['urls'] = ['dummy'] * NUM_IMAGES
    
    # Get image paths
    image_dir = os.path.join(base_path, data['uid'])
    pos_dir = os.path.join(image_dir, 'pos'); neg_dir = os.path.join(image_dir, 'neg')
    pos_images = [os.path.join(pos_dir, image) for image in os.listdir(pos_dir)]
    neg_images = [os.path.join(neg_dir, image) for image in os.listdir(neg_dir)]
    
    data_dict['imageFiles'] = pos_images + neg_images
    assert len(data_dict['imageFiles']) == NUM_IMAGES, "Number of images must be 14!"
    
    result_list.append(data_dict)

with open(output_path, 'w') as f:
    for json_object in result_list:
        json_line = json.dumps(json_object)
        f.write(json_line + '\n')
