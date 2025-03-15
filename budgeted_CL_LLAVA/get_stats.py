# How to add new dataset config
# 1. get_stats.py -> automatically append mean, std to json file (for augmentation)
# 2. upload dataset to gdrive & modify dataset sh file (PACS_final_grive.sh...)
# 3. Create json (make_collections.py) & move all jsons files to collections/ dir
# 4. modify ex.sh

import os
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

def get_stat(image_root_dir):
    # Calculate mean and std for each channel

    # Get all images recursively
    images = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith(tuple(image_exts)):
                images.append(os.path.join(root, file))

    print(f"Found {len(images)} images")
    # Calculate mean and std
    means, stds = [], []

    # Read all images and calculate mean and std
    for image in tqdm(images):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        means.append(np.mean(img / 256, axis=(0, 1)))
        stds.append(np.std(img / 256, axis=(0, 1)))

    # Calculate mean and std for all images
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return {'mean': mean, 'std': std}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mean and std for images')
    parser.add_argument('-r', '--image_root_dir', type=str, help='Root directory of images')
    args = parser.parse_args()
    result = get_stat(args.image_root_dir)

    mean = (result['mean'][0], result['mean'][1], result['mean'][2])
    std = (result['std'][0], result['std'][1], result['std'][2])
    print(f"Mean: ({result['mean'][0]:.8f}, {result['mean'][1]:.8f}, {result['mean'][2]:.8f})")
    print(f"Std: ({result['std'][0]:.8f}, {result['std'][1]:.8f}, {result['std'][2]:.8f})")

    # Write result stats to json file
    json_path = './utils/data_statistics.json'
    with open(json_path, 'r') as f:
        data_statistics = json.load(f)
    
    last_part = os.path.basename(os.path.normpath(args.image_root_dir))
    
    dataset_list = ["PACS_final", "DomainNet", "cct", "NICO", "cifar10"]
    dataset_name = next((name for name in dataset_list if last_part.startswith(name)), None)
    type_name = last_part[len(dataset_name) + 1:]

    if dataset_name not in data_statistics['mean']:
        data_statistics['mean'][dataset_name] = {}
        data_statistics['std'][dataset_name] = {}
    
    data_statistics['mean'][dataset_name][type_name] = mean
    data_statistics['std'][dataset_name][type_name] = std

    print(f"Dataset name: {dataset_name}, type name: {type_name}")
    with open(json_path, 'w') as f:
        json.dump(data_statistics, f)