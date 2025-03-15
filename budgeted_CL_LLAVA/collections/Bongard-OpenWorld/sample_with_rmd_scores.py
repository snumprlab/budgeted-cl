import os, random, shutil
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

# NORMALIZATION, CLIPPING
normalize = True # Fix
clip = True # Fix
lower_percentile = 5.0 # 5.0
# lower_percentile = 2.5 # 5.0
upper_percentile = 95.0 # 95.0
# upper_percentile = 97.5 # 95.0

NUM_POS_IMAGES = 7

equalweight = False
TopK = False
BottomK = False
INVERSE = False
TEMPERATURE = 0.5

base_path = './'
json_path = './RMD_scores/openworld.json'
target_path = './images/generated_RMD'

with open(json_path, 'r') as f:
    RMD_scores = json.load(f)
    
# Parse RMD pickle file to get PATH dict
models = list(RMD_scores.keys())
PATH_dict = {}
for model in models:
    first_uid = next(iter(RMD_scores[model]))
    relative_path = RMD_scores[model][first_uid]['pos'][0]['image_path'] # Get the first item
    PATH_dict[model] = os.path.join(base_path, str(Path(relative_path).parents[2]))

# Get uid list
first_model = next(iter(PATH_dict)); first_path = PATH_dict[first_model]
uids = os.listdir(first_path)

# Shuffle the images
for model in PATH_dict.keys():
    for uid in uids:
        random.shuffle(RMD_scores[model][uid]['pos'])
        random.shuffle(RMD_scores[model][uid]['neg'])

ensembled_images = {uid: {'pos':[], 'neg': []} for uid in uids}
model_class_selected_counter = {model: {uid: {'pos':0, 'neg':0} for uid in uids} for model in PATH_dict.keys()}

for uid in tqdm(uids):
    # For top-k, get list of [(model, image_path, score), ...]
    image_rmd_scores_pos = []; image_rmd_scores_neg = []
    for model, uid_images_dict in RMD_scores.items():
        uid_images_pos = uid_images_dict[uid]['pos']; uid_images_neg = uid_images_dict[uid]['neg']
        for image_dict_pos in uid_images_pos:
            image_path_pos = os.path.join(base_path, image_dict_pos['image_path'])
            image_rmd_scores_pos.append((model, image_path_pos, image_dict_pos['score']))
        for image_dict_neg in uid_images_neg:
            image_path_neg = os.path.join(base_path, image_dict_neg['image_path'])
            image_rmd_scores_neg.append((model, image_path_neg, image_dict_neg['score']))

    # Get sample_path -> (model, score) mapping
    sample_model_RMD_mapping_pos = {}; sample_model_RMD_mapping_neg = {}
    for sample in image_rmd_scores_pos:
        sample_model_RMD_mapping_pos[sample[1]] = sample[0], sample[2] # model, score
    for sample in image_rmd_scores_neg:
        sample_model_RMD_mapping_neg[sample[1]] = sample[0], sample[2]

    if TopK:
        sorted_data_pos = sorted(sample_model_RMD_mapping_pos.items(), key=lambda item: item[1][1], reverse=True)
        sorted_data_neg = sorted(sample_model_RMD_mapping_neg.items(), key=lambda item: item[1][1], reverse=True)
        chosen_samples_pos = [sample[0] for sample in sorted_data_pos[:NUM_POS_IMAGES]]
        chosen_samples_neg = [sample[0] for sample in sorted_data_neg[:NUM_POS_IMAGES]]
    elif BottomK:
        sorted_data_pos = sorted(sample_model_RMD_mapping_pos.items(), key=lambda item: item[1][1], reverse=False)
        sorted_data_neg = sorted(sample_model_RMD_mapping_neg.items(), key=lambda item: item[1][1], reverse=False)
        chosen_samples_pos = [sample[0] for sample in sorted_data_pos[:NUM_POS_IMAGES]]
        chosen_samples_neg = [sample[0] for sample in sorted_data_neg[:NUM_POS_IMAGES]]
    elif equalweight:
        probabilities = [1 / len(PATH_dict)] * len(PATH_dict)
        chosen_samples_pos = []; chosen_samples_neg = []
        while True:
            chosen_model_pos = random.choices(models, weights=probabilities, k=1)[0]
            if len(RMD_scores[chosen_model_pos][uid]['pos']) > 0:
                chosen_image = RMD_scores[chosen_model_pos][uid]['pos'].pop()
                chosen_image_path = os.path.join(base_path, chosen_image['image_path'])
                chosen_samples_pos.append(chosen_image_path)
            if len(chosen_samples_pos) == NUM_POS_IMAGES:
                print(f"Break for uid {uid} with {len(chosen_samples_pos)} images")
                break
        while True:
            chosen_model_neg = random.choices(models, weights=probabilities, k=1)[0]
            if len(RMD_scores[chosen_model_pos][uid]['neg']) > 0:
                chosen_image = RMD_scores[chosen_model_neg][uid]['neg'].pop()
                chosen_image_path = os.path.join(base_path, chosen_image['image_path'])
                chosen_samples_neg.append(chosen_image_path)
            if len(chosen_samples_neg) == NUM_POS_IMAGES:
                print(f"Break for uid {uid} with {len(chosen_samples_neg)} images")
                break
    else:
        # Normalize and clip RMD scores
        scores_pos = np.array([score[1] for score in sample_model_RMD_mapping_pos.values()])
        scores_neg = np.array([score[1] for score in sample_model_RMD_mapping_neg.values()])
        # mean = np.mean(scores); std = np.std(scores)
        
        if clip:
            lower_clip_pos = np.percentile(scores_pos, lower_percentile)
            lower_clip_neg = np.percentile(scores_neg, lower_percentile)
            upper_clip_pos = np.percentile(scores_pos, upper_percentile)
            upper_clip_neg = np.percentile(scores_neg, upper_percentile)
            # print(f"Lower clip: {lower_clip}, Upper clip: {upper_clip}")
            clipped_scores_pos = np.clip(scores_pos, lower_clip_pos, upper_clip_pos)
            clipped_scores_neg = np.clip(scores_neg, lower_clip_neg, upper_clip_neg)
            if normalize:
                mean_pos = np.mean(clipped_scores_pos); std_pos = np.std(clipped_scores_pos)
                mean_neg = np.mean(clipped_scores_neg); std_neg = np.std(clipped_scores_neg)
                result_scores_pos = (clipped_scores_pos - mean_pos) / std_pos
                result_scores_neg = (clipped_scores_neg - mean_neg) / std_neg
            else:
                result_scores_pos = clipped_scores_pos
                result_scores_neg = clipped_scores_neg
        else:
            result_scores_pos = scores_pos
            result_scores_neg = scores_neg
        
        probabilities_pos = softmax_with_temperature(result_scores_pos, TEMPERATURE)
        probabilities_neg = softmax_with_temperature(result_scores_neg, TEMPERATURE)
        if INVERSE:
            # To get the inverse probabilities, first handle the numerical instability
            if np.min(probabilities_pos) < 0:
                probabilities_pos -= np.min(probabilities_pos)
            if np.min(probabilities_neg) < 0:
                probabilities_neg -= np.min(probabilities_neg)
            # Handle devision by zero
            if np.sum(probabilities_pos) == 0:
                raise ValueError("All probabilities (pos) are zero")
            if np.sum(probabilities_neg) == 0:
                raise ValueError("All probabilities (neg) are zero")
            probabilities_pos = 1 / probabilities_pos; probabilities_neg = 1 / probabilities_neg
            probabilities_pos /= np.sum(probabilities_pos); probabilities_neg /= np.sum(probabilities_neg)

        chosen_samples_pos = np.random.choice(list(sample_model_RMD_mapping_pos.keys()), size=NUM_POS_IMAGES, replace=False, p=probabilities_pos)
        chosen_samples_neg = np.random.choice(list(sample_model_RMD_mapping_neg.keys()), size=NUM_POS_IMAGES, replace=False, p=probabilities_neg)
    
    # Update the result dictionary and counter
    for sample_path in chosen_samples_pos:
        ensembled_images[uid]['pos'].append({'model': sample_model_RMD_mapping_pos[sample_path][0], 'image': sample_path})
        model_class_selected_counter[sample_model_RMD_mapping_pos[sample_path][0]][uid]['pos'] += 1

    for sample_path in chosen_samples_neg:
        ensembled_images[uid]['neg'].append({'model': sample_model_RMD_mapping_neg[sample_path][0], 'image': sample_path})
        model_class_selected_counter[sample_model_RMD_mapping_neg[sample_path][0]][uid]['neg'] += 1


# Check the number of images selected for each model, for each class
for model, uid_counter in model_class_selected_counter.items():
    print(f"Model {model} selected for each uid:")
    print(uid_counter)

# Sanity check the number of images for each class
for uid, pos_neg_dict in ensembled_images.items():
    if len(pos_neg_dict['pos']) != NUM_POS_IMAGES:
        raise ValueError(f"POS - uid {uid} has {NUM_POS_IMAGES} images but {len(pos_neg_dict['pos'])} images are selected")
    if len(pos_neg_dict['neg']) != NUM_POS_IMAGES:
        raise ValueError(f"NEG - uid {uid} has {NUM_POS_IMAGES} images but {len(pos_neg_dict['neg'])} images are selected")

# Copy all the images to the target path
# Remove target path if already exists
if os.path.exists(target_path):
    raise OSError(f"Target path already exists! - {target_path}")

for uid, pos_neg_dict in ensembled_images.items():
    image_counter_pos = 0; image_counter_neg = 0
    target_uid_path_pos = os.path.join(target_path, uid, 'pos')
    target_uid_path_neg = os.path.join(target_path, uid, 'neg')
    os.makedirs(target_uid_path_pos, exist_ok=True)
    os.makedirs(target_uid_path_neg, exist_ok=True)
    for pos_img in pos_neg_dict['pos']:
        model = pos_img['model']; image_name = pos_img['image']; prompt = '_'.join(Path(image_name).stem.split('_')[:-1])
        new_image_name = f"{model}_{prompt}_{str(image_counter_pos).zfill(6)}.png"
        image_counter_pos += 1
        shutil.copy(image_name, os.path.join(target_uid_path_pos, new_image_name))
    for neg_img in pos_neg_dict['neg']:
        model = neg_img['model']; image_name = neg_img['image']; prompt = '_'.join(Path(image_name).stem.split('_')[:-1])
        new_image_name = f"{model}_{prompt}_{str(image_counter_neg).zfill(6)}.png"
        image_counter_neg += 1
        shutil.copy(image_name, os.path.join(target_uid_path_neg, new_image_name))
