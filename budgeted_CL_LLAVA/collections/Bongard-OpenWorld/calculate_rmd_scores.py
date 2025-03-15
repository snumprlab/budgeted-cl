import torch
import yaml
import os
import json
import natsort
import random
import numpy as np
import pickle
import argparse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

def calculate_features(image_paths, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(image_path)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 512).to(device))
            continue
        with torch.no_grad():
            image_input = processor(images=image, return_tensors="pt").to(device)
            image_feature = model.get_image_features(**image_input) # [1, 512]
            image_features.append(image_feature)

    # Normalize the features
    image_features = torch.cat(image_features, dim=0) # [N, 512]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # dim=-1 -> torch.Size([N, 1]) -> Normalize each data

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def compute_class_agnostic_params(image_features):
    # mu_agnostic: (512,) / cov_agnostic: (512, 512)
    mu_agnostic = image_features.mean(axis=0)
    cov_agnostic = (image_features - mu_agnostic).T @ (image_features - mu_agnostic)

    cov_agnostic = cov_agnostic / len(image_features)

    return mu_agnostic, cov_agnostic


def compute_class_specific_params(image_features, class_labels):
    # mu_specific: (C, 512) / cov_specific: (C, 512, 512)
    # Image features, class_labels are already aligned
    classes = sorted(list(set(class_labels)))
    mu_specific = {}
    cov_specific = []

    for cls in classes:
        cls_features = image_features[class_labels == cls]
        mu_specific[cls] = cls_features.mean(axis=0)
        cov_specific.append((cls_features - mu_specific[cls]).T @ (cls_features - mu_specific[cls]))

    # cov_specific should be averaged over all classes
    cov_specific = np.array(cov_specific).sum(axis=0) / len(image_features) # (512, 512)

    return mu_specific, cov_specific

def mahalanobis_distance_specific(image_features, mu_specific_dict, sigma_specific, class_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    mu_specific_list = [mu_specific_dict[cls] for cls in class_labels]
    mu_specific_matrix = np.array(mu_specific_list)

    # Calculate the mahalanobis distance
    diff = image_features - mu_specific_matrix
    distance = -1.0 * diff @ sigma_specific_inv @ diff.T

    return distance

def mahalanobis_distance_agnostic(image_features, mu_agnostic, sigma_agnostic):
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    diff = image_features - mu_agnostic
    distance = -1.0 * diff @ sigma_agnostic_inv @ diff.T

    return distance

def mahalanobis_distance_manually(image_features, mu_agnostic, sigma_agnostic, mu_specific_dict, sigma_specific, class_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    # Calculate the mahalanobis distance
    RMD = np.zeros((image_features.shape[0]))
    for i in range(image_features.shape[0]):
        feature = image_features[i]; label = class_labels[i]
        difference_specific = feature - mu_specific_dict[label] # (512,)
        M_specific = -1.0 * difference_specific @ sigma_specific_inv @ difference_specific.T

        difference_agnostic = feature - mu_agnostic
        M_agnostic = -1.0 * difference_agnostic @ sigma_agnostic_inv @ difference_agnostic.T

        RMD[i] = M_specific - M_agnostic

    return RMD

def softmax_with_temperature(z, T): 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

if __name__ == "__main__":
    image_paths = {
        'sdxl': './images/generated_sdxl',
        'floyd': './images/generated_floyd',
        'cogview2': './images/generated_cogview2',
        'sdturbo': './images/generated_sdturbo',
    }
    json_save_path = './RMD_scores/openworld.json'
    first_model = next(iter(image_paths)); first_path = image_paths[first_model]
    uids = os.listdir(first_path)
    
    images_dict = {model: {uid: [] for uid in uids} for model in image_paths.keys()}

    for model, path in image_paths.items():
        for uid in uids:
            uid_path = os.path.join(path, uid)
            uid_pos_path = os.path.join(uid_path, 'pos'); uid_neg_path = os.path.join(uid_path, 'neg')
            pos_images = natsort.natsorted(os.listdir(uid_pos_path))
            neg_images = natsort.natsorted(os.listdir(uid_neg_path))
            pos_images = [os.path.join(uid_pos_path, image) for image in pos_images]
            neg_images = [os.path.join(uid_neg_path, image) for image in neg_images]
            images_dict[model][uid] = {
                'pos': pos_images,
                'neg': neg_images
            }

    # Concat all images into one list and generate corresponding class lables
    concatenated_images_pos = []; concatenated_images_neg = []
    uid_labels_pos = []; uid_labels_neg = []
    model_labels_pos = []; model_labels_neg = []
    for model, path in image_paths.items():
        for uid, pos_neg_dict in images_dict[model].items():
            uid_labels_pos += [uid] * len(pos_neg_dict['pos'])
            uid_labels_neg += [uid] * len(pos_neg_dict['neg'])
            concatenated_images_pos += pos_neg_dict['pos']
            concatenated_images_neg += pos_neg_dict['neg']
            model_labels_pos += [model] * len(pos_neg_dict['pos'])
            model_labels_neg += [model] * len(pos_neg_dict['neg'])
    
    
    # Convert labels to numpy array
    uid_labels_pos = np.array(uid_labels_pos); uid_labels_neg = np.array(uid_labels_neg)
    model_labels_pos = np.array(model_labels_pos); model_labels_neg = np.array(model_labels_neg)

    # Calculate the features
    features_pos = calculate_features(concatenated_images_pos, model)
    features_neg = calculate_features(concatenated_images_neg, model)
    mu_agnostic_pos, cov_agnostic_pos = compute_class_agnostic_params(features_pos)
    mu_agnostic_neg, cov_agnostic_neg = compute_class_agnostic_params(features_neg)
    mu_specific_dict_pos, cov_specific_pos = compute_class_specific_params(features_pos, uid_labels_pos)
    mu_specific_dict_neg, cov_specific_neg = compute_class_specific_params(features_neg, uid_labels_neg)

    # RMD = distance_specific - distance_agnostic
    RMD_pos = mahalanobis_distance_manually(features_pos, mu_agnostic_pos, cov_agnostic_pos, mu_specific_dict_pos, cov_specific_pos, uid_labels_pos)
    RMD_neg = mahalanobis_distance_manually(features_neg, mu_agnostic_neg, cov_agnostic_neg, mu_specific_dict_neg, cov_specific_neg, uid_labels_neg)

    # Split RMD scores using model labels
    RMD_each_model_pos = {}; RMD_each_model_neg = {}
    for model in image_paths.keys():
        RMD_each_model_pos[model] = RMD_pos[model_labels_pos == model]
        RMD_each_model_neg[model] = RMD_neg[model_labels_neg == model]
        
    
    # Generate model - uid - image_path - RMD dictionary
    model_class_image_RMD = {}
    for model in image_paths.keys():
        model_class_image_RMD[model] = {}
        for uid in uids:
            indices_pos = np.where((model_labels_pos == model) & (uid_labels_pos == uid))[0]
            indices_neg = np.where((model_labels_neg == model) & (uid_labels_neg == uid))[0]
            RMDs_pos = RMD_pos[indices_pos]; RMDs_neg = RMD_neg[indices_neg]
            paths_pos = np.array(concatenated_images_pos)[indices_pos]
            paths_neg = np.array(concatenated_images_neg)[indices_neg]
            model_class_image_RMD[model][uid] = {
                'pos': [{"image_path": path, "score": score} for path, score in zip(paths_pos, RMDs_pos)],
                'neg': [{"image_path": path, "score": score} for path, score in zip(paths_neg, RMDs_neg)],
            }

    # Save the RMD scores as json
    with open(json_save_path, 'w') as f:
        json.dump(model_class_image_RMD, f)
    
    breakpoint()

    # Print the top 5 and bottom 5 RMD scores of each model
    concatenated_images = np.array(concatenated_images)
    for model in image_paths.keys():
        top5_indices = RMD_each_model[model].argsort()[-10:]
        bottom5_indices = RMD_each_model[model].argsort()[:10]
        top5_image_paths = concatenated_images[top5_indices]
        bottom5_image_paths = concatenated_images[bottom5_indices]
        print(f"Top 10 RMD for {model}:")
        print(f"{top5_image_paths}, scores: {RMD_each_model[model][top5_indices]}")
        print(f"Bottom 10 RMD for {model}:")
        print(f"{bottom5_image_paths}, scores: {RMD_each_model[model][bottom5_indices]}")


    # Get average RMD scores for each model
    for model in image_paths.keys():
        print(f"Average RMD for {model}: {RMD_each_model[model].mean()}")

    model_RMD_scores = [RMD_each_model[model].mean() for model in image_paths.keys()]
    # Calculate probabilites of numpy array using logit
    temperature_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for temperature in temperature_list:
        probabilities = softmax_with_temperature(model_RMD_scores, temperature)
        # Print {model: probability}
        print(f"Probabilities for temperature {temperature}:")
        for i, model in enumerate(image_paths.keys()):
            print(f"{model}: {probabilities[i]}")
        print()