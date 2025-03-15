import logging.config
import random
import os
import json
from typing import List
import copy
import PIL
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry.transform import resize
from utils.augmentations import CustomRandomCrop, CustomRandomHorizontalFlip, DoubleCompose, DoubleTransform

logger = logging.getLogger()

def get_custom_double_transform(transform):
    tfs = []
    for tf in transform:
        if isinstance(tf, transforms.RandomCrop):
            tfs.append(CustomRandomCrop(tf.size, tf.padding, resize=self.args.resize_maps==1, min_resize_index=2))
        elif isinstance(tf, transforms.RandomHorizontalFlip):
            tfs.append(CustomRandomHorizontalFlip(tf.p))
        elif isinstance(tf, transforms.Compose):
            tfs.append(DoubleCompose(
                get_custom_double_transform(tf.transforms)))
        else:
            tfs.append(DoubleTransform(tf))

def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, device, teacher_forcing: list = None, extern_attention_maps: list = None):

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
                len(net_feat,)).bool().to(device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_tmp = np.array(x)  # HxWxC
        x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = resize(x_out.float() / 255.0, (self.input_size, self.input_size))
        return x_out

class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None, cls_list=None, data_dir=None,
                 preload=False, device=None, transform_on_gpu=False, use_kornia=False):
        self.use_kornia = use_kornia
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.preload = preload
        self.device = device
        self.transform_on_gpu = transform_on_gpu
        if self.preload:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)
            self.preprocess = Preprocess(input_size=inp_size)
            if self.transform_on_gpu:
                self.transform_cpu = transforms.Compose(
                    [
                        transforms.Resize((inp_size, inp_size)),
                        transforms.PILToTensor()
                    ])
                self.transform_gpu = self.transform
            self.loaded_images = []
            for idx in range(len(self.data_frame)):
                sample = dict()
                try:
                    img_name = self.data_frame.iloc[idx]["file_name"]
                except KeyError:
                    img_name = self.data_frame.iloc[idx]["filepath"]
                if self.cls_list is None:
                    label = self.data_frame.iloc[idx].get("label", -1)
                else:
                    label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])
                if self.data_dir is None:
                    img_path = os.path.join("dataset", self.dataset, img_name)
                else:
                    img_path = os.path.join(self.data_dir, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                if self.use_kornia:
                    image = self.preprocess(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform_on_gpu:
                    image = self.transform_cpu(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform:
                    image = self.transform(image)
                sample["image"] = image
                sample["label"] = label
                sample["image_name"] = img_name
                self.loaded_images.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.preload:
            return self.loaded_images[idx]
        else:
            sample = dict()
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_name = self.data_frame.iloc[idx]["file_name"]
            if self.cls_list is None:
                label = self.data_frame.iloc[idx].get("label", -1)
            else:
                label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])

            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            sample["image"] = image
            sample["label"] = label
            sample["image_name"] = img_name
            return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]

    def generate_idx(self, batch_size):
        if self.preload:
            arr = np.arange(len(self.loaded_images))
        else:
            arr = np.arange(len(self.data_frame))
        np.random.shuffle(arr)
        if batch_size >= len(arr):
            return [arr]
        else:
            return np.split(arr, np.arange(batch_size, len(arr), batch_size))

    def get_data_gpu(self, indices):
        images = []
        labels = []
        data = {}
        if self.use_kornia:
            images = [self.loaded_images[i]["image"] for i in indices]
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
            data["image"] = images

            for i in indices:
            # labels
                labels.append(self.loaded_images[i]["label"])
        else:
            for i in indices:
                if self.preload:
                    if self.transform_on_gpu:
                        images.append(self.transform_gpu(self.loaded_images[i]["image"].to(self.device)))
                    else:
                        images.append(self.transform(self.loaded_images[i]["image"]).to(self.device))
                    labels.append(self.loaded_images[i]["label"])
                else:
                    try:
                        img_name = self.data_frame.iloc[i]["file_name"]
                    except KeyError:
                        img_name = self.data_frame.iloc[i]["filepath"]
                    if self.cls_list is None:
                        label = self.data_frame.iloc[i].get("label", -1)
                    else:
                        label = self.cls_list.index(self.data_frame.iloc[i]["klass"])
                    if self.data_dir is None:
                        img_path = os.path.join("dataset", self.dataset, img_name)
                    else:
                        img_path = os.path.join(self.data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    image = self.transform(image)
                    images.append(image.to(self.device))
                    labels.append(label)
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels).to(self.device)
        return data


class StreamDataset(Dataset):
    def __init__(self, datalist, dataset, transform, cls_list, data_dir=None, device=None, transform_on_gpu=False, use_kornia=True):
        self.use_kornia = use_kornia
        self.images = []
        self.labels = []
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.device = device

        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        self.preprocess = Preprocess(input_size=inp_size)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.PILToTensor()
                ])
            self.transform_gpu = transform
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            if self.use_kornia:
                self.images.append(self.preprocess(PIL.Image.open(img_path).convert('RGB')))
            elif self.transform_on_gpu:
                self.images.append(self.transform_cpu(PIL.Image.open(img_path).convert('RGB')))
            else:
                self.images.append(PIL.Image.open(img_path).convert('RGB'))
            self.labels.append(self.cls_list.index(data['klass']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    @torch.no_grad()
    def get_data(self):
        data = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            images = torch.stack(self.images).to(self.device)
            data['image'] = self.transform_gpu(images)

        if not self.use_kornia:
            for i, image in enumerate(self.images):
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(image.to(self.device)))
                else:
                    images.append(self.transform(image))
            data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(self.labels)
        return data



class MemoryDataset(Dataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_kornia=True, buf_transform=None, cls_weight_decay=None, weight_option=None, weight_ema_ratio=None, use_human_training=False):
        self.use_human_training = use_human_training
        self.use_kornia = use_kornia
        self.datalist = []
        self.weight_ema_ratio = weight_ema_ratio
        self.losses = []
        self.labels = []
        self.cls_weight_decay = cls_weight_decay
        self.weight_option = weight_option
        self.images = []
        self.cls_loss = []
        self.cls_times = []
        self.stream_images = []
        self.logits = []
        self.attention_maps = []
        self.cls_weight = []
        self.stream_labels = []
        self.dataset = dataset
        self.transform = transform
        self.counts = []
        self.class_usage_cnt = []
        self.tasks = []
        self.cls_list = []
        self.cls_used_times = []
        self.cls_dict = {cls_list[i]:i for i in range(len(cls_list))}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.keep_history = keep_history
        self.usage_cnt = []
        self.sample_weight = []
        #self.buf_transform = get_custom_double_transform(buf_transform)
        self.transform_on_gpu = transform_on_gpu
        
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=self.dataset)

        self.preprocess = Preprocess(input_size=inp_size)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.PILToTensor()
            ])
            self.transform_gpu = transform
            self.test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std)])
        self.save_test = save_test
        if self.save_test is not None:
            self.device_img = []


    def __len__(self):
        return len(self.images)

    def add_new_class(self, cls_list, sample=None):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_loss.append(None)
        if sample is not None:
            self.cls_times.append(sample['time'])
        else:
            self.cls_times.append(None)
        #self.cls_used_times.append(max(self.cls_times))
        self.cls_weight.append(1)
        self.cls_idx.append([])
        self.class_usage_cnt.append(0)
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.value()
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    def register_stream(self, datalist):
        self.stream_images = []
        self.stream_labels = []
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            if self.use_kornia:
                self.stream_images.append(self.preprocess(PIL.Image.open(img_path).convert('RGB')))
            elif self.transform_on_gpu:
                self.stream_images.append(self.transform_cpu(PIL.Image.open(img_path).convert('RGB')))
            else:
                self.stream_images.append(PIL.Image.open(img_path).convert('RGB'))
            self.stream_labels.append(self.cls_list.index(data['klass']))

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def batch_iterate(size: int, batch_size: int):
        n_chunks = size // batch_size
        if size % batch_size != 0:
            n_chunks += 1

        for i in range(n_chunks):
            yield list(range(i * batch_size, min((i + 1) * batch_size), size))

    def loop_over_buffer(model, pretrain_model, batch_size, task_id):
        with torch.no_grad():
            for buf_idxs in self.batch_iterate(len(self.images), batch_size):

                buf_labels = torch.Tensor(self.labels[buf_idxs]).to(self.device)
                buf_mask = torch.Tensor(self.tasks[buf_idxs]) == task_id

                if not buf_mask.any():
                    continue

                buf_inputs = self.device_imgs[buf_idxs][buf_mask]
                buf_labels = buf_labels[buf_mask]
                buf_inputs = torch.stack([ee for ee in buf_inputs]).to(self.device)

                _, buf_partial_features = model(buf_inputs, get_features=True)
                prenet_input = buf_inputs
                _, pret_buf_partial_features = pretrain_model(prenet_input, get_features=True)


                #buf_partial_features = buf_partial_features[:-1]
                #pret_buf_partial_features = pret_buf_partial_features[:-1]

                _, attention_masks = partial_distill_loss(model, buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels, self.device)

                for idx in buf_idxs:
                    self.attention_maps[idx] = [
                        at[idx % len(at)] for at in attention_masks]

    def time_update(self, label, time):
        if self.cls_times[label] is not None:
            self.cls_times[label] = self.cls_times[label] * (1-self.weight_ema_ratio) + time * self.weight_ema_ratio
        else:
            self.cls_times[label] = time
        

    def replace_sample(self, sample, idx=None, logit=None, attention_map=None, count=None, task=None, mode=None, online_iter=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.datalist.append(sample)
            
            # ER이면 Iteration만큼 이미 usage_cnt가 update 되어 있어야함
            if mode == 'er':
                if online_iter is not None:
                    self.usage_cnt.append(online_iter)
                else:
                    self.usage_cnt.append(0)
            else:
                self.usage_cnt.append(0)
                
            self.sample_weight.append(1)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if self.use_kornia:
                img = self.preprocess(img)
            elif self.transform_on_gpu:
                img = self.transform_cpu(img)
            self.images.append(img)
            self.labels.append(self.cls_dict[sample['klass']])
            self.losses.append(0.1)
             
            # for recent
            # self.time_update(self.cls_dict[sample['klass']], sample['time'])

            if count is not None:
                self.counts.append(count)

            # for twf
            if logit is not None:
                self.logits.append(logit)
                self.attention_maps.append(attention_map)
                self.tasks.append(task)

            if self.save_test == 'gpu':
                self.device_img.append(self.test_transform(img).to(self.device).unsqueeze(0))
            elif self.save_test == 'cpu':
                self.device_img.append(self.test_transform(img).unsqueeze(0))
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
                else:
                    self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))

        else:
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = sample
            self.usage_cnt[idx] = 0
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if self.use_kornia:
                img = self.preprocess(img)
            elif self.transform_on_gpu:
                img = self.transform_cpu(img)

            # for recent
            # self.time_update(self.cls_dict[sample['klass']], sample['time'])
                
            self.images[idx] = img
            self.labels[idx] = self.cls_list.index(sample['klass'])
            self.sample_weight[idx] = 1
            self.losses[idx] = 0.1
            
            # for twf
            if logit is not None:
                self.logits[idx] = logit
                self.attention_map[idx] = attention_map
                self.tasks[idx] = task

            if count is not None:
                self.counts[idx] = count

            if self.save_test == 'gpu':
                self.device_img[idx] = self.test_transform(img).to(self.device).unsqueeze(0)
            elif self.save_test == 'cpu':
                self.device_img[idx] = self.test_transform(img).unsqueeze(0)
            if self.keep_history:
                if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
                else:
                    self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])

    '''
    def classwise_get_weight(self, weight_method="recent_important", batch_size):
        weight = np.zeros(len(self.images))
        weight = weight.astype('float64')
        
        batch = np.zeros(batch_size)
        
        cls_weight = []
        
        if weight_method == "recent_important":
            print("self.cls_times", self.cls_times)
            
            # method 1) times 기반 class에 weight 주기
            max_time = max(self.cls_times) + 0.2
            for klass, klass_time in enumerate(self.cls_times):
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = (klass_time+0.2) / max_time #np.exp(-1*(klass_count/total_count))
                cls_weight.append(np.exp((klass_time+0.2) / max_time))

            cls_weight = F.softmax(torch.DoubleTensor(cls_weight), dim=0)
            selected_place = random.choices(range(len(self.cls_times)), k=batch_size, weights=cls_weight) # klass가 몇개 select되는지 자리 배정
            
            for place in list(set(selected_place)):
                place_index = np.where(place == selected_place)[0]
                klass_index = np.where(place == np.array(self.labels))[0]
                klass_selected_index = random.choices(klass_index, k=len(place_index))
                for i, index in enumerate(place_index):
                    batch[index] = klass_selected_index[i]
                
            #print("selected_place")
            #print(selected_place)
            

            # method 2) times 기반 class별 buffer 공간 나누기
            # then samplewise로 use_count 기반
            
        elif weight_method == "count_important":
            total_count = sum(self.class_usage_cnt)
            if total_count == 0:
                total_count = 1
            for klass, klass_count in enumerate(self.class_usage_cnt):
                # 아직 많이 학습에 안쓰인 애들
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = np.exp(-1*(klass_count/total_count))
                cls_weight.append(np.exp(-1*(klass_count/total_count)))

        elif weight_method == "mixed":
            total_count = sum(self.class_usage_cnt)
            if total_count == 0:
                total_count = 1
            max_time = max(self.cls_times) + 0.5

            for klass, klass_count in enumerate(self.class_usage_cnt):
                # 아직 많이 학습에 안쓰인 애들
                klass_time = self.cls_times[klass]
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = np.exp(-1*(klass_count/total_count))



        if self.weight_option == "softmax" or "loss":
            weight_tensor = torch.DoubleTensor(weight)
            weight = F.softmax(weight_tensor, dim=0)
            
        elif self.weight_option == "weightsum":
            weight = weight / sum(weight)

        return weight
    '''
    
    def get_std(self):
        class_std = np.std(self.class_usage_cnt)
        sample_std = np.std(self.usage_cnt)
        return class_std, sample_std
    
    
    def classwise_get_weight(self, weight_method, batch_size):
        weight = np.zeros(len(self.images))
        weight = weight.astype('float64')
        
        batch = np.zeros(batch_size)
        
        cls_weight = []
        
        if weight_method == "recent_important":
            #print("self.cls_times", self.cls_times)
            
            # method 1) times 기반 class에 weight 주기
            max_time = max(self.cls_times) + 0.4
            for klass, klass_time in enumerate(self.cls_times):
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = (klass_time+0.4) / max_time #np.exp(-1*(klass_count/total_count))
                cls_weight.append(np.exp(((klass_time+0.4) / max_time)))

            #print("selected_place")
            #print(selected_place)

            # method 2) times 기반 class별 buffer 공간 나누기
            # then samplewise로 use_count 기반
            
        elif weight_method == "count_important":
            total_count = sum(self.class_usage_cnt)
            if total_count == 0:
                total_count = 1
            for klass, klass_count in enumerate(self.class_usage_cnt):
                # 아직 많이 학습에 안쓰인 애들
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = np.exp(-1*(klass_count/total_count))
                cls_weight.append(np.exp(-1.5*(klass_count/total_count)))

        '''
        elif weight_method == "mixed":
            total_count = sum(self.class_usage_cnt)
            if total_count == 0:
                total_count = 1
            max_time = max(self.cls_times) + 0.5

            for klass, klass_count in enumerate(self.class_usage_cnt):
                # 아직 많이 학습에 안쓰인 애들
                klass_time = self.cls_times[klass]
                klass_index = np.where(klass == np.array(self.labels))[0]
                weight[klass_index] = np.exp(-1*(klass_count/total_count))
        '''
        
        if self.weight_option == "softmax" or "loss":
            weight_tensor = torch.DoubleTensor(cls_weight)
            cls_weight = F.softmax(weight_tensor, dim=0)
            
        elif self.weight_option == "weightsum":
            cls_weight = cls_weight / sum(cls_weight)

        return cls_weight

    
    def update_class_loss(self, indices, sample_loss):
        cls_loss_dict = {}
        
        # class별로 loss 묶기
        for i, index in enumerate(indices):
            klass = self.labels[index]
            
            if klass in cls_loss_dict.keys():
                cls_loss_dict[klass].append(sample_loss[i].item())
            else:
                cls_loss_dict[klass] = [sample_loss[i].item()]
        
        #self.previous_cls_loss = copy.deepcopy(self.cls_loss)
            
            
        # self.cls_loss update
        for klass in cls_loss_dict.keys():
            klass_loss_list = cls_loss_dict[klass]
            if len(klass_loss_list) == 1:
                klass_loss = klass_loss_list[0]
            else:
                klass_loss = np.mean(klass_loss_list)
            
            if self.cls_loss[klass] is not None:
                self.cls_loss[klass] = self.cls_loss[klass] * (1-self.weight_ema_ratio) + klass_loss * self.weight_ema_ratio
            else:
                self.cls_loss[klass] = klass_loss
                
        if self.use_human_training:
            self.transform_gpu.set_cls_magnitude(self.cls_loss)
            
    
    def update_sample_loss(self, indices, sample_loss):
        
        '''
        print("indices")
        print(indices)
        print("sample_loss")
        print(sample_loss)
        '''
        for i, index in enumerate(indices):
            #self.losses[index] = self.weight_ema_ratio * sample_loss[i].item()  + (1-self.weight_ema_ratio) * self.losses[index]
            self.losses[index] = sample_loss[i].item()
        
        '''
        print("losses")
        print(np.array(self.losses)[indices])
        '''
    
    def decrease_weight(self, cls_idx):
        self.cls_weight[cls_idx] *= self.cls_weight_decay

    def samplewise_get_weight(self, weight_method, indices = None):
        
        if indices is None:
            weight = np.zeros(len(self.images))
            weight = weight.astype('float64')
            for i, count in enumerate(self.usage_cnt):
                weight[i] = np.exp(-1*count)
        else:
            total_count = max(np.array(self.usage_cnt)[indices])
            if total_count == 0:
                total_count = 1
            weight = np.zeros_like(indices)
            weight = weight.astype('float64')
            for i, index in enumerate(indices):
                count = self.usage_cnt[index]
                weight[i] = np.exp(-3*((count**2)/total_count))
                
        #print("!count")
        #print(np.array(self.usage_cnt)[indices])
        #print("!weight", weight)
        
        if self.weight_option == "softmax" or self.weight_option == "loss":
            weight_tensor = torch.DoubleTensor(weight)
            weight = F.softmax(weight_tensor, dim=0)
            
        elif self.weight_option == "weightsum":
            weight = np.array(weight) / sum(weight)
        
        else:
            print("??")
        #print("!weight", weight)
        '''
        elif self.weight_option == "loss":
            
            losses = np.array(self.losses)
            losses = 1 / losses
            losses = torch.DoubleTensor(losses)
            weight = F.softmax(losses, dim=0)
            
            print("usage_count")
            print(self.usage_cnt)
            print("loss")
            print(self.losses)
            print("weight")
            print(weight)
        '''
        return weight
            
        
    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=None, transform=None, twf=False, recent_ratio=None, exp_weight=False, prev_batch_index=None, weight_method=None):
        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight == "classwise":
                weight = np.zeros(memory_batch_size)
                indices = np.zeros(memory_batch_size)
                cls_weight = self.classwise_get_weight(weight_method, memory_batch_size)

                # class별로 구간을 나누고 각 klass에 해당하는 sample들을 채우기
                selected_place = random.choices(range(len(self.cls_times)), k=memory_batch_size, weights=cls_weight) # klass가 몇개 select되는지 자리 배정

                for place in list(set(selected_place)):
                    place_index = np.where(place == np.array(selected_place))[0]
                    klass_index = np.where(place == np.array(self.labels))[0]
                    
                    if len(place_index) > len(klass_index):
                        klass_selected_index = random.sample(list(klass_index), len(klass_index))
                        klass_selected_index.extend(random.sample(range(len(self.images)), len(place_index) - len(klass_index)))
                    else:    
                        # random 기반
                        # klass_selected_index = random.sample(list(klass_index), len(place_index)) # TODO count based로 바꾸기
                        
                        # sample_count 기반
                        sample_weight = self.samplewise_get_weight(weight_method=weight_method, indices=klass_index)
                        #print("sample_weight", sample_weight)
                        klass_selected_index = np.random.choice(list(klass_index), size=len(place_index), replace=False, p=sample_weight)
                    
                    for i, index in enumerate(place_index):
                        indices[index] = klass_selected_index[i]
                        weight[index] = cls_weight[place]

                    # cls 기반 같은 class끼리 cutmix
                    # 너무 적게 뽑히는 class의 경우 걔로 인해서 좌지우지 될 수 있기 때문
                    
                    
                indices = indices.astype('int64')
            elif use_weight == "samplewise":
                weight = self.samplewise_get_weight(weight_method=weight_method)
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False, p=weight)
                
            else:
                if prev_batch_index is not None:
                    indices = np.zeros_like(prev_batch_index)
                    cand_dict = {}
                    cand_dict_index = {}
                    for i, index in enumerate(prev_batch_index):
                        target_i = self.labels[index]
                        
                        if target_i not in cand_dict.keys():
                            candidate_index = np.where(target_i == np.array(self.labels))[0]
                            cand_dict[target_i] = candidate_index
                            cand_dict_index[target_i] = [i]
                        else:
                            cand_dict_index[target_i].append(i)
                        
                    keys = list(set(list(cand_dict.keys())))
                    for key in keys:
                        candidate_index = cand_dict[key]
                        candidate_index_position = cand_dict_index[key]
                        indexes = np.random.choice(candidate_index, size=len(candidate_index_position), replace=False)
                        for i, index in enumerate(indexes):
                            indices[candidate_index_position[i]] = index
                    
                else:       
                    indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        
            # batch 내 select된 class를 count에 반영
            for i in indices:
                self.class_usage_cnt[self.labels[i]] += 1
        
            
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))
            
            # ER에서는 계속해서 학습에 사용되므로
            for i in stream_indices:
                self.class_usage_cnt[self.stream_labels[i]] += 1

        data = dict()
        buf_data = dict()
        images = []
        labels = []
        use_cnt = []
        logits = []
        counter = []
        d = []
        task_ids = []
        
        mean_usage = np.mean(self.usage_cnt)
        if self.use_kornia:
            # images
            if stream_batch_size > 0:
                for i in stream_indices:
                    images.append(self.stream_images[i])
                    labels.append(self.stream_labels[i])
                    
            if memory_batch_size > 0:
                for i in indices:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    use_cnt.append(self.usage_cnt[i])
                    if twf:
                        d.append(self.buf_transform(self.images[i].cpu(), self.attention_maps[i]))
                        logits.append(self.logits[i])
                        task_ids.append(self.tasks[i])
                    self.usage_cnt[i] += 1
            images = torch.stack(images).to(self.device)
            
            if self.use_human_training and self.cls_loss[0] is not None:
                #images = self.transform_gpu(images, use_cnt)
                images = self.transform_gpu(images, labels)
            else:
                images = self.transform_gpu(images)
        else:
            if stream_batch_size > 0:
                for i in stream_indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.stream_images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.stream_images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.stream_images[i].to(self.device)))
                        else:
                            images.append(transform(self.stream_images[i]))
                    labels.append(self.stream_labels[i])

            if memory_batch_size > 0:
                for i in indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.images[i].to(self.device)))
                        else:
                            images.append(transform(self.images[i]))

                    if twf:
                        d.append(self.buf_transform(self.images[i].cpu(), self.buf_attention_maps[i]))
                        logits.append(self.logits[i])
                        task_ids.append(self.tasks[i])


                    use_cnt.append(self.usage_cnt[i] / mean_usage)
                    labels.append(self.labels[i])
                    logits.append(self.logits[i])
                    self.cls_train_cnt[self.labels[i]] += 1
                    self.usage_cnt[i] += 1

            images = torch.stack(images)
        
        if use_weight=="classwise":
            data['cls_weight'] = weight
        data['counter'] = Counter(labels)
        data['image'] = images
        data['label'] = torch.LongTensor(labels)
        data['usage'] = torch.Tensor(use_cnt)
        if memory_batch_size>0:
            data['indices'] = indices#torch.LongTensor(labels)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)

        return data


    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    
    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data_1 = dict()
        data_2 = dict()
        images = []
        labels = []
        if self.use_kornia:
            # images
            for i in indices:
                images.append(self.images[i])
                labels.append(self.labels[i])
            images = torch.stack(images).to(self.device)
            data_1['image'] = self.transform_gpu(images)

        else:
            for i in indices:
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(self.images[i].to(self.device)))
                else:
                    images.append(self.transform(self.images[i]))
                labels.append(self.labels[i])
            data_1['image'] = torch.stack(images)
        data_1['label'] = torch.LongTensor(labels)
        data_1['index'] = torch.LongTensor(indices)
        images = []
        labels = []
        for i in indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        data_2['image'] = torch.stack(images)
        data_2['label'] = torch.LongTensor(labels)
        
        return data_1, data_2

    def update_std(self, y_list):
        for y in y_list:
            self.class_usage_cnt[y] += 1
        print("mir updated")
        print(self.class_usage_cnt)

    def make_cls_dist_set(self, labels, transform=None):
        if transform is None:
            transform = self.transform
        indices = []
        for label in labels:
            indices.append(np.random.choice(self.cls_idx[label]))
        indices = np.array(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def make_val_set(self, size=None, transform=None):
        if size is None:
            size = int(0.1*len(self.images))
        if transform is None:
            transform = self.transform
        size_per_cls = size//len(self.cls_list)
        indices = []
        for cls_list in self.cls_idx:
            if len(cls_list) >= size_per_cls:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=False))
            else:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=True))
        indices = np.concatenate(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def is_balanced(self):
        mem_per_cls = len(self.images)//len(self.cls_list)
        for cls in self.cls_count:
            if cls < mem_per_cls or cls > mem_per_cls+1:
                return False
        return True

class ASERMemory(MemoryDataset):
    def __init__(self, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_kornia=True, memory_size=None):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history)
        
        self.memory_size = memory_size

    def get_aser_train_batches(self):
        data = dict()
        images = []
        labels = []
        stream_indices = np.arange(len(self.stream_images)) 
        
        for i in stream_indices:
            images.append(self.stream_images[i])
            labels.append(self.stream_labels[i])
            self.class_usage_cnt[self.stream_labels[i]] += 1
        
        for i in self.batch_indices:
            images.append(self.images[i])
            labels.append(self.labels[i])
            self.class_usage_cnt[self.labels[i]] += 1

        images = torch.stack(images).to(self.device)
        labels = torch.LongTensor(labels)
        
        '''
        print("self.batch_indices")
        print(self.batch_indices)
        print("now")
        print(Counter(list(labels.numpy())))
        print("class total")
        print(self.class_usage_cnt)
        print("sampl total")
        print(self.usage_cnt)
        '''
        
        # use_kornia=True라고 가정되어 있음
        data['image'] = self.transform_gpu(images)
        data['label'] = labels    
        
        return data

    def get_aser_calculate_batches(self, n_smp_cls, memory_batch_size):
        #print("whole", Counter(self.labels))
        #print("current", Counter(self.stream_labels))
        current_data = dict()
        candidate_data = None
        eval_data = None
        
        ##### for current data #####
        images = []
        labels = []
        stream_indices = np.arange(len(self.stream_images)) 
        
        for i in stream_indices:
            # self.stream_images, self.images에는 이미 test_transform이 다 적용되어 있음
            images.append(self.test_transform(self.stream_images[i]))
            labels.append(self.stream_labels[i])
                
        images = torch.stack(images).to(self.device)
        labels = torch.LongTensor(labels)
        #current_data['image'] = self.transform_gpu(images) # 이건 aser에 못쓰임 augmentation 되어 있으므로 
        
        current_data['image'] = images
        current_data['label'] = labels

        if len(self.images) > 0:  
            candidate_data = dict()
            eval_data = dict()
            
            ##### for candidate data #####
            candidate_indices = self.get_class_balance_indices(n_smp_cls)
            images = []
            labels = []
            for i in candidate_indices:
                #TODO Transform??
                '''
                transform이 원래 aser에서는 그저 ToTensor만 해줌
                shape value 구할 때 transform 영향 없게 하려고 그런것 같은데.. 
                우리꺼에서는 그럼 test transform해서 shape value 한번 구하고 
                따로 augmentation 주는 step 한번 따로 해줘야 하는건가..? 애매하네
                '''
                #images.append(self.test_transform(self.images[i])) 이미 test_transform이 들어가 있음
                images.append(self.test_transform(self.images[i]))
                labels.append(self.labels[i])
            #print("candidate", Counter(labels))
            candidate_data['image'] = torch.stack(images)
            candidate_data['label'] = torch.LongTensor(labels)
            candidate_data['index'] = torch.LongTensor(candidate_indices)
            
            ##### for eval data #####           
            # discard indices는 겹치는 애들을 의미하며, 해당 index eval indices를 뽑을 때 빼주어야 한다.
            eval_indices = self.get_class_balance_indices(n_smp_cls, discard_indices = candidate_indices)
            images = []
            labels = []
            for i in eval_indices:
                images.append(self.test_transform(self.images[i]))
                labels.append(self.labels[i])
            #print("eval", Counter(labels))
            eval_data['image'] = torch.stack(images)
            eval_data['label'] = torch.LongTensor(labels)
        
        return current_data, candidate_data, eval_data

    def register_batch_indices(self, batch_indices=None, batch_size=None):
        if batch_indices is not None:
            self.batch_indices = batch_indices
        else:
            batch_indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
            self.batch_indices = batch_indices

    def get_class_balance_indices(self, n_smp_cls, discard_indices = None):
        indices = []
        for klass in range(len(self.cls_idx)):
            candidates = self.cls_idx[klass]
            if discard_indices is not None:
                candidates = list(set(candidates) - set(discard_indices))
            indices.extend(np.random.choice(candidates, size=min(n_smp_cls, len(candidates)), replace=False))
        return indices


class DistillationMemory(MemoryDataset):
    def __init__(self, dataset, memory_size, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, use_logit=True, use_feature=False, use_kornia=True):
        super().__init__(dataset, transform, cls_list, device, test_transform,
                 data_dir, transform_on_gpu, save_test, keep_history, use_kornia=use_kornia)
        self.logits = []
        self.features = []
        self.logits_mask = []
        self.use_logit = use_logit
        self.use_feature = use_feature
        self.logit_budget = 0.0
        self.memory_size = memory_size
        if self.dataset in ['cifar10', 'cifar100']:
            self.img_size = 32*32*3
        elif self.dataset == "tinyimagenet":
            self.img_size = 64*64*3
        elif self.dataset == "imagenet":
            self.img_size == 224*224*3
        else:
            raise NotImplementedError(
            "Please select the appropriate datasets (cifar10, cifar100, tinyimagenet, imagenet)"
            )
            
    def save_logit(self, logit, idx=None):
        if idx is None:
            self.logits.append(logit)
        else:
            self.logits[idx] = logit
        self.logits_mask.append(torch.ones_like(logit))

    def save_feature(self, feature, idx=None):
        if idx is None:
            self.features.append(feature)
        else:
            self.features[idx] = feature

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)
        for i, logit in enumerate(self.logits):
            self.logits[i] = torch.cat([logit, torch.zeros(1).to(self.device)])
            self.logits_mask[i] = torch.cat([self.logits_mask[i], torch.zeros(1).to(self.device)])
        if len(self.logits)>0:
            self.logit_budget = (len(self.logits) * len(self.logits[0])) / self.img_size
            print("logit_budget", self.logit_budget, "memory size", len(self.images))
            
            total_resource = math.ceil(self.logit_budget + len(self.images))
            num_discard_image = total_resource - self.memory_size
            if num_discard_image > 0:
                self.discard_images(num_discard_image)
            
    def discard_images(self, num_discard_image):
        
        print("num_discard_image", num_discard_image)
        target_index = random.sample(range(len(self.labels)), num_discard_image)
        target_index.sort()
        print("target_index")
        print(target_index)
        real_target_index = [idx-i for i, idx in enumerate(target_index)]
        print("real_target_index")
        print(real_target_index)
        
        for del_idx in real_target_index:
            print("del idx", del_idx)
            print(self.cls_idx[self.labels[del_idx]])
            self.cls_idx[self.labels[del_idx]].remove(del_idx)
            self.cls_count[self.labels[del_idx]] -= 1
            
            del self.images[del_idx]
            del self.labels[del_idx]
            del self.logits[del_idx]
            del self.logits_mask[del_idx]
            
            if self.use_feature:
                del self.features[del_idx]
        '''
        # step 1) class balanced를 맞춰서 discard
        per_klass = num_discard_image // len(self.cls_list) 
        cls_list = list(set(self.labels))
        # step 2) 나머지는 discard할 klass select하고 거기서만 제거
        # self.memory.discard_images(per_klass)
        additional_per_klass = random.sample(cls_list, num_discard_image % len(cls_list))

        print("per_klass", per_klass)
        print("additional_per_klass", additional_per_klass)
        
        for klass in range(len(self.cls_list)):
            klass_index = np.where(klass == np.array(self.labels))[0]
            print("klass_index")
            print(klass_index)
            if klass in additional_per_klass:
                num_discard = per_klass
            else:
                num_discard = per_klass + 1
                
            target_index = random.sample(list(klass_index), num_discard)
            target_index.sort()
            
            print("target_index")
            print(target_index)
            
            real_target_index = [idx-i for i, idx in enumerate(target_index)]
            print("real_target_index")
            print(real_target_index)
            
            for del_idx in real_target_index:
                del self.images[del_idx]
                del self.labels[del_idx]
                del self.logits[del_idx]
                del self.logits_mask[del_idx]
                
                if self.use_feature:
                    del self.features[del_idx]
            
        '''
            

    @torch.no_grad()
    def get_batch(self, batch_size, stream_batch_size=0, use_weight=False, transform=None):

        assert batch_size >= stream_batch_size
        stream_batch_size = min(stream_batch_size, len(self.stream_images))
        batch_size = min(batch_size, stream_batch_size + len(self.images))
        memory_batch_size = batch_size - stream_batch_size
        if memory_batch_size > 0:
            if use_weight:
                weight = self.get_weight()
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, p=weight / np.sum(weight),
                                           replace=False)
            else:
                indices = np.random.choice(range(len(self.images)), size=memory_batch_size, replace=False)
        if stream_batch_size > 0:
            if len(self.stream_images) > stream_batch_size:
                stream_indices = np.random.choice(range(len(self.stream_images)), size=stream_batch_size, replace=False)
            else:
                stream_indices = np.arange(len(self.stream_images))

        data = dict()
        images = []
        labels = []
        logits = []
        features = []
        logit_masks = []
        if self.use_kornia:
            # images
            if stream_batch_size > 0:
                for i in stream_indices:
                    images.append(self.stream_images[i])
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])
            images = torch.stack(images).to(self.device)
            images = self.transform_gpu(images)
        else:
            if stream_batch_size > 0:
                for i in stream_indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.stream_images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.stream_images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.stream_images[i].to(self.device)))
                        else:
                            images.append(transform(self.stream_images[i]))
                    labels.append(self.stream_labels[i])
            if memory_batch_size > 0:
                for i in indices:
                    if transform is None:
                        if self.transform_on_gpu:
                            images.append(self.transform_gpu(self.images[i].to(self.device)))
                        else:
                            images.append(self.transform(self.images[i]))
                    else:
                        if self.transform_on_gpu:
                            images.append(transform(self.images[i].to(self.device)))
                        else:
                            images.append(transform(self.images[i]))
                    labels.append(self.labels[i])
                    if self.use_logit:
                        logits.append(self.logits[i])
                        logit_masks.append(self.logits_mask[i])
                    if self.use_feature:
                        features.append(self.features[i])

            images = torch.stack(images)

        data['image'] = images
        data['label'] = torch.LongTensor(labels)
        if memory_batch_size > 0:
            if self.use_logit:
                data['logit'] = torch.stack(logits)
                data['logit_mask'] = torch.stack(logit_masks)
            if self.use_feature:
                data['feature'] = torch.stack(features)
        else:
            if self.use_logit:
                data['logit'] = torch.zeros(1)
                data['logit_mask'] = torch.zeros(1)
            if self.use_feature:
                data['feature'] =torch.zeros(1)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data


def get_train_datalist(dataset, sigma, repeat, init_cls, rnd_seed):
    with open(f"collections/{dataset}/{dataset}_sigma{sigma}_repeat{repeat}_init{init_cls}_seed{rnd_seed}.json") as fp:
        train_list = json.load(fp)
    return train_list['stream'], train_list['cls_dict'], train_list['cls_addition']

def get_test_datalist(dataset) -> List:
    print("test name", f"collections/{dataset}/{dataset}_val.json")
    return pd.read_json(f"collections/{dataset}/{dataset}_val.json").to_dict(orient="records")


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data_two(x, y, x2, y2, alpha=1.0, cutmix_prob=0.5, z=None):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    #index = torch.randperm(batch_size)

    '''
    if torch.cuda.is_available():
        index = index.cuda()
    '''
    if z is not None:
        z_a, z_b = z, z[index]
    y_a, y_b = y, y2
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    if z is None:
        return x, y_a, y_b, lam
    else:
        return x, y_a, y_b, lam, z_a, z_b

# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5, z=None):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    if z is not None:
        z_a, z_b = z, z[index]
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    if z is None:
        return x, y_a, y_b, lam
    else:
        return x, y_a, y_b, lam, z_a, z_b

def cutmix_feature(x, y, feature, prob, weight, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    feature_a, feature_b = feature, feature[index]
    prob_a, prob_b = prob, prob[index]
    weight_a, weight_b = weight, weight[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, feature_a, feature_b, prob_a, prob_b, weight_a, weight_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
