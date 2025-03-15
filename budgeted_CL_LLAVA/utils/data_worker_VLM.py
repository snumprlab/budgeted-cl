import torch
import random
import os
import sys
import queue

import PIL
import numpy as np
from utils.augment import DataAugmentation, Preprocess, get_statistics

from PIL import Image
import copy
from utils.data_loader_VLM import preprocess_multimodal, preprocess_text_llava, preprocess_text_bunny

IS_WINDOWS = sys.platform == "win32"
TIMEOUT = 5.0
Image.MAX_IMAGE_PIXELS = None

# from PyTorch Official Code
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


def load_data(sample, data_dir, transform=None):
    img_name = sample["file_name"]
    img_path = os.path.join(data_dir, img_name)
    image = PIL.Image.open(img_path).convert("RGB")
    if transform:
        image = transform(image)
    return image

@torch.no_grad()
def worker_multimodal(r,device='cpu', tokenizer=None, data_args=None):
    data = dict()
    images = []
    input_ids = []
    labels = []
    not_aug_img = []
    if len(r) > 0:
        for sample in r:
            processor = data_args.image_processor
            if 'image' in sample:
                image_file = sample['image']
                # if ' |sep| ' in image_file:
                if isinstance(image_file, list):
                    image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
                else:
                    image = [Image.open(image_file).convert('RGB')]
                if data_args.image_aspect_ratio == 'pad':
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
                images.append(image)
                sources = preprocess_multimodal(
                    copy.deepcopy([sample['conversations']]),
                    data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sample])
            
            if 'llava' in data_args.model_name_for_dataarg.lower():
                data_dict = preprocess_text_llava(
                    sources,
                    tokenizer,
                    has_image=('image' in sample))
            elif 'bunny' in data_args.model_name_for_dataarg.lower():
                data_dict = preprocess_text_bunny(
                    sources,
                    tokenizer,
                    has_image=('image' in sample))
            
            input_ids.append(data_dict['input_ids'][0])
            labels.append(data_dict['labels'][0])

        data['image'] = images#torch.stack(images)
        data['input_id'] = input_ids#torch.stack(input_ids)
        data['label'] = labels#torch.stack(labels)
        data['sample'] = r
        # if not scl and test_transform is not None:
        #     data['not_aug_img'] = not_aug_img
        return data
    else:
        return None


@torch.no_grad()
def worker_loop_multimodal(index_queue, data_queue, device='cpu', tokenizer=None, data_args=None):
    torch.set_num_threads(1)
    watchdog = ManagerWatchdog()
    while watchdog.is_alive():
        try:
            r = index_queue.get(timeout=TIMEOUT)
        except queue.Empty:
            continue

        data = dict()
        images = []
        input_ids = []
        labels = []
        not_aug_img = []
        if len(r) > 0:
            processor = data_args.image_processor
            for sample in r:
                if 'image' in sample:
                    image_file = sample['image']
                    # if ' |sep| ' in image_file:
                    if isinstance(image_file, list):
                        image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
                    else:
                        image = [Image.open(image_file).convert('RGB')]
                    if data_args.image_aspect_ratio == 'pad':
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
                    else:
                        image = torch.stack([processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
                    images.append(image)
                    sources = preprocess_multimodal(
                        copy.deepcopy([sample['conversations']]),
                        data_args)
                else:
                    sources = copy.deepcopy([sample["conversations"]])
                    images.append(torch.zeros(0))
                if 'llava' in data_args.model_name_for_dataarg.lower():
                    data_dict = preprocess_text_llava(
                        sources,
                        tokenizer,
                        has_image=('image' in sample))
                elif 'bunny' in data_args.model_name_for_dataarg.lower():
                    data_dict = preprocess_text_bunny(
                        sources,
                        tokenizer,
                        has_image=('image' in sample))
                
                input_ids.append(data_dict['input_ids'][0])
                labels.append(data_dict['labels'][0])

            data['image'] = images#torch.stack(images)
            data['input_id'] = input_ids#torch.stack(input_ids)
            data['label'] = labels#torch.stack(labels)
            data['sample'] = r
            data_queue.put(data)
        else:
            data_queue.put(None)