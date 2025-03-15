import torch
import random
import os
import sys
import queue

import PIL
import numpy as np
from utils.augment import DataAugmentation, Preprocess, get_statistics

IS_WINDOWS = sys.platform == "win32"
TIMEOUT = 5.0

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
def worker_loop(index_queue, data_queue, data_dir, transform, transform_on_gpu=False, cpu_transform=None, device='cpu', use_kornia=False, transform_on_worker=True, test_transform=None, scl=False):
    torch.set_num_threads(1)
    watchdog = ManagerWatchdog()
    if use_kornia:
        if 'cifar100' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar100')
        elif 'cifar10' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
        elif 'tinyimagenet' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='tinyimagenet')
        elif 'imagenet' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='imagenet')
        preprocess = Preprocess(inp_size)
        kornia_randaug = DataAugmentation(inp_size, mean, std)
    while watchdog.is_alive():
        try:
            r = index_queue.get(timeout=TIMEOUT)
        except queue.Empty:
            continue
        data = dict()
        images = []
        labels = []
        not_aug_img = []
        if len(r) > 0:
            for sample in r:
                if use_kornia:
                    img_name = sample["file_name"]
                    img_path = os.path.join(data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    images.append(preprocess(image))
                elif transform_on_gpu:
                    images.append(load_data(sample, data_dir, cpu_transform))
                    if not scl and test_transform is not None:
                        not_aug_img.append(load_data(sample, data_dir, test_transform))
                else:
                    if scl:
                        images.append(load_data(sample, data_dir, test_transform))
                    else:
                        images.append(load_data(sample, data_dir, transform))
                        if test_transform is not None:
                            not_aug_img.append(load_data(sample, data_dir, test_transform))
                labels.append(sample["label"])
            if transform_on_worker:
                if use_kornia:
                    images = kornia_randaug(torch.stack(images).to(device))
                elif transform_on_gpu:
                    if scl:
                        images = torch.stack(images).to(device)
                    else:
                        images = transform(torch.stack(images).to(device))
                        if test_transform is not None:
                            not_aug_img = torch.stack(not_aug_img).to(device)
                else:
                    images = torch.stack(images)
                    if not scl and test_transform is not None:
                        not_aug_img = torch.stack(not_aug_img).to(device)
            else:
                images = torch.stack(images)
                if not scl and test_transform is not None:
                    not_aug_img = torch.stack(not_aug_img)
            data['image'] = images
            data['label'] = torch.LongTensor(labels)
            if not scl and test_transform is not None:
                data['not_aug_img'] = not_aug_img
            data_queue.put(data)
        else:
            data_queue.put(None)
