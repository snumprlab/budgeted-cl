## the code is mostly taken from autoaugment pytorch repo:
# https://github.com/DeepVoltaire/AutoAugment


from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import logging
import json

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from kornia import image_to_tensor, tensor_to_image
import kornia.augmentation as K
from kornia.geometry.transform import resize

from utils.my_augment import Kornia_Randaugment

logger = logging.getLogger()


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


class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size=(inp_size, inp_size)),
            K.RandomCrop(size=(inp_size, inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
        )
        # self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        # if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size=(self.inp_size, self.inp_size)),
            K.RandomCrop(size=(self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
        )
        # print("transform")
        # print(self.transforms)
        x_out = self.transforms(x)  # BxCxHxW
        return x_out

def get_transform(dataset, transform_list, method_name, type_name, transform_on_gpu=False, input_size=None):
    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset, type_name=type_name)
    if input_size is not None:
        inp_size = input_size
        
    train_transform = []
    if "cutout" in transform_list:
        train_transform.append(Cutout(size=16))
    if "randaug" in transform_list:
        train_transform.append(transforms.RandAugment())
    if "autoaug" in transform_list:
        if hasattr(transforms, 'AutoAugment'):
            if 'cifar' in dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in dataset:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
        else:
            train_transform.append(select_autoaugment(dataset))
    if "trivaug" in transform_list:
        train_transform.append(transforms.TrivialAugmentWide())
    if transform_on_gpu:
        cpu_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.PILToTensor()
            ])
        if "xder" in method_name:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ConvertImageDtype(torch.float32)
                ]
            )            
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        cpu_transform = None
        if "xder" in method_name:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor()
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    logger.info(f"Using train-transforms {train_transform}")

    if "xder" in method_name:
        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return train_transform, test_transform, cpu_transform, n_classes


def select_autoaugment(dataset):
    logger.info(f"{dataset}: autoaugmentation is applied")
    if 'imagenet' in dataset:
        return ImageNetPolicy()
    else:
        return CIFAR10Policy()

class ImageNetPolicy(object):
    """Randomly choose one of the best 24 Sub-policies on ImageNet.
    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.
    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """Randomly choose one of the best 25 Sub-policies on SVHN.
    Example:
    >>> policy = SVHNPolicy()
    >>> transformed = policy(image)
    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     SVHNPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),
            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class Cutout:
    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[
            upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :
        ] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img


class Invert:
    def __call__(self, pil_img):
        return ImageOps.invert(pil_img)


class Equalize:
    def __call__(self, pil_img):
        return ImageOps.equalize(pil_img)


class Solarize:
    def __init__(self, v):
        assert 0 <= v <= 256
        self.v = v

    def __call__(self, pil_img):
        return ImageOps.solarize(pil_img, self.v)
    
def get_statistics(dataset: str, type_name:str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
        
    assert dataset in [
        "cifar10", 
        "PACS",
        "OfficeHome",
        "DomainNet",
        "birdsnap",
        "aircraft", 
        "food101",
        "cct", 
        "PACS_final",
        "NICO"
    ]

    with open('./utils/data_statistics.json', 'r') as f:
        data = json.load(f)

    classes = {
        "cifar10": 10,
        "PACS": 7,
        "PACS_final": 7,
        "OfficeHome": 65,
        "DomainNet": 345,
        "birdsnap": 500,
        "aircraft": 100,
        "food101": 101,
        "cct": 12,
        "NICO": 60
    }

    in_channels = 3
    inp_size = 224
    
    return (
        data['mean'][dataset][type_name],
        data['std'][dataset][type_name],
        classes[dataset],
        inp_size,
        in_channels,
    )