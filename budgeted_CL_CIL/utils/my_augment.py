import math
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch import Tensor

#from . import functional as F, InterpolationMode
from kornia.augmentation import RandomAffine, ColorJiggle, RandomSharpness, RandomPosterize, RandomSolarize, RandomEqualize, RandomInvert
__all__ = ["RandAugment", "TrivialAugmentWide"]


def get_op(
    op_name: str, magnitude: float
):
    if op_name == "ShearX":
        #TODO 180 단위 check
        return RandomAffine(degrees=0, translate=None, scale=None, shear=[-180,180,0,0], p=1.0)

    elif op_name == "ShearY":
        return RandomAffine(degrees=0, translate=None, scale=None, shear=[0,0,-180,180], p=1.0)

    elif op_name == "TranslateX":
        return RandomAffine(degrees=0, translate=(150.0 / 331.0, 0), scale=None, shear=None, p=1.0)

    elif op_name == "TranslateY":
        return RandomAffine(degrees=0, translate=(0, 150.0 / 331.0), scale=None, shear=None, p=1.0)

    elif op_name == "Rotate":
        # img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
        return RandomAffine(degrees=30, translate=None, scale=None, shear=None)
        
    elif op_name == "Brightness":
        # img = F.adjust_brightness(img, 1.0 + magnitude)
        return ColorJiggle(brightness = 1.0 + magnitude, p=1.0)

    elif op_name == "Color":
        # img = F.adjust_saturation(img, 1.0 + magnitude)
        return ColorJiggle(saturation = 1.0 + magnitude, p=1.0)

    elif op_name == "Contrast":
        # img = F.adjust_contrast(img, 1.0 + magnitude)
        return ColorJiggle(contrast = 1.0 + magnitude, p=1.0)

    elif op_name == "Sharpness":
        # img = F.adjust_sharpness(img, 1.0 + magnitude)
        return RandomSharpness(sharpness = 1.0 + magnitude, p=0.5) #TODO magnitude check

    elif op_name == "Posterize":
        # img = F.posterize(img, int(magnitude))
        return RandomPosterize(int(magnitude), p=1)

    elif op_name == "Solarize":
        # img = F.solarize(img, magnitude)
        return RandomSolarize(magnitude, p=1)

    #elif op_name == "AutoContrast":
    #img = F.autocontrast(img)

    elif op_name == "Equalize":
        # img = F.equalize(img)
        return RandomEqualize(p=1.0)

    elif op_name == "Invert":
        # img = F.invert(img)
        return RandomInvert(p=1.0)

    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")


class Kornia_Randaugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 16,
        #interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.original_magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        #self.interpolation = interpolation
        self.fill = fill
        # Formula 
        #

    def set_aug_space(self, y_min, y_max, num_bins):
        #x = np.linspace(0, 8, num_bins)
        x = np.array(range(num_bins))
        #y = (y_min+y_max)-(y_min + 0.5 * (y_max - y_min) * (1 + np.cos(((x%self.T)/self.T) * np.pi)))
        y = (y_min+y_max)-(y_min + 0.5 * (y_max - y_min) * (1 + np.cos((x/num_bins) * np.pi)))
        return y

    '''
    "Identity": (torch.tensor(0.0), False),
    "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
    "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
    "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * 1, num_bins), True),
    "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * 1, num_bins), True),
    "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
    "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
    "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
    #"AutoContrast": (torch.tensor(0.0), False),
    "Equalize": (torch.tensor(0.0), False),
    '''

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (self.set_aug_space(0.04, 0.14, num_bins), True),
            "ShearY": (self.set_aug_space(0.04, 0.14, num_bins), True),
            "TranslateX": (self.set_aug_space(0.06042296, 0.21148036, num_bins), True),
            "TranslateY": (self.set_aug_space(0.06042296, 0.21148036, num_bins), True),
            "Rotate": (self.set_aug_space(4, 14, num_bins), True),
            "Brightness": (self.set_aug_space(0.12, 0.42, num_bins), True),
            "Color": (self.set_aug_space(0.12, 0.42, num_bins), True),
            "Contrast": (self.set_aug_space(0.12, 0.42, num_bins), True),
            "Sharpness": (self.set_aug_space(0.12, 0.42, num_bins), True),
            #"Posterize": (8 - (np.range(num_bins) / ((num_bins - 1) / 4)).round(), False),
            "Posterize": ((8 - (np.array(range(num_bins)) / ((num_bins - 1) / 4)).round())[::-1], False),
            "Solarize": (self.set_aug_space(221.0, 136.0, num_bins), False),
            #"AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def set_num_magnitude_bins(self, magnitude):
        self.magnitude = magnitude

    def form_transforms(self):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        '''
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]
        '''
        #if train_count is None:
        ops = []
        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins)
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            if op_name == 'Identity':
                continue
            op = get_op(op_name, magnitude)
            ops.append(op)
        return ops

        '''
        else:
            total_ops = []
            for count in train_count:
                ops = []
                for _ in range(self.num_ops):
                    op_meta = self._augmentation_space(self.num_magnitude_bins)
                    op_index = int(torch.randint(len(op_meta), (1,)).item())
                    op_name = list(op_meta.keys())[op_index]
                    magnitudes, signed = op_meta[op_name]

                    #magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                    index = int(count/self.num_magnitude_bins) + count%self.num_magnitude_bins
                    if index >= self.num_magnitude_bins:
                        index = self.num_magnitude_bins - 1
                    magnitude = float(magnitudes[index].item()) if magnitudes.ndim > 0 else 0.0

                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0
                    if op_name == 'Identity':
                        continue
                    op = get_op(op_name, magnitude)
                    ops.append(op)
                total_ops.append(ops)
            return total_ops
            '''
    '''
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s
    '''

