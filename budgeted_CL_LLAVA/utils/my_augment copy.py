import math
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch import Tensor
import copy
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

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
        magnitude: int = 2,
        num_magnitude_bins: int = 10,
        #interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        T = None,
        min_magnitude: int = 1,
        max_magnitude: int = 9
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.original_magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        #self.interpolation = interpolation
        self.fill = fill
        self.T = T
        self.prev_cls_loss = []
        self.cls_magnitude = []
        self.cls_num_ops = []
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude

    # loss gradient
    def set_cls_magnitude(self, option, cls_loss, class_count):
        
        #if option == "cls_acc":
        #elif option == "cls_loss":
        # step 1) warm up 적용
        #cls_list = [cls for cls, loss in enumerate(cls_loss) if class_count[cls]>=20]
        #cls_loss = [loss for cls, loss in enumerate(cls_loss) if class_count[cls]>=20]
        
        print("class_count")
        print(class_count)
        
        '''
        print("cls_list")
        print(cls_list)
        print("cls_loss")
        print(cls_loss)
        '''
        
        # step 2) linear regression 적용
        for cls in range(len(cls_loss)):
            if cls < len(self.prev_cls_loss):
                self.prev_cls_loss[cls].append(cls_loss[cls])
            else:
                self.prev_cls_loss.append([cls_loss[cls]])
            
            line_fitter = LinearRegression()
            length = min(3, len(self.prev_cls_loss[cls]))
            '''
            if length==1:
                pass
            else:
            '''
            line_fitter.fit(np.array((range(length))).reshape(-1,1), self.prev_cls_loss[cls][-length:])
            print("cls 기울기", line_fitter.coef_)
            
            if cls >= len(self.cls_magnitude):
                # 처음 들어온 class라면 default 값인 self.magnitude로 채워주기
                self.cls_magnitude.append(self.magnitude)
                self.cls_num_ops.append(2)
                
            else:        
                if class_count[cls]<20:
                    #warm up stage
                    continue
                
                if option == "class_loss":
                    # 기울기 check
                    # TODO Tolerance check
                    if abs(line_fitter.coef_) <= 0.01:
                        # 기울기 0 허용 범위 내라면 stay
                        pass
                    
                    elif line_fitter.coef_ < 0:
                        # 허용범위 외 음수라면 magnitude 증가시키기
                        self.cls_magnitude[cls] += 1
                        
                    elif line_fitter.coef_ > 0:
                        # 허용범위 외 양수라면 magnitude 감소시키기
                        self.cls_magnitude[cls] -= 1

                elif option == "class_acc":
                    # 기울기 check
                    # TODO Tolerance check
                    if abs(line_fitter.coef_) <= 0.01:
                        # 기울기 0 허용 범위 내라면 stay
                        pass
                    
                    elif line_fitter.coef_ > 0:
                        # 허용범위 외 음수라면 magnitude 증가시키기
                        self.cls_magnitude[cls] += 1
                        
                    elif line_fitter.coef_ < 0:
                        # 허용범위 외 양수라면 magnitude 감소시키기
                        self.cls_magnitude[cls] -= 1
                    
                
                # CLIP for diff # of ops
                if self.cls_magnitude[cls] <= 0:
                    if self.cls_num_ops[cls] > 2:
                        #self.num_ops -= 1
                        self.cls_num_ops[cls] -= 1
                        self.cls_magnitude[cls] = 9
                    else:
                        self.cls_magnitude[cls] = 1
                    
                
                elif self.cls_magnitude[cls] >= 10:
                    if self.cls_num_ops[cls] == 3:
                        self.cls_magnitude[cls] = 9
                    else:
                        #self.num_ops+=1
                        self.cls_num_ops[cls] += 1
                        self.cls_magnitude[cls] = 1
                
                
                '''
                # CLIP for same # of ops
                if self.cls_magnitude[cls] <= 0:
                    self.cls_magnitude[cls] = 1
                
                elif self.cls_magnitude[cls] >= 10:
                    self.cls_magnitude[cls] = 9
                '''
                        
                

    '''
    # normalize 
    def set_cls_magnitude(self, cls_loss, class_count):
        # step 1) class loss weight sum
        #avg_loss = sum(np.array(cls_loss) * (np.array(class_count) / sum(class_count)))
        cls_loss = [loss for cls, loss in enumerate(cls_loss) if class_count[cls]>=50]
        avg_loss = np.mean(cls_loss)

        print("cls_loss")
        print(cls_loss)
        print("class_count")
        print(class_count)

        if len(cls_loss) <= 1:
            self.cls_magnitude = np.ones(len(class_count))
        else:
            std_loss = np.std(cls_loss)
            norm_loss =  (np.array(cls_loss) - avg_loss) / std_loss
            cls_magnitude = [10-round(norm.cdf(stat)*10) for stat in norm_loss]
            
            # cls_count가 warm_up count 이하면 cls_magnitude 계산시 zero, 일단 magnitude 0으로 시작
            print("cls_magnitude")
            print(cls_magnitude)
            
            self.cls_magnitude = np.ones(len(cls_magnitude))
            for cls, magnitude in enumerate(cls_magnitude):
                self.cls_magnitude[cls] = magnitude
    '''

    def get_cls_num_ops(self):
        return self.cls_num_ops 

    def get_cls_magnitude(self):
        return self.cls_magnitude
    '''
    def set_cls_magnitude(self, current_cls_loss):

        if self.prev_cls_loss is None:
            self.prev_cls_loss = copy.deepcopy(current_cls_loss)
            self.cls_magnitude = [self.magnitude for _ in range(len(current_cls_loss))] # set default
            
        else:
            for klass, loss in enumerate(self.prev_cls_loss):
                if current_cls_loss[klass] < loss:
                    self.cls_magnitude[klass] -= 1
                    self.cls_magnitude[klass] = max(self.min_magnitude, self.cls_magnitude[klass])
                else:
                    self.cls_magnitude[klass] += 1
                    self.cls_magnitude[klass] = min(self.max_magnitude, self.cls_magnitude[klass])
            
            # add_new_class인 상황
            if len(self.prev_cls_loss) != len(current_cls_loss):
                num_new_class = len(current_cls_loss) - len(self.prev_cls_loss)
                
                for _ in range(num_new_class):
                    self.cls_magnitude.append(self.magnitude)
                    
            self.prev_cls_loss = copy.deepcopy(current_cls_loss)
                
    '''
    
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

    def form_transforms(self, klasses=None):
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
        
        
        '''
        print("self.prev_cls_loss")
        print(self.prev_cls_loss)
        print("self.cls_magnitude")
        print(self.cls_magnitude)
        '''
        
        if klasses is None:
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
        
        else:
            #total_ops = [[] for _ in range(len(klasses))]
            total_ops = {}
            for klass in klasses:
                total_ops[klass] = []
            
            print("klasses")
            print(klasses)
            print("cls_num_ops")
            print(self.cls_num_ops)
            print("self.cls_magnitude")
            print(self.cls_magnitude)
            
            for i in range(max(self.cls_num_ops)):
                op_meta = self._augmentation_space(self.num_magnitude_bins)
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                
                for klass in klasses:
                    
                    if klass >= len(self.cls_magnitude):
                        index = self.magnitude
                        num_ops = self.num_ops
                    else:
                        index = self.cls_magnitude[klass]
                        num_ops = self.cls_num_ops[klass]
                    
                    # class별로 num_ops 개수가 다르다.
                    if i >= num_ops:
                        continue
                        
                    magnitude = float(magnitudes[index].item()) if magnitudes.ndim > 0 else 0.0

                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0
                        
                    if op_name == 'Identity':
                        continue
                    
                    op = get_op(op_name, magnitude)
                    total_ops[klass].append(op)
                #total_ops.append(ops)
                
            return total_ops
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

