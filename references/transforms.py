import cv2
import numpy as np
import torch
import torchvision

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional


MIN_PIXEL_VAL = 0.1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        tgt = None
        if target:
            tgt = {}
            for k, v in target.items():
                tgt[k] = target[k].clone()
        for t in self.transforms:
            image, tgt = t(image, tgt)
        return image, tgt

    def forward(self, image, target = None):
        tgt = None
        if target:
            tgt = {}
            for k, v in target.items():
                tgt[k] = target[k].clone()
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

    def __call__(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height = F._get_image_size(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        return image, target

    def __call__(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height = F._get_image_size(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

    def __call__(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not torch.is_tensor(image):
            image = F.to_tensor(image) 
        return image, target

class Normalize(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def forward(self, image, target = None):
        image = torchvision.transforms.Normalize(self.mean, self.std)(image)
        return image, target

    def __call__(self, image, target = None):
        image = torchvision.transforms.Normalize(self.mean, self.std)(image)
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1.0, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0, sampler_options: Optional[List[float]] = None, trials: int = 40):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F._get_image_size(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(boxes, torch.tensor([[left, top, right, bottom]],
                                                                         dtype=boxes.dtype, device=boxes.device))
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class RandomZoomOut(nn.Module):
    def __init__(self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1., 4.), p: float = 0.5):
        super().__init__()
        if fill is None:
            fill = [0., 0., 0.]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1. or side_range[0] > side_range[1]:
            raise ValueError("Invalid canvas side range provided {}.".format(side_range))
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) < self.p:
            return image, target

        orig_w, orig_h = F._get_image_size(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h):, :] = \
                image[..., :, (left + orig_w):] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(self, contrast: Tuple[float] = (0., 1.5), saturation: Tuple[float] = (0., 1.5),
                 hue: Tuple[float] = (-0.1, 0.1), brightness: Tuple[float] = (0.875, 1.125), p: float = 0.5):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels = F._get_image_num_channels(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.to_tensor(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target


class RandomRotation(nn.Module):
    def __init__(self, degrees: float = 150, p: float = 0.5):
        super().__init__()
        self.degrees = degrees
        self.p = p

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            #print("IN!")
            angle = float(torch.empty(1).uniform_(float(-180), float(180)).item())
            image = F.rotate(image, angle)
            if target is not None:
                width, height = F._get_image_size(image)

                target["boxes"][:, [0, 2]] = target["boxes"][:, [2, 0]] - width/2
                target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]] - height/2

                for i, box in enumerate(target["boxes"]):
                    #print(box)
                    new_box = self.get_new_box(box, angle)
                    target["boxes"][i,:] = new_box
                    #print(new_box, angle)
                
                target["boxes"][:, [0, 2]] = target["boxes"][:, [2, 0]] + width/2
                target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]] + height/2
                target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1] ) * (target["boxes"][:, 2] - target["boxes"][:, 0] )
        return image, target

    def __call__(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        
        if torch.rand(1) < self.p:
        #if True:
            #print("IN!")
            angle = float(torch.empty(1).uniform_(float(-180), float(180)).item())
            image = F.rotate(image, angle)
            if target is not None:
                width, height = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = target["boxes"][:, [2, 0]] - width/2
                target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]] - height/2
                for i, box in enumerate(target["boxes"]):
                    #print(box)
                    new_box = self.get_new_box(box, angle)
                    target["boxes"][i,:] = new_box
                    #print(new_box)
                
                target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]] + width/2
                target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]] + height/2
                #print(target["boxes"])
                #print(width, height)
                target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1] ) * (target["boxes"][:, 2] - target["boxes"][:, 0] )
        return image, target

    def get_new_coordinate(self, old_coordinate, angle):
        radius = (old_coordinate[0]**2 + old_coordinate[1]**2) ** 0.5
        old_angle = np.arctan2(old_coordinate[1], old_coordinate[0])
        new_angle = old_angle - np.pi*angle/180
        new_x = np.cos(new_angle)*radius
        new_y = np.sin(new_angle)*radius
        return [new_x, new_y]

    def get_new_box(self, box, angle):
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]

        new_coordinates = np.zeros((4,2))
        new_coordinates[0,:] = self.get_new_coordinate((x_min, y_min), angle)
        new_coordinates[1,:] = self.get_new_coordinate((x_min, y_max), angle)
        new_coordinates[2,:] = self.get_new_coordinate((x_max, y_min), angle)
        new_coordinates[3,:] = self.get_new_coordinate((x_max, y_max), angle)

        x_min = np.min(new_coordinates[:,0])
        x_max = np.max(new_coordinates[:,0])
        y_min = np.min(new_coordinates[:,1])
        y_max = np.max(new_coordinates[:,1])
        return torch.tensor([x_min, y_min, x_max, y_max])


class MotionBlur(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        #self.kernel_sizes = np.arange(21, 79, 4)
        self.kernel_sizes = np.arange(15, 35, 4)

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        kernel_size = np.random.choice(self.kernel_sizes)
        if torch.rand(1) < self.p:
            # Horizontal
            # applying the kernel to the input image
            image = cv2.filter2D(image.numpy(), -1, self.hkernel)
        elif torch.rand(1) < self.p:
            image = cv2.filter2D(image.numpy(), -1, self.vkernel)
        return F.to_tensor(image), target

    def __call__(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        kernel_size = np.random.choice(self.kernel_sizes)
        if torch.rand(1) < self.p:
            #print("horizontal")
            # applying the kernel to the input image
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel/kernel_size
            image = F.to_tensor(cv2.filter2D(image.permute(1,2,0).numpy(), -1, kernel))
        elif torch.rand(1) < self.p:
            #print("vertical")
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
            kernel = kernel/kernel_size
            image = F.to_tensor(cv2.filter2D(image.permute(1,2,0).numpy(), -1, kernel))
        return image, target


# This class is used ONLY for CycleGAN training
class RandomCrop(nn.Module):
    def __init__(self, size=[512, 512], content_threshold=0.30, contain_target = True):
        super().__init__()
        self.size = size
        self.crop = T.RandomCrop(self.size)
        self.content_threshold = content_threshold
        self.contain_target = contain_target

    def forward(self, image:Tensor,
            target: Optional[Dict[str, Tensor]] = None):
        return image, target

    def __call__(self, image: Tensor,
                    target: Optional[Dict[str, Tensor]] = None):
        cropped = self.crop(image)
        #print(torch.max(image), torch.sum(image > MIN_PIXEL_VAL)/np.prod(image.shape))
        count = 0
        while ((torch.sum(image > MIN_PIXEL_VAL)/np.prod(image.shape)) < self.content_threshold) and (count < 15):
            cropped = self.crop(image)
            count += 1
        return cropped, target

class UnNormalize(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
