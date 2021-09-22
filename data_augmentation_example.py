# Python script to display the various transformations for data augmentation
import cv2
import numpy as np
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

def show_differences(imgs, titles=''):
	fig, axs = plt.subplots(1,len(imgs), figsize=(len(imgs)*5,5))
	for i, img in enumerate(imgs):
		axs[i].imshow(img.permute(1,2,0).numpy())
		axs[i].set_xticks([])
		axs[i].set_yticks([])
		axs[i].set_title(titles[i])
	fig.suptitle(titles[-1])
	plt.show()

img_path = r'C:\Users\jazma\RA\dataset\ascaris\ass95.jpg'
img = Image.open(img_path).convert("RGB")

img = F.to_tensor(img)

mean = 0
std = 1.
img = torchvision.transforms.Normalize(mean, std)(img)

# Saturation
img_min = T.ColorJitter(saturation=0.)(img)
img_max = T.ColorJitter(saturation=(1.5, 1.5))(img)
#show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Saturation"])

# Contrast
img_min = T.ColorJitter(contrast=0.)(img)
img_max = T.ColorJitter(contrast=(1.5,1.5))(img)
#show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Contrast"])

# Hue
img_min = T.ColorJitter(hue=(-0.1, -0.1))(img)
img_max = T.ColorJitter(hue=(0.1,0.1))(img)
#show_differences([img_min, img, img_max], ["Min (-0.1)", "Original", "Max (0.1)", "Hue"])

#Brightness
img_min = T.ColorJitter(brightness=(0.5, 0.5))(img)
img_max = T.ColorJitter(brightness=(1.7,1.7))(img)
#show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Brightness"])

# Rotation
img_min = F.rotate(img, -150)
img_max = F.rotate(img, 150)
#show_differences([img_min, img, img_max], ["Min (-150)", "Original", "Max (150)", "Rotation"])

# HFlip
img_flip = F.hflip(img)
#show_differences([img, img_flip], ["Original", "After Flip", "Horizontal Flip"])

# Vflip
img_flip = F.vflip(img)
#show_differences([img, img_flip], ["Original", "After Flip", "Vertical Flip"])

# Motion Blurring
kernel_size = 71 # 71- 21
hkernel = np.zeros((kernel_size, kernel_size))
vkernel = np.zeros((kernel_size, kernel_size))
hkernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
vkernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
hkernel = hkernel/kernel_size
vkernel = vkernel/kernel_size

hblur = F.to_tensor(cv2.filter2D(img.permute(1,2,0).numpy(), -1, hkernel))
vblur = F.to_tensor(cv2.filter2D(img.permute(1,2,0).numpy(), -1, vkernel))
show_differences([hblur, img, vblur], ["Horizontal", "Original", "Vertical", "Motion blurring"])