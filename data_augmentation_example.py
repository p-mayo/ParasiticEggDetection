# Python script to display the various transformations for data augmentation
import cv2
import numpy as np
import torch
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

def show_differences(imgs, titles='', output_path=""):
	fig, axs = plt.subplots(1,len(imgs), figsize=(len(imgs)*5,5))
	for i, img in enumerate(imgs):
		axs[i].imshow(img.permute(1,2,0).numpy())
		axs[i].set_xticks([])
		axs[i].set_yticks([])
		axs[i].set_title(titles[i])
	fig.suptitle(titles[-1])
	if output_path:
		plt.savefig(output_path)
	else:
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
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\saturation.png'
show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Saturation"], output_path)

# Contrast
img_min = T.ColorJitter(contrast=0.)(img)
img_max = T.ColorJitter(contrast=(1.5,1.5))(img)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\contrast.png'
show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Contrast"], output_path)

# Hue
img_min = T.ColorJitter(hue=(-0.1, -0.1))(img)
img_max = T.ColorJitter(hue=(0.1,0.1))(img)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\hue.png'
show_differences([img_min, img, img_max], ["Min (-0.1)", "Original", "Max (0.1)", "Hue"], output_path)

#Brightness
img_min = T.ColorJitter(brightness=(0.5, 0.5))(img)
img_max = T.ColorJitter(brightness=(1.7,1.7))(img)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\brightness.png'
show_differences([img_min, img, img_max], ["Min (0)", "Original", "Max (1.5)", "Brightness"], output_path)

# Rotation
img_min = F.rotate(img, -150)
img_max = F.rotate(img, 150)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\rotation.png'
show_differences([img_min, img, img_max], ["Min (-150)", "Original", "Max (150)", "Rotation"], output_path)

# HFlip
img_flip = F.hflip(img)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\horizontalflip.png'
show_differences([img, img_flip], ["Original", "After Flip", "Horizontal Flip"], output_path)

# Vflip
img_flip = F.vflip(img)
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\verticalflip.png'
show_differences([img, img_flip], ["Original", "After Flip", "Vertical Flip"], output_path)

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
output_path = r'C:\Users\jazma\RA\data_augmentation_examples\motionblur.png'
show_differences([hblur, img, vblur], ["Horizontal", "Original", "Vertical", "Motion blurring"], output_path)