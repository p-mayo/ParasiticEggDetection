# Python file for the dataset loader
# for the CycleGAN implementation
 
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

def get_file_content(file_name):
	with open(file_name) as f:
		lines = f.readlines()
	return lines

class CycleGANDataset(Dataset):
	def __init__(self, root_domain_a, root_domain_b, transforms_a = None, transforms_b = None):
		self.root_domain_a = root_domain_a
		self.root_domain_b = root_domain_b
		self.transforms_a = transforms_a
		self.transforms_b = transforms_b

		self.domain_a_images = get_domain_files(self.root_domain_a)
		self.domain_b_images = get_domain_files(self.root_domain_b)
		self.domain_a_len = len(self.domain_a_images)
		self.domain_b_len = len(self.domain_b_images)

		self.length_dataset = min(self.domain_a_len, self.domain_b_len)

	def __len__(self):
		return self.length_dataset

	def __getitem__(self, index):
		domain_a_path = self.domain_a_images[index % self.domain_a_len]
		domain_b_path = self.domain_b_images[index % self.domain_b_len]

		domain_a_img = Image.open(domain_a_path).convert("RGB")
		domain_b_img = Image.open(domain_b_path).convert("RGB")

		if self.transforms_a:
			domain_a_img, __ = self.transforms_a(domain_a_img)
		if self.transforms_b:
			domain_b_img, __ = self.transforms_b(domain_b_img)

		return domain_a_img, domain_b_img

def get_domain_files(dataset_path):
	if type(dataset_path) == str:
		content = os.listdir(dataset_path)
		list_images = []
		for c in content:
			full_path = os.path.join(dataset_path, c)
			if os.path.isdir(full_path):
				list_images = list_images + get_domain_files(full_path)
			elif os.path.splitext(full_path)[1].lower() in ['.jpeg', '.png', '.jpg']:
				list_images.append(full_path)
	else:
		list_images = dataset_path
	return list_images

def get_image(img_path, transforms):
	img = Image.open(img_path).convert("RGB")
	img, __ = transforms(img)
	return img