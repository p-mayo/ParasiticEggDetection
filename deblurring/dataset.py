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
	def __init__(self, root_domain_a, root_domain_b, transform_a = None, transforms_b = None):
		self.root_domain_a = root_domain_a
		self.root_domain_b = root_domain_b
		self.transform_a = transform_a
		self.transform_b = transform_b

		self.domain_a_images = get_file_content(self.root_domain_a)
		self.domain_b_images = get_file_content(self.root_domain_b)
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

		if self.transform_a:
			domain_a_img, __ = self.transform_a(domain_a_img)
		if self.transform_b:
			domain_b_img, __ = self.transform_b(domain_b_img)

		return domain_a_img, domain_b_img