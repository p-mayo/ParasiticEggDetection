# Python file for the dataset loader
# for the CycleGAN implementation
 
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from torchvision import transforms as T
from torch.utils.data import Dataset

from CycleGAN.utils import get_transforms
from ParasiticEggDataset import get_data, get_targets
from references.transforms import UnNormalize

def get_file_content(file_name):
	with open(file_name) as f:
		lines = f.readlines()
	return lines

class CycleGAN_PED(Dataset):
	def __init__(self, root_domain_a, root_domain_b, annotations_path, 
					transforms_a, transforms_b,
					imsize_b = (512, 512), ratio_ab = 0.7):
		dataset_path_a = {
			'ascaris': os.path.join(root_domain_a, 'ascaris'),
			'hookworm': os.path.join(root_domain_a, 'hookworm'),
			'large_egg': os.path.join(root_domain_a, 'large_egg'),
			'ov': os.path.join(root_domain_a, 'ov'),
			'tenia': os.path.join(root_domain_a, 'tenia'),
			'trichuris': os.path.join(root_domain_a, 'trichuris')
		}

		dataset_path_b = {
			'ascaris': os.path.join(root_domain_b, 'ascaris'),
			'hookworm': os.path.join(root_domain_b, 'hookworm'),
			'large_egg': os.path.join(root_domain_b, 'large_egg'),
			'ov': os.path.join(root_domain_b, 'ov'),
			'tenia': os.path.join(root_domain_b, 'tenia'),
			'trichuris': os.path.join(root_domain_b, 'trichuris')
		}

		paths, targets = get_data(annotations_path, dataset_path_a, dataset = 's')
		data = [[p, l, b, a] for p, l, b, a in zip(paths, targets['labels'], targets['boxes'], targets['area'])]
		self.data_a = sorted(data, key = lambda data : data[1] + data[0].split(os.sep)[3:])

		paths, targets = get_data(annotations_path, dataset_path_b, dataset = 'c')
		data = [[p, l, b, a] for p, l, b, a in zip(paths, targets['labels'], targets['boxes'], targets['area'])]
		self.data_b = sorted(data, key = lambda data : data[1] + data[0].split(os.sep)[3:])
		#for a, b in zip(self.data_a, self.data_b):
		#	print(a[0], a[2], a[3])
		#	print(b[0], b[2], b[3])
		#	print()
		
		# For Samsung
		self.data_a_size = (3264, 2448)
		self.data_b_size = (1920, 1080)
		self.imsize_b = imsize_b
		self.imsize_a = (int(imsize_b[0]/ratio_ab), int(imsize_b[1]/ratio_ab))

		self.new_a_size = (self.data_a_size[0] - self.imsize_a[0], self.data_a_size[0] - self.imsize_a[0])
		self.new_b_size = (self.data_b_size[0] - self.imsize_b[0], self.data_b_size[0] - self.imsize_b[0])
		self.len_a = len(self.data_a)
		self.len_b = len(self.data_b)
		
		self.transforms_a = transforms_a
		self.transforms_b = transforms_b
		self.length_dataset = min(self.len_a, self.len_b)

	def __len__(self):
		return self.length_dataset

	def __getitem__(self, index):
		path_a = self.data_a[index % self.len_a][0]
		path_b = self.data_b[index % self.len_b][0]

		img_a = Image.open(path_a).convert("RGB")
		img_b = Image.open(path_b).convert("RGB")
		#print(self.data_a[index])
		#print(self.data_b[index])
		box_a = self.data_a[index % self.len_a][2][0]
		box_b = self.data_b[index % self.len_b][2][0]	
		angle = np.random.randint(-170,170)
		img_a = img_a.rotate(angle, center = get_center(box_a))
		angle = np.random.randint(-170,170)
		img_b = img_b.rotate(angle, center = get_center(box_b))
		crop_a = region_to_crop(self.data_a_size, box_a, self.imsize_a)
		crop_b = region_to_crop(self.data_b_size, box_b, self.imsize_b)
		cropped_a = img_a.crop(crop_a).resize((512, 512))
		cropped_b = img_b.crop(crop_b).resize((512, 512))
		if self.transforms_a:
			cropped_a, __ = self.transforms_a(cropped_a)
		if self.transforms_b:
			cropped_b, __ = self.transforms_a(cropped_b)
		return cropped_a, cropped_b

def get_center(target_box):
	center_x = int(0.5*(target_box[0] + target_box[2]))
	center_y = int(0.5*(target_box[1] + target_box[3]))
	return (center_x, center_y)

def region_to_crop(im_size, target_box, crop_size):
	#center_x = int(0.5*(target_box[0] + target_box[2]))
	#center_y = int(0.5*(target_box[1] + target_box[3]))
	(center_x, center_y) = get_center(target_box)
	#center_x = target_box[0]
	#center_y = target_box[3]
	min_x = max(center_x - crop_size[0], 0)
	max_x = center_x if (center_x + crop_size[0]) < im_size[0] else im_size[0] - crop_size[0]
	max_y = min(center_y + crop_size[1], im_size[1])
	min_y = center_y if (center_y - crop_size[1]) > 0 else crop_size[1]
	#print((center_x, center_y), (min_x, max_x, min_y, max_y), target_box)
	left = np.random.randint(min_x, max_x)
	top = np.random.randint(min_y, max_y)
	
	return left, top - crop_size[1], left + crop_size[0], top


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

if __name__ == '__main__':
	root_a = r'C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung'
	root_b = r'C:\Users\pm15334\ra\ParasiticEggDetection\dataset_canon'
	annotations_path = r"C:\Users\pm15334\ra\ParasiticEggDetection\dataset_samsung\Annotations.json"
	ds = CycleGAN_PED(root_a, root_b, annotations_path, 
		transforms_a = get_transforms("b", False),
		transforms_b = get_transforms("b", False))
	unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	for i in range(15):
		img_a, img_b = ds[i]
		fig, axs = plt.subplots(1,2)
		axs[0].imshow(T.ToPILImage()(unnorm(img_a)))
		axs[1].imshow(T.ToPILImage()(unnorm(img_b)))
	#crop_a = region_to_crop(ds.data_a_size, box_a[0], ds.imsize_a)
	#crop_b = region_to_crop(ds.data_b_size, box_b[0], ds.imsize_b)
	#cropped_a = img_a.crop(crop_a)
	#cropped_b = img_b.crop(crop_b)

	#draw_a = ImageDraw.Draw(img_a)
	#draw_b = ImageDraw.Draw(img_b)

	#draw_a.rectangle(box_a[0], outline = "red", width = 10)
	#draw_b.rectangle(box_b[0], outline = "red", width = 10)

	#draw_a.rectangle(crop_a, outline = "blue", width = 10)
	#draw_b.rectangle(crop_b, outline = "blue", width = 10)

	#print(img_a, img_a.size, ds.imsize_a)
	#print(img_b, img_b.size, ds.imsize_b)
	#print(box_a, crop_a)
	#print(box_b, crop_b)
	#print(cropped_a)
	#print(cropped_b)
	print(img_a.size)
	print(img_b.size)
	#axs[0].imshow(img_a)
	#axs[1].imshow(img_b)
	#axs[0,0].imshow(img_a)
	#axs[0,1].imshow(img_b)
	#axs[1,0].imshow(cropped_a.resize((512, 512)))
	#axs[1,1].imshow(cropped_b.resize((512, 512)))
	#axs[0].imshow(draw_a)
	#axs[1].imshow(draw_b)
	plt.show()


