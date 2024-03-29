import os
import cv2
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

class Settings:
	def __init__(self, mode, settings):
		self.mode = mode.lower()
		self.update_settings(settings)

	def update_settings(self, settings):
		self.root_path = settings['root_path']
		self.output_path = valid_value(settings, 'output_path', '')
		self.remove_scores = valid_value(settings, 'remove_scores', 0.5)
		self.colour = valid_value(settings, 'colour', False)
		self.transforms = valid_value(settings, 'transforms', [])
		self.use_gpu = valid_value(settings, 'use_gpu', True)
		self.device = torch.device('cuda') if (torch.cuda.is_available() and self.use_gpu) else torch.device('cpu')
		if self.colour:
			self.colour_mean = valid_value(settings, 'colour_mean', [0.485, 0.456, 0.406])
			self.colour_std = valid_value(settings, 'colour_std', [0.229, 0.224, 0.225])
		else:
			self.colour_mean = valid_value(settings, 'colour_mean', 0.5)
			self.colour_std = valid_value(settings, 'colour_std', 0.25)
		if self.mode in ['train', 'test']:
			self.balance_dataset = False
			self.annotations_path = settings['annotations_path']
			self.seed = valid_value(settings, 'seed', 1)
			self.num_epochs = settings['num_epochs']
			self.batch_size = settings['batch_size']
			self.kfolds = valid_value(settings, 'kfolds', 5)
			self.folds = valid_value(settings, 'folds', [i for i in range(1, self.kfolds + 1)])
			self.augment_test = valid_value(settings, 'augment_test', False)
			if self.augment_test:
				self.augment_test = transforms
			if self.mode == 'test':
				self.model_path = settings['model_path']
				self.evaluate_model = valid_value(settings, 'evaluate_model', False)
				self.show_predictions = valid_value(settings, 'show_predictions', False)
				self.idxs = valid_value(settings, 'idxs', -1)
		else:
			self.model_path = settings['model_path']



categories = ['ascaris', 'hookworm', 'large_egg', 'ov', 'tenia']

label_mapping = {
		'ascaris':1, 
		'hookworm':2, 
		#'large_egg':3, 
		'ov':3, 
		'tenia':4,
		'trichuris':5
}

lbl2text = {
		1:'ascaris', 
		2:'hookworm', 
		#3:'large_egg', 
		3:'ov', 
		4:'tenia',
		5:'trichuris'
}

# Should be in BGR
color_mapping = {
		1:(0, 0, 255),   # Red
		2:(66, 245, 66), # Green
		3:(245, 93, 66), # Blue
		4:(0, 236, 252), # Yellow
		5:(210, 0, 252), # Pink
		#6:(100, 100, 100)
}

color_mapping_gt = {
		1:(0, 0, 0), # Red
		2:(0, 0, 0), # Green
		3:(0, 0, 0), # Blue
		4:(0, 0, 0), # Yellow
		5:(0, 0, 0),  # Dark Pink
		#6:(0, 0, 0)  # Dark Pink
}

offset_mapping = {
		1:0, 
		2:40,
		3:80, 
		4:120,
		5:160,  
		#6:180
}


# Window name in which image is displayed
window_name = 'Image'
  
# font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

def extract_metrics(log_file):
	with open(log_file, 'r') as f:
		lines = f.readlines()

	summary = {'fold':[], 'epoch':[], 'iou':[], 'area':[], 'maxdets':[], 'metric':[], 'all':[], 
				'ascaris':[], 'hookworm':[], 'large_egg':[], 'ov':[], 'tenia':[]}
	category = [0,0,0,0,0]
	for line in lines:
		if "FOLD" in line:
			fold = int(line.split()[-1].strip())
		elif "epoch" in line.lower():
			epoch = int(line.split()[1][1:-1])
		elif "category" in line:
			items = line.split(':')
			category[int(items[1].strip())] = float(items[-1].strip())
		elif "Average" in line:
			if "Precision" in line:
				metric = 'precision'
			elif "Recall" in line:
				metric = 'recall'
			else:
				continue
			items = line.split()
			iou=items[4][4:]
			area=items[7]
			maxdets = int(items[9].split('=')[-1].strip())
			all_classes = float(items[-1].strip())
			if all_classes == -1:
				category[0] = -1
				category[1] = -1
				category[2] = -1
				category[3] = -1
				category[4] = -1
			summary['fold'].append(fold)
			summary['epoch'].append(epoch)
			summary['iou'].append(iou)
			summary['area'].append(area)
			summary['maxdets'].append(maxdets)
			summary['metric'].append(metric)
			summary['all'].append(all_classes)
			summary['ascaris'].append(category[0])
			summary['hookworm'].append(category[1])
			summary['large_egg'].append(category[2])
			summary['ov'].append(category[3])
			summary['tenia'].append(category[4])
	return pd.DataFrame.from_dict(summary)

def plot_metrics(df, fold, area = 'all', iou = '0.50:0.95', maxdets = 100, metric='precision', column ='all', ax = None):
	ax = df[(df.fold==fold) & (df.area == area) & (df.maxdets == maxdets) & (df.iou == iou) & (df.metric == metric)].plot(x='epoch', y=column, ax=ax)
	plot_title = metric.title() + ' for fold ' + str(fold) + '\nArea = ' + area + ', IoU = ' + iou + ', MaxDets = ' + str(maxdets)
	ax.set_title(plot_title)
	ax.set_xticks([i for i in range(0, max(df[df.fold == fold].epoch)+1)])
	ax.set_ylim([0,1])
	#plt.show()
	return ax

def plot_several_classes(df, fold, area = 'all', iou = '0.50:0.95', maxdets = 100, metric='precision', all=True):
	ax = None
	for cat in categories:
		ax = plot_metrics(df, fold, area, iou, maxdets, metric, cat, ax)
	
	if all:
		ax = plot_metrics(df, fold, area, iou, maxdets, metric, 'all', ax)
	plt.show()

def draw_boxes(image, boxes, labels=None, scores=None):
	if torch.is_tensor(scores): # If there are scores, then it is predictions, otherwise is GT
		cmapping = color_mapping
	else:
		cmapping = color_mapping_gt
	for idx, box in enumerate(boxes):
		#print(box)
		if torch.is_tensor(labels):
			lbl = labels[idx].item()
			color = cmapping[lbl]
		else:
			color = cmapping[0]
		h, w, _ = image.shape
		image = cv2.rectangle(
			image,
			(int(box[0]), int(box[3])), # Top-left
			(int(box[2]), int(box[1])), # Bottom-right
			color, 3
		)
		if torch.is_tensor(scores):
			text = "%s (%0.2f)" % (lbl2text[lbl], scores[idx].item()) 
			x = int(box[2])
			y = int(box[1]) + offset_mapping[lbl]
		else:
			text = lbl2text[lbl]
			x = int(box[0])
			y = int(box[3]) + 20
		image = cv2.putText(image, text, (x, y), 
						font, fontScale, color, thickness, cv2.LINE_AA)
		if type(image) == cv2.UMat:
			image = image.get()
	return image

def load_settings(settings_file, mode = 'train'):
	with open(settings_file, 'r') as f:
		settings = json.load(f)
	return settings

def check_path(path):
	if not os.path.exists(path):
		folders = os.path.split(path)
		check_path(folders[0])
		os.mkdir(path)

def get_str(list_trans, concatenator=" "):
	return concatenator.join(str(x).replace("\n","") for x in list_trans)

def write_csv(csv_path, data, mode='a'):
	file = open(csv_path, mode)
	if type(data) == str:
		data = [data]
	for row in data:
		file.write(row + "\n")
	file.close()

def write_metrics(output_path, results):
	if os.path.isfile(output_path) == False:
		write_csv(output_path, get_str(results.keys(),","))
	write_csv(output_path, get_str(results.values(),","))

def compare_images(original, augmented, img_path):
	fig, axs = plt.subplots(1,2)
	img_o = draw_boxes(original[0].permute(1,2,0).numpy().copy(), original[1]["boxes"], 
			original[1]["labels"])
	img_a = draw_boxes(augmented[0].permute(1,2,0).numpy().copy(), augmented[1]["boxes"], 
			augmented[1]["labels"])
	axs[0].imshow(img_o)
	axs[1].imshow(img_a)
	image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
	#axs[2].imshow(image)
	plt.savefig("comparing.png", transparent=True, bbox_inches='tight')
	plt.close()

def valid_value(settings, item, default):
	if (item in settings.keys()) and settings[item]:
		return settings[item]
	return default


def check_valid_images(imgs):
	return [img for img in imgs if os.path.splitext(img)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting metrics from a log file')
	parser.add_argument('-f','--log_file', help='Path of the log file from training', type=str)

	args = vars(parser.parse_args())

	log_file     = args['log_file'] 
	extract_metrics(log_file)

