import os
import cv2
import json
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt

categories = ['ascaris', 'hookworm', 'large_egg', 'ov', 'tenia']

label_mapping = {
		'ascaris':1, 
		'hookworm':2, 
		'large_egg':3, 
		'ov':4, 
		'tenia':5,
		'trichuris':6
}

lbl2text = {
		1:'ascaris', 
		2:'hookworm', 
		3:'large_egg', 
		4:'ov', 
		5:'tenia',
		6:'trichuris'
}

# Should be in BGR
color_mapping = {
		1:(0, 0, 255),   # Red
		2:(66, 245, 66), # Green
		3:(245, 93, 66), # Blue
		4:(0, 236, 252), # Yellow
		5:(210, 0, 252), # Pink
		6:(100, 100, 100)
}

color_mapping_gt = {
		1:(0, 0, 0), # Red
		2:(0, 0, 0), # Green
		3:(0, 0, 0), # Blue
		4:(0, 0, 0), # Yellow
		5:(0, 0, 0),  # Dark Pink
		6:(0, 0, 0)  # Dark Pink
}

offset_mapping = {
		1:0, 
		2:40,
		3:80, 
		4:120,
		5:160,  
		6:180
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
		if torch.is_tensor(labels):
			lbl = labels[idx].item()
			color = cmapping[lbl]
		else:
			color = cmapping[0]
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

def load_settings(settings_file):
	with open(settings_file, 'r') as f:
		settings = json.load(f)
	return settings

def check_path(path):
	if not os.path.exists(path):
		folders = os.path.split(path)
		check_path(folders[0])
		os.mkdir(path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting metrics from a log file')
	parser.add_argument('-f','--log_file', help='Path of the log file from training', type=str)

	args = vars(parser.parse_args())

	log_file     = args['log_file'] 
	extract_metrics(log_file)
