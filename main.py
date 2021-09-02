# Python script to detect parasitic eggs from microscopic images

import os
import cv2
import json
import torch
import torchvision
import numpy as np

from references import utils
from references import transforms as T
from references.engine import train_one_epoch, evaluate
from sklearn.model_selection import StratifiedKFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ParasiticEggDataset import ParasiticEggDataset

def get_data(annotations_path, root_path):
	with open(annotations_path, 'r') as f:
		annotations = json.load(f)
	paths = []
	targets = {'boxes':[], 'labels':[], 'area':[], 'iscrowd':[]}
	for item in annotations:
		if item['External ID'].split('.')[0][-3] == 's':
			temp_label = []
			temp_bbox = []
			temp_area = []
			temp_iscrowd = []
			for label in item['Label']['objects']:
				temp_label.append(label['value'])
				xmin = label['bbox']['left']          # This is okay
				xmax = xmin + label['bbox']['width']  # This is fine
				ymin = label['bbox']['top']           # It says top but is actually bottom
				ymax = ymin + label['bbox']['height'] 
				temp_bbox.append([xmin, ymin, xmax, ymax])
				temp_area.append(label['bbox']['width'] * label['bbox']['height'])
				#temp_bbox = torch.as_tensor(temp_bbox, dtype=torch.float32)
			img_path = os.path.join(root_path[temp_label[0]], item['External ID'])
			if os.path.exists(img_path):
				paths.append(img_path)
				targets['labels'].append(temp_label)
				targets['boxes'].append(temp_bbox)
				targets['area'].append(temp_area)
				targets['iscrowd'].append([0.]*len(temp_bbox))
	return paths, targets

def draw_boxes(image, boxes):
	for box in boxes:
		image = cv2.rectangle(
			image,
			(int(box[0]), int(box[3])), # Top-left
			(int(box[2]), int(box[1])), # Bottom-right
			(255, 0, 0), 3
		)
		if type(image) == cv2.UMat:
			image = image.get()
	return image

def get_model(num_classes):
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model

def get_transform(train):
	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
	transforms.append(T.Normalize())
	if train:
		# during training, randomly flip the training images
		# and ground-truth for data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)

def get_labels(targets):
	labels = []
	for t in targets['labels']:
		labels.append(t[0])
	return labels

def get_targets(targets, idxs):
	new_targets = {'labels':[], 'area':[], 'boxes':[], 'iscrowd':[]}
	for idx in idxs:
		new_targets['labels'].append(targets['labels'][idx])
		new_targets['boxes'].append(targets['boxes'][idx])
		new_targets['area'].append(targets['area'][idx])
		new_targets['iscrowd'].append(targets['iscrowd'][idx])
	return new_targets

def main(annotations_path, root_path, num_epochs, batch_size, seed=1, output_path=None, kfolds=5):
	# root_path = /content/drive/MyDrive/ParasiticEggDataset
	dataset_path = {
		'ascaris': os.path.join(root_path, 'ascaris'),
		'hookworm': os.path.join(root_path, 'hookworm'),
		'large_egg': os.path.join(root_path, 'large_egg'),
		'ov': os.path.join(root_path, 'ov'),
		'tenia': os.path.join(root_path, 'tenia')
	}

	label_mapping = {
		'ascaris':1, 
		'hookworm':2, 
		'large_egg':3, 
		'ov':4, 
		'tenia':5
	}

	paths, targets = get_data(annotations_path, dataset_path)
	labels = get_labels(targets)
	skf = StratifiedKFold(n_splits=kfolds)
	skf.get_n_splits(paths, labels)

	for fold, (train_idx, test_idx) in enumerate(skf.split(paths,labels),1):
		print('\n\n---------------------------------------')
		print('STARTING FOLD ', fold)
		print('---------------------------------------')
		torch.manual_seed(seed)
		eggs_dataset = ParasiticEggDataset(np.array(paths)[train_idx].tolist(), get_targets(targets, train_idx), get_transform(train=False), label_mapping=label_mapping)
		eggs_dataset_test = ParasiticEggDataset(np.array(paths)[test_idx].tolist(), get_targets(targets, test_idx), get_transform(train=False), label_mapping=label_mapping)
		#eggs_dataset = torch.utils.data.Subset(eggs_dataset, train_idx)
		#eggs_dataset_test = torch.utils.data.Subset(eggs_dataset_test, test_idx)
		# define training and validation data loaders
		data_loader = torch.utils.data.DataLoader(
		                eggs_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
		                collate_fn=utils.collate_fn)

		data_loader_test = torch.utils.data.DataLoader(
		                eggs_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1,
		                collate_fn=utils.collate_fn)

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		num_classes = 6
		model = get_model(num_classes)
		model.to(device)
		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.SGD(params, lr=0.005,
		                            momentum=0.9, weight_decay=0.0005)

		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
														step_size=3,
														gamma=0.1)

		for epoch in range(num_epochs):
			# train for one epoch, printing every 10 iterations
			train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
			# update the learning rate
			lr_scheduler.step()
			# evaluate on the test dataset
			evaluate(model, data_loader_test, device=device)

		if output_path:
			torch.save(model, os.path.join(output_path, 'model_fold_%d' % fold))


