# Python script to detect parasitic eggs from microscopic images

import os
import cv2
import argparse

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from references import utils
from references import transforms as T
from references.engine import train_one_epoch, evaluate, keep_outputs
from sklearn.model_selection import StratifiedKFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ParasiticEggDataset import ParasiticEggDataset, get_data
from utils import load_settings, check_path, label_mapping, draw_boxes

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
		saturation = (0.5, 1.5) if "saturation" in train else (0,0)
		contrast = (0.5, 1.5) if "contrast" in train else (0,0)
		hue = (-0.05, 0.05) if "hue" in train else (0,0)
		brightness = (0.875, 1.125) if "brightness" in train else (0,0) 
		transforms.append(T.RandomPhotometricDistort(contrast, saturation, hue, brightness))
		if "rotation" in train: 
			transforms.append(T.RandomRotation())
		if "hflip" in train:
			transforms.append(T.RandomHorizontalFlip())
		if  "vflip" in train:
			transforms.append(T.RandomVerticalFlip())
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

def valid_value(settings, item, default):
	if (item in settings.keys()) and settings[item]:
		return settings[item]
	return default

def train(settings):
	annotations_path = settings['annotations_path']
	root_path = settings['root_path']
	num_epochs = settings['num_epochs']
	batch_size = settings['batch_size']
	seed = valid_value(settings, 'seed', 1)
	output_path = valid_value(settings, 'output_path', '')
	kfolds = valid_value(settings, 'kfolds', 5)
	folds = valid_value(settings, 'folds', [i for i in range(1, kfolds + 1)])

	# root_path = /content/drive/MyDrive/ParasiticEggDataset
	dataset_path = {
		'ascaris': os.path.join(root_path, 'ascaris'),
		'hookworm': os.path.join(root_path, 'hookworm'),
		'large_egg': os.path.join(root_path, 'large_egg'),
		'ov': os.path.join(root_path, 'ov'),
		'tenia': os.path.join(root_path, 'tenia')
	}

	paths, targets = get_data(annotations_path, dataset_path)
	labels = get_labels(targets)

	skf = StratifiedKFold(n_splits=kfolds)
	skf.get_n_splits(paths, labels)

	for fold, (train_idx, test_idx) in enumerate(skf.split(paths,labels),1):
		if fold in folds:
			fold_path = os.path.join(output_path, 'fold_%d' % fold)
			check_path(fold_path)
			print('---------------------------------------')
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
					torch.save(model, os.path.join(fold_path, 'fold_%d_epoch_%d.pkl' % (fold, epoch)))
	return model

def test(settings):
	annotations_path = settings['annotations_path']
	root_path = settings['root_path']
	seed = valid_value(settings, 'seed', 1)
	output_path = valid_value(settings, 'output_path', '')
	kfolds = valid_value(settings, 'kfolds', -1)
	folds = valid_value(settings, 'folds', [-1])
	model_path = settings['model_path']
	evaluate_model = valid_value(settings, 'evaluate_model', False)
	idxs = valid_value(settings, 'idxs', -1)
	# root_path = /content/drive/MyDrive/ParasiticEggDataset
	dataset_path = {
		'ascaris': os.path.join(root_path, 'ascaris'),
		'hookworm': os.path.join(root_path, 'hookworm'),
		'large_egg': os.path.join(root_path, 'large_egg'),
		'ov': os.path.join(root_path, 'ov'),
		'tenia': os.path.join(root_path, 'tenia')
	}

	paths, targets = get_data(annotations_path, dataset_path)
	labels = get_labels(targets)

	skf = StratifiedKFold(n_splits=kfolds)
	skf.get_n_splits(paths, labels)

	model = torch.load(model_path)
	model.eval()
	with torch.no_grad():
		for fold, (train_idx, test_idx) in enumerate(skf.split(paths,labels),1):
			if fold in folds:
				fold_path = os.path.join(output_path, 'fold_%d' % fold)
				print('---------------------------------------')
				print('TESTING FOLD ', fold)
				print('---------------------------------------')
				torch.manual_seed(seed)
				eggs_dataset_test = ParasiticEggDataset(np.array(paths)[test_idx].tolist(), get_targets(targets, test_idx), get_transform(train=False), label_mapping=label_mapping)
				device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
				
				if evaluate_model:
					data_loader_test = torch.utils.data.DataLoader(
				                eggs_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1,
				                collate_fn=utils.collate_fn)
					evaluate(model, data_loader_test, device=device)
				if idxs == -1:
					idxs = range(len(eggs_dataset_test))
				else:
					idxs = [np.random.randint(0,len(eggs_dataset_test))]
				for idx in idxs:
					prediction = model([eggs_dataset_test[idx][0].to('cuda')])
					boxes = prediction[0]['boxes']
					scores = prediction[0]['scores']
					labels = prediction[0]['labels']
					keep = torchvision.ops.nms(boxes, scores, 0.5)
					new_outputs = keep_outputs(prediction[0], keep)

					img = draw_boxes(eggs_dataset_test[idx][0].permute(1,2,0).numpy().copy(), boxes, labels,scores)
					img = draw_boxes(img, eggs_dataset_test[idx][1]['boxes'],eggs_dataset_test[idx][1]['labels'])

					fname = os.path.join(output_path, 'test_%d.%d.png' % (fold, idx))
					fig, axs = plt.subplots(figsize=(20,20))
					axs.imshow(img)
					plt.savefig(fname, transparent=True, bbox_inches='tight')
					plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training and testing model for Parasitic Egg Detection')
	parser.add_argument('-f','--settings_file', help='Path of the JSON file containing the training settings', type=str)
	parser.add_argument('-m','--mode', help='Mode to run the program (train or test)', type=str, default='train')
	args = vars(parser.parse_args())

	settings_file     = args['settings_file'] 
	mode              = args['mode']

	settings = load_settings(settings_file)
	#print(settings)

	if mode.lower() == 'train':
		train(settings)
	elif mode.lower() == 'test':
		test(settings)
	else:
		print('Mode not recognised')

