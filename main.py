# Python script to detect parasitic eggs from microscopic images

import os
import cv2
import json
import argparse

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.utils import save_image

from references import utils
from references import transforms as T
from references.transforms import UnNormalize
from references.engine import train_one_epoch, evaluate, keep_outputs
from sklearn.model_selection import StratifiedKFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ParasiticEggDataset import ParasiticEggDataset, get_data, get_targets, get_labels
from utils import load_settings, check_path, label_mapping, draw_boxes, write_metrics, lbl2text
from utils import Settings

unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def get_model(num_classes, backbone = "resnet50fpn"):
	if backbone == "resnet50fpn":
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	if backbone == "vgg16":
		model = torchvision.models.vgg16(pretrained=True).features
		model.out_channels = 512
		anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7,
		                                                        sampling_ratio=2)
		model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone, 
		                                num_classes, rpn_anchor_generator=anchor_generator,
		                                box_roi_pool=roi_pooler)
	return model

def get_transform(train, colour_mean=[0.485, 0.456, 0.406], 
			colour_std=[0.229, 0.224, 0.225], p=0.5, normalise = True):
	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
	if train:
		# during training, randomly flip the training images
		# and ground-truth for data augmentation
		if "crop" in train:
			print("... Random Crop for CycleGAN")
			transforms.append(T.RandomCrop())
		saturation = (0., 1.5) if "saturation" in train else (0,0)
		contrast = (0., 1.5) if "contrast" in train else (0,0)
		hue = (-0.1, 0.1) if "hue" in train else (0,0)
		brightness = (0.875, 1.125) if "brightness" in train else (0,0) 
		if (saturation != (0,0)) or (contrast != (0,0)) or (hue != (0,0)) or (brightness != (0,0)):
			print("... RandomPhotometricDistort for Data Augmentation")
			transforms.append(T.RandomPhotometricDistort(contrast, saturation, hue, brightness))
		if "rotation" in train: 
			print("... Random Rotation for Data Augmentation")
			transforms.append(T.RandomRotation())
		if "hflip" in train:
			print("... Random Horizontal Flip for Data Augmentation")
			transforms.append(T.RandomVerticalFlip())
		if  "vflip" in train:
			print("... Random Vertical Flip for Data Augmentation")
			transforms.append(T.RandomVerticalFlip())
		if  "blur" in train:
			print("... Random Blur for Data Augmentation")
			transforms.append(T.MotionBlur(p))
	if normalise:
		transforms.append(T.Normalize(mean = colour_mean, std = colour_std))
		#transforms.append(T.Normalize(mean = 0.5, std = 0.225))
	return T.Compose(transforms)

def get_data_for_split(root_path, balance_dataset=False):
	if type(root_path) == str:
		root_path = [root_path]
	all_paths = []
	all_labels = []
	all_targets = {'boxes':[], 'labels':[], 'area':[], 'iscrowd':[]}
	for rp in root_path:
		annotations_path = os.path.join(rp, "Annotations.json")
		dataset_path = {
			'ascaris': os.path.join(rp, 'ascaris'),
			'hookworm': os.path.join(rp, 'hookworm'),
			#'large_egg': os.path.join(rp, 'large_egg'),
			'ov': os.path.join(rp, 'ov'),
			'tenia': os.path.join(rp, 'tenia'),
			'trichuris': os.path.join(rp, 'trichuris')
		}

		print("... Loading annotations from ", annotations_path)	
		paths, targets = get_data(annotations_path, dataset_path, rp.split('_')[-1])
		labels = get_labels(targets)
		all_paths += paths
		all_labels += labels
		all_targets['boxes'] += targets['boxes']
		all_targets['labels'] += targets['labels']
		all_targets['area'] += targets['area']
		all_targets['iscrowd'] += targets['iscrowd']
	classes = set(all_labels)
	occurrences = {}
	for c in classes:
		occurrences[c] = all_labels.count(c)
		print(c, ': ', occurrences[c])
	if balance_dataset:
		print("... Balancing dataset")
		max_occurrences = max(occurrences.values())
		for k, v in occurrences.items():
			diff = max_occurrences - occurrences[k]
			idxs = [i for i, lbl in enumerate(all_labels) if lbl == k]
			idxs = np.random.choice(idxs, diff)
			for d in idxs:
				all_paths += [all_paths[idx]]
				all_labels += [all_labels[idx]]
				all_targets += [all_targets[idx]]


	return all_paths, all_targets, all_labels


def log_metrics(output_path, metrics, fold, loss='Test', epoch='Test'):
	results = {}
	results['fold'] = fold
	results['epoch'] = epoch
	results['loss'] = loss
	for i in range(metrics.shape[0]):
		clss = lbl2text[int(metrics[i,0])]
		results['TP (%s)' % clss] = metrics[i,1]
		results['FP (%s)' % clss] = metrics[i,2]
		results['FN (%s)' % clss] = metrics[i,3]
		results['Precision (%s)' % clss] = metrics[i,4]
		results['Recall (%s)' % clss] = metrics[i,5]
		results['F1-Score (%s)' % clss] = metrics[i,6]
	metrics_path = os.path.join(output_path, "metrics.csv")
	write_metrics(metrics_path, results)

def train(settings):
	paths, targets, labels = get_data_for_split(settings.root_path, settings.balance_dataset)
	skf = StratifiedKFold(n_splits=settings.kfolds, shuffle=True, random_state=settings.seed)
	skf.get_n_splits(paths, labels)
	num_classes = len(set(labels)) + 1
	print("... Classes in dataset: ", num_classes, set(labels), "(+1 for background)")		
	for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels),1):
		if fold in settings.folds:
			if settings.output_path:
				fold_path = os.path.join(settings.output_path, 'fold_%d' % fold)
				check_path(fold_path)
			print('---------------------------------------')
			print('STARTING FOLD ', fold)
			print('---------------------------------------')
			torch.manual_seed(settings.seed)
			eggs_dataset = ParasiticEggDataset(np.array(paths)[train_idx].tolist(), 
												get_targets(targets, train_idx), 
												get_transform(settings.transforms), 
												label_mapping=label_mapping, 
												colour=settings.colour)
			eggs_dataset_test = ParasiticEggDataset(np.array(paths)[test_idx].tolist(), 
												get_targets(targets, test_idx), 
												get_transform(settings.augment_test), 
												label_mapping=label_mapping, 
												colour=settings.colour)
			# define training and validation data loaders
			data_loader = torch.utils.data.DataLoader(
			                eggs_dataset, batch_size=settings.batch_size, 
			                shuffle=True, num_workers=1,
			                collate_fn=utils.collate_fn)

			data_loader_test = torch.utils.data.DataLoader(
			                eggs_dataset_test, batch_size=settings.batch_size, 
			                shuffle=False, num_workers=1,
			                collate_fn=utils.collate_fn)

			model = get_model(num_classes)
			model.to(settings.device)
			params = [p for p in model.parameters() if p.requires_grad]
			optimizer = torch.optim.SGD(params, lr=0.005,
			                            momentum=0.9, weight_decay=0.0005)

			lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
															step_size=3,
															gamma=0.1)

			for epoch in range(settings.num_epochs):
				# train for one epoch, printing every 10 iterations
				metric_logger = train_one_epoch(model, optimizer, data_loader, 
									settings.device, epoch, print_freq=10)
				# update the learning rate
				lr_scheduler.step()
				# evaluate on the test dataset
				metrics = evaluate(model, data_loader_test, device=settings.device, 
									remove_scores=settings.remove_scores)
				# coco_evaluator.stats
				if settings.output_path:
					log_metrics(settings.output_path, metrics, fold, metric_logger.meters['loss'], epoch)
					torch.save(model, os.path.join(fold_path, 'fold_%d_epoch_%d.pkl' % (fold, epoch)))
	return model

def test(settings):
	paths, targets, labels = get_data_for_split(settings.root_path, settings.balance_dataset)
	skf = StratifiedKFold(n_splits=settings.kfolds)#, shuffle=True, random_state=settings.seed)
	skf.get_n_splits(paths, labels)
	with torch.no_grad():
		for fold, (train_idx, test_idx) in enumerate(skf.split(paths,labels),1):
			if fold in settings.folds:
				fold_path = os.path.join(settings.output_path, 'fold_%d' % fold)
				#model = torch.load(settings.model_path)
				model = torch.load(os.path.join(settings.model_path, 'fold_%d' % fold, 'fold_%d_epoch_49.pkl' % fold))
				model.eval()
				print("... Model ", settings.model_path, "loaded!!!!")
				print('---------------------------------------')
				print('TESTING FOLD ', fold)
				print('---------------------------------------')
				torch.manual_seed(settings.seed)
				eggs_dataset_test = ParasiticEggDataset(np.array(paths)[test_idx].tolist(), 
					get_targets(targets, test_idx), get_transform(train=settings.transforms), 
					label_mapping=label_mapping,colour=settings.colour)
				if settings.evaluate_model:
					data_loader_test = torch.utils.data.DataLoader(
				                eggs_dataset_test, batch_size=settings.batch_size, 
				                shuffle=False, num_workers=1,
				                collate_fn=utils.collate_fn)
					metrics = evaluate(model, data_loader_test, device=settings.device)
					if settings.output_path:
						log_metrics(settings.output_path, metrics, fold)
				if settings.show_predictions:
					if settings.idxs == -1:
						idxs = range(len(eggs_dataset_test))
					else:
						idxs = [np.random.randint(0,len(eggs_dataset_test), idxs)]
					for idx in idxs:
						filename = 'test_%d.%d_nms.png' % (fold, idx)
						predict_image(model, eggs_dataset_test[idx], settings, filename)

def test_directory(settings):
	if os.path.isdir(settings.root_path):
		imgs = os.listdir(settings.root_path)
		imgs = [os.path.join(settings.root_path, img) for img in imgs]
	else:
		imgs = [settings.root_path]
	model = torch.load(settings.model_path)
	model.eval()
	with torch.no_grad():
		eggs_dataset_test = ParasiticEggDataset(imgs, None, 
			get_transform(train=settings.transforms), colour=settings.colour)
		for i in range(len(imgs)):
			print("Processing image: ", imgs[i])
			filename = '%s_pred.png' % imgs[i].split(os.path.sep)[-1].split('.')[0]
			predict_image(model, eggs_dataset_test[i], settings, filename)

def predict_image(model, items, settings, filename, gan_model = None):
	img, target = items
	prediction = model([img.to(settings.device)])
	boxes = prediction[0]['boxes']
	scores = prediction[0]['scores']
	labels = prediction[0]['labels']
	keep = torchvision.ops.nms(boxes, scores, 0.5)
	new_outputs = keep_outputs(prediction[0], keep, 
		remove_scores = settings.remove_scores)
	img = draw_boxes(unnorm(img).permute(1,2,0).numpy().copy(), 
		new_outputs['boxes'], new_outputs['labels'],new_outputs['scores'])
	if target:
		img = draw_boxes(img, target['boxes'], target['labels'])
	fname = os.path.join(settings.output_path, filename)
	fig, axs = plt.subplots(figsize=(20,20))
	axs.imshow(img)
	axs.set_xticks([])
	axs.set_yticks([])
	plt.savefig(fname, transparent=True, bbox_inches='tight')
	plt.close()

def to_coco(bbox):
	x = bbox[0].item()
	y = bbox[1].item()
	width = bbox[2].item() - bbox[0].item()
	height = bbox[3].item() - bbox[1].item()
	return [x, x, width, height]

def test_challenge(settings):
	lbl_mapping = [0,4,7,9,10]
	results = {'annotations':[]}
	if os.path.isdir(settings.root_path):
		imgs = os.listdir(settings.root_path)
		imgs = [os.path.join(settings.root_path, img) for img in imgs]
	else:
		imgs = [settings.root_path]
	model = torch.load(settings.model_path)
	model.eval()
	with torch.no_grad():
		eggs_dataset_test = ParasiticEggDataset(imgs, None, 
			get_transform(train=settings.transforms), colour=settings.colour)
		egg_id = 0
		for i in range(len(imgs)):
			filename = imgs[i]
			if imgs[i].endswith('.jpg'):
				print("Processing image: ", filename)
				img, target = eggs_dataset_test[i]
				prediction = model([img.to(settings.device)])
				boxes = prediction[0]['boxes']
				scores = prediction[0]['scores']
				labels = prediction[0]['labels']
				keep = torchvision.ops.nms(boxes, scores, 0.5)
				new_outputs = keep_outputs(prediction[0], keep, 
					remove_scores = settings.remove_scores)
				for it in range(len(new_outputs['boxes'])):
					results['annotations'].append({
						'id':egg_id,
						'file_name':filename.split(os.sep)[-1],
						'categoy_id':lbl_mapping[new_outputs['labels'][it]-1],
						'bbox':[to_coco(new_outputs['boxes'][it])]
						})
					#print(to_coco(new_outputs['boxes'][it]))
					egg_id += 1
	with open('test_results.json', 'w') as outfile:
		json.dump(results, outfile)


def test_gan(settings):
	if os.path.isdir(settings.root_path):
		imgs = os.listdir(settings.root_path)
		imgs = [os.path.join(settings.root_path, img) for img in imgs]
	else:
		imgs = [settings.root_path]
	model = torch.load(settings.model_path)
	model.eval()
	with torch.no_grad():
		eggs_dataset_test = ParasiticEggDataset(imgs, None, 
			get_transform(train=settings.transforms, p=1.), colour=settings.colour)
		for i in range(len(imgs)):
			print("Processing image: ", imgs[i])
			filename = '%s_pred.png' % imgs[i].split(os.path.sep)[-1].split('.')[0]
			predict_image(model, eggs_dataset_test[i], settings, filename)
			deblurred = model_gan(eggs_dataset_test[i][0])
			filename = '%s_deblur_pred.png' % imgs[i].split(os.path.sep)[-1].split('.')[0]
			predict_image(model, (deblurred, None), settings, filename)

# python main.py -f C:\Users\pm15334\ParasiticEggDetection\settings\model_settings.json
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training and testing model for Parasitic Egg Detection')
	parser.add_argument('-f','--settings_file', help='Path of the JSON file containing the training settings', type=str)
	parser.add_argument('-m','--mode', help='Mode to run the program (train, test or image)', type=str, default='train')

	args = vars(parser.parse_args())

	settings_file     = args['settings_file'] 
	mode              = args['mode']

	settings = load_settings(settings_file)
	#print(settings)
	settings = Settings(mode, settings)
	
	if settings.output_path:
		check_path(settings.output_path)

	if True:
		print('---------------------------------------')
		print('STARTING SCRIPT FOR SELECTED MODE (%s) ' % mode )
		print('---------------------------------------\n')
		if mode.lower() == 'train':
			train(settings)
		elif mode.lower() == 'test':
			test(settings)
		elif mode.lower() == 'image':
			print('---------------------------------------')
			print('TESTING IMAGE ', settings.root_path)
			print('---------------------------------------')
			test_directory(settings)
		elif mode.lower() == 'dir':
			print('---------------------------------------')
			print('TESTING DIRECTORY ', settings.root_path)
			print('---------------------------------------')
			test_directory(settings)
		elif mode.lower() == 'gan':
			print('---------------------------------------')
			print('TESTING WITH CYCLE GAN ', settings.root_path)
			print('---------------------------------------')
			test_gan(settings)
		elif mode.lower() == 'challenge':
			print('---------------------------------------')
			print('TESTING FOR CHALLENGE ', settings.root_path)
			print('---------------------------------------')
			test_challenge(settings)
		else:
			print('Mode not recognised')
