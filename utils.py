import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

categories = ['ascaris', 'hookworm', 'large_egg', 'ov', 'tenia']

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting metrics from a log file')
	parser.add_argument('-f','--log_file', help='Path of the log file from training', type=str)
	args = vars(parser.parse_args())

	log_file     = args['log_file'] 
	extract_metrics(log_file)