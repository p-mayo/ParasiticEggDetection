# Python script to extract metrics from the CSV file containing the results

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils import check_path

iou_titles = {
	1: "IoU: 0.5:0.95",
	2: "IoU: 0.5",
	3: "IoU: 0.75",
	4: "IoU: 0.85",
	5: "IoU: 0.90",
	6: "IoU: 0.95",
	7: "IoU: 0.5:0.95",
	8: "IoU: 0.5",
	9: "IoU: 0.75",
	10: "IoU: 0.85",
	11: "IoU: 0.90",
	12: "IoU: 0.95"
}

lbls = ['All', 'ascaris', 'hookworm', 'ov', 'tenia', 'large_egg', 'trichuris']

def compare_folds(data, iou_setting=1, clss='All', output_path=""):
	folds = set(data.fold)
	fig, ax = plt.subplots()
	legends = []
	for f in folds:
		ax = data[data.fold == f].plot(x='epoch', y=["Settings %02d (%s)" % (iou_setting, clss)], ax = ax)
		legends.append('Fold %d' % f)
	title = 'mAP' if iou_setting < 7 else 'Recall'
	fig.suptitle(title + ' for ' + clss.upper())
	ax.set_ylim([0,1.])
	ax.set_title(iou_titles[iou_setting])
	ax.legend(legends)
	if output_path:
		img_path = os.path.join(output_path, '%s_%s_%s.png' %(title, clss, iou_titles[iou_setting].replace(':', '-')))
		plt.savefig(img_path)
		plt.close()
	else:
		plt.show()

def compare_classes(data, iou_setting = 1, fold = 1, output_path = ""):
	fig, ax = plt.subplots()
	legends = []
	for lbl in lbls:
		ax = data[data.fold == fold].plot(x='epoch', y=["Settings %02d (%s)" % (iou_setting, lbl)], ax = ax)
		legends.append(lbl.upper())
	title = 'mAP' if iou_setting < 7 else 'Recall'
	fig.suptitle(title + ' for fold ' + str(fold))
	ax.set_ylim([0,1.])
	ax.set_title(iou_titles[iou_setting])
	ax.legend(legends)
	if output_path:
		img_path = os.path.join(output_path, '%s_%02d_%s.png' %(title, fold, iou_titles[iou_setting].replace(':', '-')))
		plt.savefig(img_path)
		plt.close()
	else:
		plt.show()


def plot_metrics(data, classes = lbls, fold=1, output_path = "", title = ""):
	fig, ax = plt.subplots()
	legends = []
	if 'All' in classes:
		average = data[data.fold == fold][['Precision (ascaris)']].rename(columns={'Precision (ascaris)': 'Precision (All)'})
		for lbl in set(lbls) - set(['All', 'ascaris']):
			average += data[data.fold == fold][['Precision (%s)' % lbl]].rename(columns={'Precision (%s)' % lbl: 'Precision (All)'})
		average/= len(lbls) - 1
		new_data = data[data.fold == fold].join(average)
	else:
		new_data = data
	for lbl in lbls :
		ax = new_data[new_data.fold == fold].plot(x='epoch', y=["Precision (%s)" % (lbl)], ax = ax)
		legends.append(lbl.upper())
	fig.suptitle('Precision for fold ' + str(fold))
	ax.set_ylim([0,1.1])
	ax.legend(legends)
	if output_path:
		img_path = os.path.join(output_path, 'precision_%d_fold.png' %(fold))
		plt.savefig(img_path)
		plt.close()
	else:
		plt.show()


def get_precision_averages(data):
	fig, ax = plt.subplots()
	legends = []

	for fold in [1,2,3]:
		average = data[data.fold == fold][['Precision (ascaris)']].rename(columns={'Precision (ascaris)': 'Precision (All)'})
		for lbl in set(lbls) - set(['All', 'ascaris']):
			average += data[data.fold == fold][['Precision (%s)' % lbl]].rename(columns={'Precision (%s)' % lbl: 'Precision (All)'})
		average/= len(lbls) - 1
		new_data = data[data.fold == fold].join(average)
		print(new_data)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Analysisng metrics for Parasitic Egg Detection')
	parser.add_argument('-f','--metrics_file', help='Path of the CSV file containing the metrics', type=str)
	parser.add_argument('-o','--output_path', help='Path of save the plots', type=str)

	args = vars(parser.parse_args())

	metrics_file     = args['metrics_file'] 
	output_path      = args['output_path'] 
	if (not metrics_file) or (not os.path.exists(metrics_file)):
		print("Selected file does not exist")
	else:
		if output_path:
			check_path(output_path)
		data = pd.read_csv(metrics_file)
		folds = set(data.fold)
		for iou in iou_titles.keys():
			for lbl in lbls:
				compare_folds(data, iou, lbl, output_path)
			for f in folds:
				compare_classes(data, iou, f, output_path)

