# Parasitic Egg Detection
[![DOI](https://zenodo.org/badge/399761339.svg)](https://zenodo.org/badge/latestdoi/399761339)
This repository provides the code for the detection of Parasitic Eggs from microscopy images. The code has been fully implemented in Python using Pytorch as main framework. 

Some of the tools used here come from these sources:
* [Torchvision 0.3 Object Detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
* [Train your own object detector with Faster-RCNN & PyTorch](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70)

Both of them based on the Faster-RCNN architecture.

##The dataset
The dataset is composed of six classes: Ascaris, Hookworm, Large Egg, OV, Teania and Trichuris. The images to analyse are high-resolution microscopic images of size **2448x3264**. Some preprocessing might be ideal to remove undesired regions that contribute no information to the discriminative process.

The annotations are provided in a spreadsheet and in a JSON file. Each row corresponds to an annotations in a given image. The code in "main.py" specifies the way the annotations are retrieved. The relevant keys for the file are:

*   External ID. This contains the filename
*   Label -> value. Contains the class of the object
*   Label -> bbox. Contains the coordinates of the bounding box associated to the class. These values provided are: top, left, height and width. 

## Running the code
The file main.py contains all the relevant code to train and test the model on the task. To run, use the following command:

```
python main.py -f <settings_file_location> [-m <mode>]

```

Thus, the main script requires us to specify the path where a JSON file describing the settings for the task and, optionally, the mode to execute it, which can be:

* **train**. To train the network. This will save the model for each epoch during training in the specified output location.
* **test**. To test the model. This mode requires to specify the path where the model has been saved, so it can be loaded. If specified and the annotations are available, it can evaluate the model on the specified dataset. If specifided, it can also produce the output image the model has been tested on.
* **image**. Test the model on a specific image.

The options expected in the JSON file are detailed below.

| option|description |mode
|--|--|--|
|annotations_path| Path where the JSON file containing the annotations for the images is saved | train/test|
|root_path | Root path where the dataset is located. This is, where the ascaris, hookworm, large_egg, ov, taenia, and trichuris folders are contained | train/test|
|num_epochs | Num epochs for training |train|
|batch_size | Batch size for training | train |
|seed | Seed for random numbers generation. Default: 1| train/test|
|output_path | Path to save the output of the task, this could be the model, the metrics and/or the predictions | train/test/image |
|kfolds | The number of folds to perform. Default: 5 | train/test |
|folds | From the splits, use only the ones specified in here. Optional. Default uses all the folds requested.| train/test|
|transforms| Transforms to use for data augmentation, options: `"saturation", "contrast", "brightness", "hue", "rotation", "hflip", "vflip", "blur"`. | train/test|
|remove_scores | Threshold specifying the lower bound of scores allowed in the metrics. Default: 0.5 | train/test|
|augment_test | Flag. Specify whether to use transforms for data augmentation or not. If True, the same transform functions used for training would be used for the testing | train |
|model_path | Location of the model to load for evaluation | test/image |
|idxs| Specifies any number of indexes to show predictions during testing. Default uses all the images in the dataset. This does not affect the dataset used for evaluation | test|
|evaluate_model | Flag. Specifies whether to compute also the metrics for the dataset specified or not. Default: False| test|
|show_predictions | Specifies whetehr to produce images showing the prediction and ground truth as bounding boxes for inspection. Default: False | test|
|use_gpu | Specifies whether, if available, to use GPU or not. Default: True. | train/test/image|

Example of execution of a test task:

    python main.py -f C:\Users\parasitic_egg_detection\settings.json -m test

Example of a JSON file content:

    {
		"annotations_path":"C:\\Users\\parasitic_egg_detection\\dataset\\Annotations.json",
		"root_path":"C:\\Users\\parasitic_egg_detection\\dataset",
		"seed":0,
		"output_path":"C:\\Users\\parasitic_egg_detection\\output",
		"kfolds":3,
		"folds":[2],
		"transforms":["saturation", "contrast", "brightness", "hue", "rotation", "hflip", "vflip", "blur"],
		"model_path":"C:\\Users\\parasitic_egg_detection\\output\\fold_2\\fold_2_epoch_24.pkl",
		"idxs":-1,
		"evaluate_model":1,
		"show_predictions":0,
		"use_gpu":0
	}

The above file will evaluate the model only without using the GPU. The model to evaluate corresponds to the epoch 24 of fold 2. The output, which is the metrics computed during evaluation, will be saved in the specified location.

The metrics CSV file contains the fold, epoch, loss, True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall and F1-Score for each one of the classes in the dataset. If mode is testing,, epoch and loss will have "test" as value. 

[![DOI](https://zenodo.org/badge/399761339.svg)](https://zenodo.org/badge/latestdoi/399761339)