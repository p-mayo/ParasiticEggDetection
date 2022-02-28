import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision.ops import box_convert

IMAGE_EXTENSIONS = ["JPG", "JPEG", "PNG"]

class ParasiticEggDataset(torch.utils.data.Dataset):
  def __init__(self, 
               inputs, 
               targets, 
               transform, 
               convert_boxes_coordinates=None, 
               label_mapping=None,
               colour=True,
               transforms_gan=None):
    self.inputs = inputs
    self.targets = targets
    self.transform = transform
    self.convert_boxes_coordinates = convert_boxes_coordinates
    self.label_mapping = label_mapping
    self.colour = colour
    self.transforms_gan = transforms_gan
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, idx):
    if self.colour:
      images = Image.open(self.inputs[idx]).convert("RGB")
    else:
      images = Image.open(self.inputs[idx]).convert("L").convert("RGB")
    if self.transforms_gan:
      input_image = np.array(images)
      target_image = np.array(images)
      #input_image = torch.tensor(np.array(images))
      #target_image = torch.tensor(np.array(images))
      new_shape =  list(input_image.shape)
      old_shape = list(input_image.shape)

      new_shape[0] = int(np.ceil(new_shape[0]/256)*256)
      new_shape[1] = int(np.ceil(new_shape[1]/256)*256)

      new_input = np.zeros(new_shape)
      new_target = np.zeros(new_shape)
      #print(new_shape, old_shape)
      for i in range(3):
        new_input[0:old_shape[0],0:old_shape[1],i] = input_image[0:old_shape[0],0:old_shape[1],i]
        new_target[0:old_shape[0],0:old_shape[1],i] = target_image[0:old_shape[0],0:old_shape[1],i]
      input_image = new_input.astype('uint8')
      target_image = new_target.astype('uint8')
      #print(input_image.shape, target_image.shape)
      if self.targets:
        augmentations = self.transforms_gan['both'](image = input_image, image0 = target_image,
                                 bboxes = self.targets['boxes'][idx], labels = self.targets['labels'][idx])
      else:
        augmentations = self.transforms_gan['both'](image = input_image, image0 = target_image)
      input_image, target_image = augmentations['image'], augmentations['image0']
      input_image = self.transforms_gan['input'](image = input_image)['image']
      target_image = self.transforms_gan['target'](image = target_image)['image']
      return input_image, target_image
    else:
      #images = cv2.cvtColor(cv2.imread(self.inputs[idx]), cv2.COLOR_BGR2RGB)
      #print(images.shape)
      if self.targets:
        boxes = self.targets['boxes'][idx].copy()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if self.convert_boxes_coordinates:
          boxes = box_convert(boxes, in_fmt=self.convert_boxes_coordinates['from'], 
                              out_fmt=[self.convert_boxes_coordinates['to']])    
        labels_idx = self.targets['labels'][idx].copy()
        if self.label_mapping:
          for i in range(len(labels_idx)):
            labels_idx[i] = self.label_mapping[labels_idx[i]]
          labels_idx = torch.as_tensor(labels_idx, dtype=torch.int64)
        idx = torch.tensor([idx])
        iscrowd = torch.tensor(self.targets['iscrowd'][idx], dtype=torch.int64)
        area = torch.tensor(self.targets['area'][idx].copy(), dtype=torch.float32)
        target = {'labels':labels_idx, 'area':area, 'boxes':boxes, 'image_id':idx, 'iscrowd':iscrowd}
      else:
        target = None
      if self.transform:
        images, target = self.transform(images, target)
    return images, target


def get_data(annotations_path, root_path, dataset = 's'):
  with open(annotations_path, 'r') as f:
    annotations = json.load(f)
  paths = []
  targets = {'boxes':[], 'labels':[], 'area':[], 'iscrowd':[]}
  for item in annotations:
    #print(item['External ID'], item['External ID'].split('.')[0][2], dataset)
    if len(dataset) == 1:
      condition = item['External ID'].lower().split('.')[0][2] == dataset
    else:
      condition = item['External ID'].lower().find(dataset) > -1
    if condition:
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
      if temp_label[0].lower() not in ["large_egg", "largeegg"]:
        img_path = os.path.join(root_path[temp_label[0]], item['External ID'])
        # If image doesn't exist, try with different image extension
        if not os.path.exists(img_path):
          ext_starts = img_path.find('.')
          for ext in IMAGE_EXTENSIONS:
            new_path = img_path.replace(img_path[ext_starts+1:], ext)
            if os.path.exists(new_path):
              print(img_path, ". Image not found!!")
              print(new_path, ". Found!! Using that file instead!!")
              img_path = new_path
              break
            new_path = img_path.replace(img_path[ext_starts+1:], ext.lower())
            if os.path.exists(new_path):
              print(img_path, ". Image not found!!")
              print(new_path, ". Found!! Using that file instead!!")
              img_path = new_path
              break
        if os.path.exists(img_path):
          paths.append(img_path)
          targets['labels'].append(temp_label)
          targets['boxes'].append(temp_bbox)
          targets['area'].append(temp_area)
          targets['iscrowd'].append([0.]*len(temp_bbox))
        else:
          print("Skipping ", img_path, ". Image not found!!")
  return paths, targets

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