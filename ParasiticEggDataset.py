import os
import cv2
import json
import torch

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
               width=2448,
               height=3264):
    self.inputs = inputs
    self.targets = targets
    self.transform = transform
    self.convert_boxes_coordinates = convert_boxes_coordinates
    self.label_mapping = label_mapping
    self.width = width
    self.height = height
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, idx):
    images = Image.open(self.inputs[idx]).convert("RGB")
    #images = cv2.cvtColor(cv2.imread(self.inputs[idx]), cv2.COLOR_BGR2RGB)
    #print(images.shape)
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
    
    if self.transform:
      images, target = self.transform(images,target)
    return images, target


def get_data(annotations_path, root_path, dataset = 's'):
  with open(annotations_path, 'r') as f:
    annotations = json.load(f)
  paths = []
  targets = {'boxes':[], 'labels':[], 'area':[], 'iscrowd':[]}
  for item in annotations:
    #print(item['External ID'], item['External ID'].split('.')[0][2], dataset)
    if item['External ID'].split('.')[0][2] == dataset:
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