import os
import torch
import json

from PIL import Image
from torchvision.ops import box_convert

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