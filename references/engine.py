import math
import sys
import time
import torch
import numpy as np
import torchvision.models.detection.mask_rcnn

from references.coco_utils import get_coco_api_from_dataset
from references.coco_eval import CocoEvaluator
from references import utils
import utils as ut

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def keep_outputs(outputs, idxs, remove_scores = 0.):
    if remove_scores > 0.:
        new_idxs = []
        for idx in idxs:
            #print(outputs['scores'][idx])
            if outputs['scores'][idx].item() > remove_scores:
                new_idxs.append(idx.item())
        #print('original: ', idxs, 'new:', new_idxs)
        idxs = new_idxs
    new_outputs = {'boxes':outputs['boxes'][idxs], 
                    'scores':outputs['scores'][idxs], 
                    'labels':outputs['labels'][idxs]
                    }
    return new_outputs

@torch.no_grad()
def evaluate_old(model, data_loader, device, nms_threshold=0.7):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device(device)
    #print(device, cpu_device)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    i=0
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if device.type != 'cpu':
                torch.cuda.synchronize()
            model_time = time.time()
            #print(len(images), images[0].shape)
            outputs = model(images)
            new_outputs = []
            for o in outputs:
                boxes = o['boxes']
                scores = o['scores']
                keep = torchvision.ops.nms(boxes, scores, nms_threshold)
                new_outputs.append(keep_outputs(o, keep))

            outputs = new_outputs
            # cpu_device
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            #print(i, outputs[0]['labels'], outputs[0]['scores'])
            i = i + 1
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            print(res)
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
def evaluate(model, data_loader, device, nms_threshold=0.7, iou_thresholds = [],
                remove_scores = 0.5):
    if not iou_thresholds:
        iou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95]
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device(device)
    #print(device, cpu_device)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    #coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    #coco_evaluator = CocoEvaluator(coco, iou_types)
    i=0
    fps = 0
    fns = 0
    tps = []
    n_classes = len(ut.label_mapping) + 1 # For background
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if device.type != 'cpu':
                torch.cuda.synchronize()
            model_time = time.time()
            #print(len(images), images[0].shape)
            outputs = model(images)
            new_outputs = []
            for o in outputs:
                boxes = o['boxes']
                scores = o['scores']
                keep = torchvision.ops.nms(boxes, scores, nms_threshold)
                new_outputs.append(keep_outputs(o, keep, remove_scores))
            outputs = new_outputs
            # cpu_device
            #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            #print(i, outputs[0]['labels'], outputs[0]['scores'])
            i = i + 1
            #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            #coco_evaluator.update(res)
            # print(targets, outputs)
            for i in range(len(images)):
                confusion_matrix_sample = get_conf_matrix(targets[0]['boxes'], targets[0]['labels'], 
                                            outputs[0]['boxes'], outputs[0]['labels'], n_classes)
                confusion_matrix += confusion_matrix_sample
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        #print(len(tps), fps, fns)
        #print(compute_prec_rec(tps, fps, fns))
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        #coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        #coco_evaluator.accumulate()
        #coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
    return get_metrics(confusion_matrix)


def compute_iou(gt, pred):
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])

    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])

    int_area = max(0, (x2 - x1)) * max(0, (y2 - y1))

    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    
    return  int_area/(gt_area + pred_area - int_area)

def get_metrics_old(gt_boxes, gt_labels, pred_boxes, pred_labels):
    fp = 0
    fn = 0
    tp = []
    #print(gt_labels, pred_labels)
    if set(gt_labels) & set(pred_labels) == 0:
        fp = len(pred_labels)
        fn = len(gt_labels)
    else:
        pred_ids = [i for i in range(len(pred_labels))]
        for i in range(len(gt_labels)):
            best_iou = 0
            best_idx = -1
            for j in pred_ids:
                if gt_labels[i].item() == pred_labels[j].item():
                    iou = compute_iou(gt_boxes[i], pred_boxes[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
            if best_idx > -1:
                tp.append([pred_labels[best_idx], best_iou])
                pred_ids.remove(best_idx)
            else:
                fn += 1
        fp += len(pred_ids)
    return fp, fn, tp

def get_conf_matrix(gt_boxes, gt_labels, pred_boxes, pred_labels, n_classes):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    #print(gt_labels, pred_labels)
    pred_ids = [i for i in range(len(pred_labels))]
    for i in range(len(gt_labels)):
        best_iou = 0
        best_idx = -1
        for j in pred_ids:
            iou = compute_iou(gt_boxes[i], pred_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx > -1:
            pred_ids.remove(best_idx)
            conf_matrix[gt_labels[i], gt_labels[i]] += 1
        else:
            conf_matrix[0, gt_labels[i]] += 1
    for idx in pred_ids:
        conf_matrix[pred_labels[idx], 0] += 1
    return conf_matrix

def compute_prec_rec_old(tps, fps, fns, iou = 0.5):
    #tp_iou = np.sum(np.array(tps) >= iou)
    tp_iou = 0
    for tp in tps:
        if tp[1].item() >= iou:
            tp_iou += 1
    fps += len(tps) - tp_iou
    tps = tp_iou

    print(tps, fps, fns)
    precision = tps/(tps + fps)
    recall = tps/(tps + fns)
    return precision, recall
    
def get_metrics(confusion_matrix):
    counts = np.zeros((confusion_matrix.shape[0]-1, 7))
    for i in range(confusion_matrix.shape[0]-1):
        counts[i,0] = i + 1     # Class label
        counts[i,1] = confusion_matrix[i+1,i+1]                         # True Positives
        counts[i,2] = np.sum(confusion_matrix[i+1, :]) - counts[i,1]    # False Positives
        counts[i,3] = np.sum(confusion_matrix[:, i+1]) - counts[i,1]    # False Negatives
        counts[i,4] = counts[i,1]/(counts[i,1] +  counts[i,2])          # Precision
        counts[i,5] = counts[i,1]/(counts[i,1] +  counts[i,3])          # Recall
        counts[i,6] = counts[i,1]/(counts[i,1] + 0.5*(counts[i,2] + counts[i,3])) # F1 score
    return counts
