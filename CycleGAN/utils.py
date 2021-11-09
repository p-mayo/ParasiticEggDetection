# Utils

import random, torch, os, numpy as np
import torch.nn as nn
import copy

from CycleGAN import config
from references import transforms as T

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    #for param_group in optimizer.param_groups:
    #    param_group["lr"] = lr

def get_transforms(domain, crop_image = True):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.RandomRotation())
    if crop_image:
        transforms.append(T.RandomCrop())
    transforms.append(T.RandomVerticalFlip())
    transforms.append(T.RandomHorizontalFlip())
    if domain.lower() == "a": # With Motion Blur
        transforms.append(T.MotionBlur())
    transforms.append(T.Normalize())
    return T.Compose(transforms)
