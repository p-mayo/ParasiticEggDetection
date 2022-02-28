# Main script for training the GAN-DO model
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.model_selection import StratifiedKFold

import ParasiticEggDataset as ped

from GAN_DO import config
from GAN_DO.discriminator import Discriminator
from pix2pix.utils import save_checkpoint, load_checkpoint, save_some_examples
from references import utils
from references.engine import keep_outputs
from main import get_transform, get_data_for_split, predict_image, get_targets, unnorm
from utils import label_mapping, check_path, draw_boxes

def get_target_for_disc(x):
    data = []
    for i in range(len(x)):
        for j in range(len(x[i]['boxes'])):
            bbx_lbl = torch.zeros(5)
            bbx_lbl[0:4] = x[i]['boxes'][j]
            bbx_lbl[-1] = x[i]['labels'][j]
            data.append(bbx_lbl)
    #print(data)
    if not data:
        data = [torch.zeros(5)]
    x = torch.stack(data)
    return x

def train(disc, baseline, gen, loader, opt_disc, opt_gen, bce, 
    gen_scaler, disc_scaler, transforms, device):

    loop = tqdm(loader, leave=True)
    baseline.eval()
    for idx, (x, gt) in enumerate(loop):
        # I need to train discriminator and generator as per the loss functions
    	# x is the original image, y is the ground truth (bounding boxes)
        x_real = list(xi.to(device) for xi in x)
        #x = torch.unsqueeze(x[0],0).to(config.DEVICE)

        # x_hat is the original image distorted
        x_fake = [transforms(xi)[0].to(device) for xi in x]

        # Ground truth is not part of the input of this model
        #gt = [{k: v.to(device) for k, v in t.items()} for t in gt]
        #y = torch.unsqueeze(y[0],0).to(config.DEVICE)

        # First train the discriminator on real data
        with torch.cuda.amp.autocast():
            y_real = get_target_for_disc(baseline(x_real)).to(device)
            y_fake = get_target_for_disc(gen(x_fake)).to(device)
            d_real = disc(y_real)
            d_real_loss = bce(d_real, torch.ones_like(d_real))

            d_fake = disc(y_fake.detach())
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss)/2

        opt_disc.zero_grad()
        disc_scaler.scale(d_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()


        # Secondly, train the discriminator on the generated/fake data
        with torch.cuda.amp.autocast():           
            d_fake = disc(y_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            g_loss = g_fake_loss

        opt_gen.zero_grad()
        gen_scaler.scale(g_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                d_real=torch.sigmoid(d_real).mean().item(),
                d_fake=torch.sigmoid(d_fake).mean().item(),
            )


def test():
	pass

def save_image(img, prediction, target, filename):
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    new_outputs = keep_outputs(prediction[0], keep, 
        remove_scores = config.remove_scores)
    img = draw_boxes(unnorm(img).permute(1,2,0).numpy().copy(), 
        new_outputs['boxes'], new_outputs['labels'], new_outputs['scores'])
    img = draw_boxes(img, target['boxes'], target['labels'])
    fname = os.path.join(config.output_path, filename)
    fig, axs = plt.subplots(figsize=(20,20))
    axs.imshow(img)
    axs.set_xticks([])
    axs.set_yticks([])
    plt.savefig(fname, transparent=True, bbox_inches='tight')

def main():
    paths, targets, labels = get_data_for_split(config.root_path, False)
    skf = StratifiedKFold(n_splits=config.kfolds)#, shuffle=True, random_state=settings.seed)
    skf.get_n_splits(paths, labels)
    with torch.no_grad():
        for fold, (train_idx, test_idx) in enumerate(skf.split(paths,labels),1):
            if fold == config.fold:
                break

    disc = Discriminator().to(config.DEVICE)
    #gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    
    baseline_model = torch.load(config.model_path)
    generator_model = torch.load(config.model_path)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(generator_model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    #L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, generator_model, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    transforms_both = get_transform(train=["crop", "hflip", "vflip"], normalise = False) if 'both_transform' not in dir(config) else config.both_transform
    transforms_source = get_transform(train=["blur", "hue", "saturation", "contrast", "brightness"]) if 'transform_only_input' not in dir(config) else config.transform_only_input
    transforms_target = get_transform(train=[]) if 'transform_only_target' not in dir(config) else config.transform_only_target
    #paths, targets, __ = get_data_for_split(config.root_path)
    #print(get_targets(targets, range(len(paths))))
    dataset = ped.ParasiticEggDataset(np.array(paths)[train_idx].tolist(), 
                    get_targets(targets, train_idx),
            get_transform(train=[]), label_mapping=label_mapping, colour=config.colour)

    train_loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
        #collate_fn=utils.collate_fn # If used, it retrieves as list of tensors
    )
    #train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    #train_loader = DataLoader(
    #    train_dataset,
    #    batch_size=config.BATCH_SIZE,
    #    shuffle=True,
    #    num_workers=config.NUM_WORKERS,
    #)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    #val_dataset = MapDataset(root_dir=config.VAL_DIR)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    check_path(config.OUTPUT_PATH)
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch: {epoch}')
        train(
            disc, baseline_model, generator_model, train_loader, opt_disc, opt_gen, 
            BCE, g_scaler, d_scaler, transforms_source, config.DEVICE
        )

        if config.SAVE_MODEL:
            save_checkpoint(generator_model, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        generator_model.eval()
        with torch.no_grad():
            for i in np.random.randint(0,len(dataset), 10):
                print("Processing image: ", paths[i])
                x, y = dataset[i]
                with torch.no_grad():
                    x_real = [x.to(config.device)]
                    x_fake = [transforms_source(x)[0].to(config.device)]

                    with torch.cuda.amp.autocast():
                        y_real = baseline_model(x_real)
                        y_fake = generator_model(x_fake)
                        filename = 'epoch_%d_%s_pred_hq_baseline.png' % (epoch, paths[i].split(os.path.sep)[-1].split('.')[0])
                        save_image(x_real[0].to('cpu'), y_real, y, filename)
                        filename = 'epoch_%d_%s_pred_lq_generator.png' % (epoch, paths[i].split(os.path.sep)[-1].split('.')[0])
                        save_image(x_fake[0].to('cpu'), y_fake, y, filename)
                        y_real = generator_model(x_real)
                        y_fake = baseline_model(x_fake)
                        filename = 'epoch_%d_%s_pred_hq_generator.png' % (epoch, paths[i].split(os.path.sep)[-1].split('.')[0])
                        save_image(x_real[0].to('cpu'), y_real, y, filename)
                        filename = 'epoch_%d_%s_pred_lq_baseline.png' % (epoch, paths[i].split(os.path.sep)[-1].split('.')[0])
                        save_image(x_fake[0].to('cpu'), y_fake, y, filename)


if __name__ == '__main__':
	main()