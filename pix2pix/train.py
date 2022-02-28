import torch
from pix2pix.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from pix2pix import config
#from dataset import MapDataset
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

import ParasiticEggDataset as ped

from references import utils
from main import get_transform, get_data_for_split

torch.backends.cudnn.benchmark = True



def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        #x = torch.unsqueeze(x[0],0).to(config.DEVICE)
        y = y.to(config.DEVICE)
        #y = torch.unsqueeze(y[0],0).to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    transforms_both = get_transform(train=["crop", "hflip", "vflip"], normalise = False) if 'both_transform' not in dir(config) else config.both_transform
    transforms_source = get_transform(train=["blur", "hue", "saturation", "contrast", "brightness"]) if 'transform_only_input' not in dir(config) else config.transform_only_input
    transforms_target = get_transform(train=[]) if 'transform_only_target' not in dir(config) else config.transform_only_target
    paths, targets, __ = get_data_for_split(config.root_path)
    dataset = ped.ParasiticEggDataset(paths, targets, 
            transforms_both, colour=config.colour, 
            transforms_gan = {'both':transforms_both, 'input' : transforms_source, 'target' : transforms_target})

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

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, train_loader, epoch, folder="evaluation2")


if __name__ == "__main__":
    main()