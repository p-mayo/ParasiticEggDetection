import os
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
from utils import check_path, check_valid_images

torch.backends.cudnn.benchmark = True



def test(gen, loader, output_path = None, filenames = None, output_path_input = None):
    loop = tqdm(loader, leave=True)

    gen.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loop):
            #print(x.size())
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            #save_image(unnorm(y_fake), folder + f"\\y_gen_{epoch}.png")
            #save_image(unnorm(x), folder + f"\\input_{epoch}.png")
            #save_image(unnorm(y), folder + f"\\label_{epoch}.png")
            if filenames:
                save_image(x * 0.5 + 0.5, os.path.join(output_path_input, filenames[idx]))
                save_image(y_fake, os.path.join(output_path, filenames[idx]))
            else:
                save_image(y_fake, os.path.join(output_path, f"y_gen_{epoch}.png"))
                save_image(x * 0.5 + 0.5, os.path.join(output_path,  f"input_{epoch}.png"))
                save_image(y * 0.5 + 0.5, os.path.join(output_path, f"label_{epoch}.png"))


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
    transforms_both = get_transform(train=["crop", "hflip", "vflip"], normalise = False) if 'both_transform' not in dir(config) else config.both_transform
    transforms_source = get_transform(train=["blur", "hue", "saturation", "contrast", "brightness"]) if 'transform_only_input' not in dir(config) else config.transform_only_input
    transforms_target = get_transform(train=[]) if 'transform_only_target' not in dir(config) else config.transform_only_target
    if ('REPLICATE_DIRECTORY' in dir(config)) and config.REPLICATE_DIRECTORY:
        print('REPLICATE_DIRECTORY', config.root_path)
        for root_dir, dir_names, file_names in os.walk(config.root_path):
            output_path = root_dir.replace(config.root_path, config.OUTPUT_PATH)
            output_path_input = output_path.replace(config.OUTPUT_PATH, config.OUTPUT_PATH + '_blur')
            check_path(output_path)
            check_path(output_path_input)
            imgs = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
            imgs = check_valid_images(imgs)
            if imgs:
                dataset = ped.ParasiticEggDataset(imgs, None, 
                    transforms_both, colour=config.colour, 
                    transforms_gan = {'both':transforms_both, 'input' : transforms_source, 'target' : transforms_target})

                loader = DataLoader(
                    dataset,
                    batch_size = config.BATCH_SIZE,
                    shuffle = False,
                    num_workers = 1,
                    pin_memory = True
                )
                check_path(config.OUTPUT_PATH)
                test(gen, loader, output_path, filenames = [img.split(os.sep)[-1] for img in imgs], output_path_input = output_path_input)
    else:
        print('Not REPLICATE_DIRECTORY')
        paths, __, __ = get_data_for_split(config.root_path)
        dataset = ped.ParasiticEggDataset(paths, None, 
                get_transform(train=transforms_a, p = 1.), colour=config.colour, 
                transform_source = get_transform(train=transforms_b))

        loader = DataLoader(
            dataset,
            batch_size = config.BATCH_SIZE,
            shuffle = False,
            num_workers = 1,
            pin_memory = True,
            collate_fn=utils.collate_fn
        )
        check_path(config.OUTPUT_PATH)
        test(gen_A, gen_B, loader)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )
        save_some_examples(gen, train_loader, epoch, folder="evaluation2")


if __name__ == "__main__":
    main()