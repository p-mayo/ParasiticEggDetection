# Python file for the training the CycleGAN architecture
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import ParasiticEggDataset as ped

from main import get_transform, get_data_for_split
from utils import check_path
from CycleGAN.ped import CycleGAN_PED
from CycleGAN.generator import Generator
from CycleGAN.discriminator import Discriminator
from CycleGAN.utils import save_checkpoint, load_checkpoint, get_transforms
from CycleGAN import config
from references import utils
from references.transforms import UnNormalize

unnorm_samsung = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
unnorm_canon = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
unnorm = UnNormalize()

# H -> A, Z -> B
def train(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
	loop = tqdm(loader, leave=True)
	for idx, (input_a, input_b) in enumerate(loop):
	#for input_a, input_b in loader:
		#input_a = list(image.to(config.DEVICE) for image in input_a)
		#input_b = list(image.to(config.DEVICE) for image in input_b)
		input_a = torch.unsqueeze(input_a[0],0).to(config.DEVICE)
		input_b = torch.unsqueeze(input_b[0],0).to(config.DEVICE)

		# Training the Discriminators
		with torch.cuda.amp.autocast():
			fake_a = gen_A(input_b)
			D_A_real = disc_A(input_a)
			D_A_fake = disc_A(fake_a.detach())
			D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
			D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
			D_A_loss = D_A_real_loss + D_A_fake_loss

			fake_b = gen_B(input_a)
			D_B_real = disc_B(input_b)
			D_B_fake = disc_B(fake_b.detach())
			D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
			D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
			D_B_loss = D_B_real_loss + D_B_fake_loss

			D_loss = (D_A_loss + D_B_loss)/2

		opt_disc.zero_grad()
		d_scaler.scale(D_loss).backward()
		d_scaler.step(opt_disc)
		d_scaler.update()


		with torch.cuda.amp.autocast():
			# Adversarial loss
			D_A_fake = disc_A(fake_a)
			D_B_fake = disc_B(fake_b)
			loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
			loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

			# Cycle Loss
			cycle_a = gen_A(fake_b)
			cycle_b = gen_B(fake_a)
			#print(input_a.shape)
			#print(cycle_a.shape)
			cycle_a_loss = l1(input_a, cycle_a)
			cycle_b_loss = l1(input_b, cycle_b)

			# Identity Loss
			identity_b = gen_A(input_b)
			identity_a = gen_B(input_a)
			identity_b_loss = l1(input_b, identity_b)
			identity_a_loss = l1(input_a, identity_a)

			# Putting everything together
			G_loss = loss_G_B + loss_G_A + cycle_b_loss * config.LAMBDA_CYCLE + \
						cycle_a_loss * config.LAMBDA_CYCLE + \
						identity_b_loss * config.LAMBDA_IDENTITY + \
						identity_a_loss * config.LAMBDA_IDENTITY
		
		#if epoch == 0:
		#	with torch.no_grad():
		#		save_image(unnorm(input_a.clone().detach()), os.path.join(config.OUTPUT_PATH, "a_%d_1_real.png" % (idx)))
		#		save_image(unnorm(fake_b.clone().detach()), os.path.join(config.OUTPUT_PATH, "a_%d_2_supres.png" % (idx)))
		#		save_image(unnorm(cycle_a.clone().detach()), os.path.join(config.OUTPUT_PATH, "a_%d_3_recons.png" % (idx)))
		#		save_image(unnorm(input_b.clone().detach()), os.path.join(config.OUTPUT_PATH, "b_%d_1_real.png" % (idx)))
		#		save_image(unnorm(fake_a.clone().detach()), os.path.join(config.OUTPUT_PATH, "b_%d_2_lowres.png" % (idx)))
		#		save_image(unnorm(cycle_b.clone().detach()), os.path.join(config.OUTPUT_PATH, "b_%d_3_recons.png" % (idx)))

		opt_gen.zero_grad()
		g_scaler.scale(G_loss).backward()
		g_scaler.step(opt_gen)
		g_scaler.update()


def main():
	disc_A = Discriminator(in_channels=3).to(config.DEVICE)
	disc_B = Discriminator(in_channels=3).to(config.DEVICE)
	gen_A = Generator(img_channels = 3, num_residuals=9).to(config.DEVICE)
	gen_B = Generator(img_channels = 3, num_residuals=9).to(config.DEVICE)

	opt_disc = optim.Adam(
		list(disc_A.parameters()) + list(disc_B.parameters()),
		lr = config.LEARNING_RATE,
		betas = (0.5, 0.999)
	)

	opt_gen = optim.Adam(
		list(gen_A.parameters()) + list(gen_B.parameters()),
		lr = config.LEARNING_RATE,
		betas = (0.5, 0.999)
	)

	L1 = nn.L1Loss()
	mse = nn.MSELoss()
	genA_full_path = os.path.join(config.OUTPUT_PATH, config.CHECKPOINT_GEN_A)
	genB_full_path = os.path.join(config.OUTPUT_PATH, config.CHECKPOINT_GEN_B)
	discA_full_path = os.path.join(config.OUTPUT_PATH, config.CHECKPOINT_DISC_A)
	discB_full_path = os.path.join(config.OUTPUT_PATH, config.CHECKPOINT_DISC_B)
	if config.LOAD_MODEL:
		load_checkpoint(genA_full_path, gen_A, opt_gen, config.LEARNING_RATE)
		load_checkpoint(genB_full_path, gen_B, opt_gen, config.LEARNING_RATE)
		load_checkpoint(discA_full_path, disc_A, opt_disc, config.LEARNING_RATE)
		load_checkpoint(discB_full_path, disc_B, opt_disc, config.LEARNING_RATE)

	transforms_a = ["crop", "blur"]
	transforms_b = ["crop"]
	paths, __, __ = get_data_for_split(config.root_path)
	dataset = ped.ParasiticEggDataset(paths, None, 
			get_transform(train=transforms_a), colour=config.colour, 
			transform_source = get_transform(train=transforms_b))

	loader = DataLoader(
		dataset,
		batch_size = config.BATCH_SIZE,
		shuffle = True,
		num_workers = 1,
		pin_memory = True,
		collate_fn=utils.collate_fn
	)

	g_scaler = torch.cuda.amp.GradScaler()
	d_scaler = torch.cuda.amp.GradScaler()

	check_path(config.OUTPUT_PATH)

	for epoch in range(config.NUM_EPOCHS):
		print('Epoch ', epoch)
		train(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

		if config.SAVE_MODEL:
			save_checkpoint(gen_A, opt_gen, filename=genA_full_path)
			save_checkpoint(gen_B, opt_gen, filename=genB_full_path)
			save_checkpoint(disc_A, opt_disc, filename=discA_full_path)
			save_checkpoint(disc_B, opt_disc, filename=discB_full_path)

if __name__ == '__main__':
	main()
