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

from deblurring.dataset import CycleGANDataset
from deblurring.generator import Generator
from deblurring.discriminator import Discriminator
from deblurring.utils import save_checkpoint, load_checkpoint
from references import transforms as T
from deblurring import config

# H -> A, Z -> B
def train(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
	loop = tqdm(loader, leave=True)
	for idx, (input_a, input_b) in enumerate(loop):
		input_a = input_a.to(config.DEVICE)
		input_b = input_b.to(config.DEVICE)

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

		opt_gen.zero_grad()
		g_scaler.scale(G_loss).backward()
		g_scaler.step(opt_gen)
		g_scaler.update()

		if (epoch == 99) and (980 < idx < 990):
			save_image(fake_a*0.5 + 0.5, os.path.join("saved_images", "a_fake_%d.png" % (idx)))
			save_image(input_a*0.5 + 0.5, os.path.join("saved_images", "a_real_%d.png" % (idx)))
			save_image(fake_b*0.5 + 0.5, os.path.join("saved_images", "b_fake_%d.png" % (idx)))
			save_image(input_b*0.5 + 0.5, os.path.join("saved_images", "b_real_%d.png" % (idx)))

def get_transform(domain):
	transforms = []
	transforms.append(T.ToTensor())
	transforms.append(T.RandomRotation())
	transforms.append(T.RandomVerticalFlip())
	transforms.append(T.RandomHorizontalFlip())
	if domain.lower() == "a": # With Motion Blur
		transforms.append(T.MotionBlur())
	transforms.append(T.Normalize())


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

	if config.LOAD_MODEL:
		load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
		load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)
		load_checkpoint(config.CHECKPOINT_DISC_A, disc_A, opt_disc, config.LEARNING_RATE)
		load_checkpoint(config.CHECKPOINT_DISC_B, disc_B, opt_disc, config.LEARNING_RATE)

	dataset = CycleGANDataset(
		root_horse = os.path.join(config.TRAIN_DIR, "horses"), 
		root_zebra = os.path.join(config.TRAIN_DIR, "zebras"), 
		transform = config.transforms)

	loader = DataLoader(
		dataset,
		batch_size = config.BATCH_SIZE,
		shuffle = True,
		num_workers = config.NUM_WORKERS,
		pin_memory = True
	)

	g_scaler = torch.cuda.amp.GradScaler()
	d_scaler = torch.cuda.amp.GradScaler()

	for epoch in range(config.NUM_EPOCHS):
		train(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

		if config.SAVE_MODEL:
			save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
			save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
			save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_DISC_A)
			save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_DISC_B)

if __name__ == '__main__':
	main()
