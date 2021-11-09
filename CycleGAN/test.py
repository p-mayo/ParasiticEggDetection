# Python script for inference
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import check_path

from CycleGAN.dataset import CycleGANDataset, get_image
from CycleGAN.generator import Generator
from CycleGAN.discriminator import Discriminator
from CycleGAN.utils import save_checkpoint, load_checkpoint, get_transforms
from CycleGAN import config


def test_image(model, image, kernel_size=256, stride=256):
	image = torch.unsqueeze(image, axis=0)
	[B, C, H, W] = image.shape
	patches = image.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)
	patches_shape = patches.shape
	new_patches = torch.zeros_like(patches)
	model.eval()
	for i in range(patches_shape[2]):
		for j in range(patches_shape[3]):
			img = patches[:,:,i,j,:].to(config.DEVICE)
			with torch.no_grad():
				new_domain = model(img)
			new_patches[:,:,i,j,:] = new_domain[0].to('cpu')
			print(i,j, img.shape, new_domain.shape, new_patches[:,:,i,j,:].shape)
			torch.cuda.empty_cache()
			save_image(img, os.path.join(config.OUTPUT_PATH, "test_a_orig_%d_%d.png" % (i,j) ))
			save_image(new_patches[0,:,i,j,:], os.path.join(config.OUTPUT_PATH, "test_a_deblur_%d_%d.png" % (i,j) ))
	new_patches = new_patches.contiguous().view(1, 3, -1, kernel_size*kernel_size)
	new_patches = new_patches.permute(0, 1, 3, 2) 
	new_patches = new_patches.contiguous().view(1, 3*kernel_size*kernel_size, -1)
	new_image = F.fold(new_patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)
	return new_image


def test(gen_A, gen_B, loader):
	loop = tqdm(loader, leave=True)
	with torch.no_grad():
		for idx, (input_a, input_b) in enumerate(loop):
			input_a = input_a.to(config.DEVICE)
			input_b = input_b.to(config.DEVICE)
			fake_b = gen_B(input_a)
			fake_a = gen_A(input_b)
			save_image(fake_a, os.path.join(config.OUTPUT_PATH, "a_fake_%d.png" % (idx)))
			save_image(input_a, os.path.join(config.OUTPUT_PATH, "a_real_%d.png" % (idx)))
			save_image(fake_b, os.path.join(config.OUTPUT_PATH, "b_fake_%d.png" % (idx)))
			save_image(input_b, os.path.join(config.OUTPUT_PATH, "b_real_%d.png" % (idx)))

def main():
	disc_A = Discriminator(in_channels=3).to(config.DEVICE)
	gen_A = Generator(img_channels = 3, num_residuals=9).to(config.DEVICE)
	disc_B = Discriminator(in_channels=3).to(config.DEVICE)
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
	load_checkpoint(config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE)
	load_checkpoint(config.CHECKPOINT_DISC_A, disc_A, opt_disc, config.LEARNING_RATE)

	check_path(config.OUTPUT_PATH)
	print("=> Saving output at ", config.OUTPUT_PATH)

	if not os.path.isdir(config.DOMAIN_A_DIR):
		#img = get_image(config.DOMAIN_A_DIR, get_transforms('a', False))
		#img_b = test_image(gen_B, img)
		#save_image(img, os.path.join(config.OUTPUT_PATH, "test_a_orig.png" ))
		#save_image(img_b, os.path.join(config.OUTPUT_PATH, "test_a_deblur.png" ))

		dataset = CycleGANDataset(
			root_domain_a = [config.DOMAIN_A_DIR], 
			root_domain_b = [config.DOMAIN_A_DIR], 
			transforms_a = get_transforms("a"),
			transforms_b = get_transforms("b"))

		loader = DataLoader(
			dataset,
			batch_size = config.BATCH_SIZE,
			shuffle = True,
			num_workers = config.NUM_WORKERS,
			pin_memory = True
		)

		with torch.no_grad():
			for i in range(10):
				img = get_image(config.DOMAIN_A_DIR, get_transforms('a'))
				img = torch.unsqueeze(img, axis=0).to(config.DEVICE)
				new_domain = gen_B(img)
				save_image(img, os.path.join(config.OUTPUT_PATH, "test_a_random_orig_%d.png" % (i) ))
				save_image(new_domain, os.path.join(config.OUTPUT_PATH, "test_a_random_deblur_%d.png" % (i) ))
				test(gen_A, gen_B, loader)
	else:
		load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)
		load_checkpoint(config.CHECKPOINT_DISC_B, disc_B, opt_disc, config.LEARNING_RATE)
		dataset = CycleGANDataset(
			root_domain_a = config.DOMAIN_A_DIR, 
			root_domain_b = config.DOMAIN_B_DIR, 
			transforms_a = get_transforms("a"),
			transforms_b = get_transforms("b"))

		loader = DataLoader(
			dataset,
			batch_size = config.BATCH_SIZE,
			shuffle = True,
			num_workers = config.NUM_WORKERS,
			pin_memory = True
		)
		test(gen_A, gen_B, loader)

if __name__ == '__main__':
	main()
