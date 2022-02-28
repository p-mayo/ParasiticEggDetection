# Discriminator
import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.disc = nn.Sequential(
			nn.Linear(5, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.disc(x)

