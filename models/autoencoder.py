import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from preprocessor import PreProcessor
from pprint import pprint

class AutoEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, output_size),
			nn.Sigmoid()
		)
		self.decoder = nn.Sequential(
			nn.Linear(output_size, input_size),
			nn.Sigmoid()
		)

		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

	def forward(self, x):
		x = x.detach()

		# add noise here?
		y = self.encoder(x)

		if self.training:
			x_reconstruct = self.decoder(y)
			loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		return y.detach()

	def reconstruct(self, x):
		return self.decoder(x)


class StackedAutoEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(StackedAutoEncoder, self).__init__()

		self.ae1 = AutoEncoder(input_size, output_size)
		self.ae2 = AutoEncoder(output_size, output_size)
		self.ae3 = AutoEncoder(output_size, output_size)
		self.ae4 = AutoEncoder(output_size, output_size)
		self.ae5 = AutoEncoder(output_size, output_size)
		# not sure about these params

	def forward(self, x):
		a1 = self.ae1(x)
		a2 = self.ae2(a1)
		a3 = self.ae2(a2)
		a4 = self.ae2(a3)
		a5 = self.ae2(a4)

		if self.training:
			return a5
		else:
			return a5, self.reconstruct(a5)

	def reconstruct(self, x):
		a4_reconstruct = self.ae5.reconstruct(x)
		a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
		a2_reconstruct = self.ae3.reconstruct(a3_reconstruct)
		a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
		x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
		return x_reconstruct



