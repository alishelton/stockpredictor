import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from pprint import pprint


class LSTMModel(nn.Module):
	def __init__(self, input_size, output_size, num_layers, dropout):
		super(LSTMModel, self).__init__()
		self.lstm = nn.LSTM(input_size, output_size, num_layers, dropout)

	def forward(self, x):
		return self.lstm(x)