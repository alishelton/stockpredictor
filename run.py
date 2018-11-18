import torch
import time
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from preprocessor import PreProcessor
from autoencoder import StackedAutoEncoder
from pprint import pprint

file_path, file_ext = '/Users/alishelton/Desktop/deeplearning/stocks/stock_data/', '.csv'

if __name__ == '__main__':
	# Load in all data and preprocess
	SnP500 = np.load('snp names list filepath')
	preprocessor = PreProcessor()
	for symbol in SnP500:
		preprocessor.read_data(name, os.path.join(file_path, symbol + file_ext))
	preprocessor.preprocess_all()

	# Set up device to move models and data into (CPU/GPU)
	device = torch.device('cuda:0' if torch.cuda_is_available() else 'cpu')

	# Build autoencoder model and train it
	input_size, output_size = 15, 10
	model = StackedAutoEncoder(input_size, output_size).to(device)

	num_epochs = 50
	for epoch in range(num_epochs):
		if epoch % 10 == 0:
			# some stuff here

		model.train()
		total_time = time.time()
		correct = 0
		for i, data in enumerate(dataloader):
			
