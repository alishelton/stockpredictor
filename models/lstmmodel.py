import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pprint import pprint


class LSTMModel(nn.Module):
	def __init__(self, input_size, output_size, num_layers, dropout):
		super(self.__class__, self).__init__()
		self.lstm = nn.LSTM(input_size, output_size, num_layers, dropout)

	def forward(self, x):
		return self.lstm(x)

	def load_model(self, path):
		pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict) 
		self.load_state_dict(model_dict)

	def save_model(self, path):
		torch.save(self.state_dict(), path)

	def fit(self, trainloader, validloader, device, lr=0.001, num_epochs=10):
		self.to(device)

		print("=====LSTM Layer=======")
		optimizer = optim.SGD(self.parameters(), lr=lr)
		criterion = nn.MSELoss()

        # validate
		total_loss = 0.0
		total_num = 0
		for batch_idx, (inputs, targets) in enumerate(validloader):
			inputs = inputs.view(inputs.size(0), -1).float().to(device)
			inputs = Variable(inputs)
			output = self.forward(inputs)

			valid_recon_loss = criterion(output, targets)
			total_loss += valid_recon_loss.item() * len(inputs)
			total_num += inputs.size()[0]

		valid_loss = total_loss / total_num
		print("#Epoch 0: Valid LSTM Loss: %.3f" % (valid_loss))

		for epoch in range(num_epochs):
			# train 1 epoch
			train_loss = 0.0
			for batch_idx, (inputs, targets) in enumerate(trainloader):
				inputs = inputs.view(inputs.size(0), -1).float().to(device)

				optimizer.zero_grad()
				inputs = Variable(inputs)

				output = self.forward(inputs)
				recon_loss = criterion(output, targets)
				train_loss += recon_loss.item() * len(inputs)
				recon_loss.backward()
				optimizer.step()

            # validate
			valid_loss = 0.0
			for batch_idx, (inputs, targets) in enumerate(validloader):
				inputs = inputs.view(inputs.size(0), -1).float().to(device)
				inputs = Variable(inputs)
				output = self.forward(inputs)

				valid_recon_loss = criterion(output, targets)
				valid_loss += valid_recon_loss.item() * len(inputs)

			print("#Epoch %3d: LSTM Loss: %.3f, Valid LSTM Loss: %.3f" % (
				epoch+1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))