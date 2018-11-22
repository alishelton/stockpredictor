import torch
import os
import json
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from processing.stockdataset import StockDataset
from processing.preprocessor import PreProcessor
from models.stackedautoencoder import StackedDAE
from models.lstmmodel import LSTMModel
from pprint import pprint

base_file_path = '/Users/alishelton/Desktop/deeplearning/stocks/'
data_file_path = os.path.join(base_file_path, 'stock_data')
model_file_path = os.path.join(base_file_path, 'models')

TRAIN_AUTO = False

data_params = {
	'batch_size': 30,
	'shuffle': True,
	'num_workers': 4
}

model_params = {
	'num_epochs': 5000,
	'lr': 0.05
}

if __name__ == '__main__':
	# Load in data and preprocess using DWT
	print('============STARTING UP TRAINING============')
	with open(os.path.join(base_file_path, 'SnP500Dump.json'), 'r') as f:
		SnP500 = json.load(f)
	preprocessor = PreProcessor() 
	for symbol in SnP500:
		preprocessor.read_data(symbol, os.path.join(data_file_path, symbol + '.csv'))
	preprocessor.preprocess_all()
	data, targets = preprocessor.get_data() # gets all data and targets

	# maybe fix how this works, we need some better way of processing the data at the encoder level
	# maybe even only form this specific data if train auto encoder is true!
	dataset = StockDataset(data['FB'], targets['FB'])

	train_size = int(0.7 * len(dataset))
	train_data, test_data = Subset(dataset, [i for i in range(train_size)]), \
		Subset(dataset, [i for i in range(train_size, len(dataset))])

	trainloader = DataLoader(train_data, **data_params)
	validloader = DataLoader(test_data, **data_params)

	print('============DONE PRE-PROCESSING============')
	# Set up device to move models and data into (CPU/GPU)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Build and train autoencoder model
	input_size, z_size = 11, 5
	auto_model = StackedDAE(input_dim=input_size, z_dim=z_size, encodeLayer=[9], \
	 decodeLayer=[9], dropout=0.2)
	if TRAIN_AUTO:
		auto_model.pretrain(trainloader, validloader, device, **model_params)
		auto_model.save_model(os.path.join(model_file_path, 'stackedDAEModel'))
	else:
		auto_model.load_model(os.path.join(model_file_path, 'stackedDAEModel'))

	# # Auto encode the data
	train_shift, val_shift = -4, -2
	train_data, test_data = torch.tensor(data['FB'][:train_size+train_shift]).float(), \
		torch.tensor(data['FB'][train_size+train_shift:val_shift]).float()
	train_targets, test_targets = torch.tensor(targets['FB'][:train_size-4]).float(), \
		torch.tensor(targets['FB'][train_size+train_shift:val_shift]).float()
	
	auto_encoded_train = auto_model(train_data)[0].detach().numpy()
	auto_encoded_val = auto_model(test_data)[0].detach().numpy()

	post_auto_trainloader = DataLoader(StockDataset(train_data, train_targets), **data_params)
	post_auto_validloader = DataLoader(StockDataset(test_data, test_targets), **data_params)
	
	# Build lstm model
	input_size, hidden_size, num_layers, window_size, dropout = 11, 11, 5, 10, 0.2
	lstm_model = LSTMModel(input_size, hidden_size, num_layers, window_size, dropout, device)
	lstm_model.fit(post_auto_trainloader, post_auto_validloader, **model_params)
	lstm_model.save_model(os.path.join(model_file_path, 'LSTMModel'))







