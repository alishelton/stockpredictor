import torch
import time
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

TRAIN_AUTO = True

data_params = {
	'batch_size': 32,
	'shuffle': True,
	'num_workers': 4
}

model_params = {
	'num_epochs': 1000,
	'lr': 0.005
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

	dataset = StockDataset(data['FB'], targets['FB'])

	train_size = int(0.7 * len(dataset))
	train_data, test_data = Subset(dataset, [i for i in range(train_size)]), Subset(dataset, [i for i in range(train_size, len(dataset))])

	trainloader = DataLoader(train_data, **data_params)
	validloader = DataLoader(test_data, **data_params)

	print('============DONE PRE-PROCESSING============')
	# Set up device to move models and data into (CPU/GPU)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Build and train autoencoder model
	input_size, output_size = 11, 7
	auto_model = StackedDAE(input_dim=input_size, z_dim=output_size, encodeLayer=[output_size] * 3,\
	 decodeLayer=[output_size], dropout=0.2)
	if TRAIN_AUTO:
		auto_model.pretrain(trainloader, validloader, device, **model_params)
		auto_model.save_model(os.path.join(model_file_path, 'stackedDAEModel'))
	else:
		auto_model.load_model(os.path.join(model_file_path, 'stackedDAEModel'))

	# # Auto encode the data
	# auto_encoded_train = auto_model() # is this a get?
	# auto_encoded_val = auto_model() # is this a get?

	# # are we sure that we are getting back dataloaders? maybe just call it on the original
	# # data, then pass that into a new loader

	# # Build lstm model
	# lstm_model = LSTMModel(10, 1, dropout=0.4)
	# lstm_model.fit(auto_encoded_train, auto_encoded_val, device, **model_params)
	# lstm_model.save_model(os.path.join(model_file_path, 'LSTMModel'))







