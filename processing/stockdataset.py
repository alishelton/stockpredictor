import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __getitem__(self, idx):
		return self.data[idx], self.targets[idx]


	def __len__(self):
		return len(self.data)

