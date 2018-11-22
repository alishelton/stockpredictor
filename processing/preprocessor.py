import pandas as pd
import numpy as np
import pywt
from statsmodels.robust import mad

class PreProcessor():
	def __init__(self):
		self.data = {}
		self.targets = {}
		self.is_processed = {}

	def add_dataframe(self, name, dataframe):
		self.data[name] = dataframe
		self.is_processed[name] = False

	def read_data(self, name, filepath):
		try:
			data = pd.read_csv(filepath)
			if len(data) > 0:
				self.add_dataframe(name, data)
		except IOError:
			pass

	def get_data_with_name(self, name):
		return self.data.get(name, None), self.targets.get(name, None), self.is_processed.get(name, None)

	def get_data(self):
		return self.data, self.targets

	def preproccess(self, name):
		if not self.is_processed[name]:
			data = self.data[name].dropna()
			self.targets[name] = data['close'].shift(-1)[:-1].tolist()

			as_tensor = data.iloc[:-2, 1:].values
			transformed = pywt.wavedec(as_tensor, wavelet='haar', level=2, axis=0)
			reduced = [transformed[0]]
			reduced.extend([pywt.threshold(tensor, \
				value=mad(tensor) * np.sqrt(2 * np.log(tensor.shape[0])), mode='soft') for tensor in transformed[1:]])
			reconstructed = pywt.waverec(reduced, wavelet='haar', axis=0)

			self.data[name] = reconstructed
			self.is_processed[name] = True

	def preprocess_all(self):
		for name in self.data:
			self.preproccess(name)
