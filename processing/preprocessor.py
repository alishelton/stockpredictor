import pandas as pd
import numpy as np
import pywt
from pprint import pprint

class PreProcessor():
	def __init__(self):
		self.data = {}
		self.is_processed = {}

	def add_dataframe(self, name, dataframe):
		self.data[name] = dataframe
		self.is_processed[name] = False

	def read_data(self, name, filepath):
		data = pd.read_csv(filepath)
		self.add_dataframe(name, data)

	def get_data(self, name):
		return self.data[name], self.is_processed[name]

	def preproccess(self, name):
		if not self.is_processed[name]:
			data = self.data[name]
			as_list = data.iloc[:, 1:].values.tolist()
			transformed = pywt.wavedec(as_list, wavelet='haar', level=2, axis=1)
			self.data[name] = transformed
			self.is_processed[name] = True

	def preprocess_all(self):
		for name in self.data:
			self.preproccess(name)



