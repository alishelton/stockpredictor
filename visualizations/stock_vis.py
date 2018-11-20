import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import PreProcessor

base_file_path = '/Users/alishelton/Desktop/deeplearning/stocks/'
data_file_path = os.path.join(base_file_path, 'stock_data')

if __name__ == '__main__':
	preprocessor = PreProcessor()
	preprocessor.read_data('FB', os.path.join(data_file_path, 'FB' + '.csv'))

	plt.subplot(211)
	data, targets, is_processed = preprocessor.get_data_with_name('FB')
	x = np.arange(len(data))
	sns.lineplot(x=x, y=data['close'])

	preprocessor.preprocess_all()
	plt.subplot(212)
	data, targets, is_processed = preprocessor.get_data_with_name('FB')
	data = data[:,2]
	x = np.arange(len(data))
	sns.lineplot(x=x, y=data)
	plt.show()

