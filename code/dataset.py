from __future__ import print_function

import os
import numpy as np
import random
import pickle
from scipy.misc import toimage

class Dataset:
	def __init__(self, dataset = "mnist"):
		self.dataset = dataset
		self.num_classes = 10
		self.num_channels = 1
		self.height = 28
		self.width = 28

		self.datapath = "../mnist/"

		#self.data_train = torchfile.load(os.path.join(self.datapath, 'train_28x28.th7'))
		#self.data_test = torchfile.load(os.path.join(self.datapath, 'test_28x28.th7'))

		self.data = {}
		self.labels = {}

		"""
		self.data["train"] = self.data_train["data"]
		self.labels["train"] = self.data_train["labels"]

		self.data["test"] = self.data_test["data"]
		self.labels["test"] = self.data_test["labels"]
		"""

		self.data = pickle.load(open("data/data.pkl", "r"))
		self.labels = pickle.load(open("data/labels.pkl", "r"))


		self.data["train"] = self.normalize(self.data["train"])
		self.data["test"] = self.normalize(self.data["test"])

		#print(self.data["train"].shape)
		#print(self.data["test"].shape)

		self.sorted_train = self.sortDataset(self.data["train"], self.labels["train"])
		self.sorted_test = self.sortDataset(self.data["test"], self.labels["test"])


	def getBatch(self, batch_size, data_type="train"):
		if data_type == "train":
			dataset = self.sorted_train
		else:
			dataset = self.sorted_test

		sids = np.random.randint(low=0, high=10, size=(3, batch_size))
		sids[1] = sids[0]
		
		data = np.zeros(shape=(3, batch_size, self.num_channels, self.height, self.width), dtype=np.float32)

		for i in xrange(3):
			for j in xrange(batch_size):
				temp = dataset[sids[i][j]]
				temp = temp[random.randrange(0,temp.shape[0])]
				
				data[i][j] = temp

		data = np.reshape(data, (-1, self.num_channels, self.height, self.width))
		sids = np.reshape(sids, (-1, 1))

		return data, sids


	def getBatchDiscriminator(self, batch_size, data_type="train"):
		if data_type == "train":
			dataset = self.sorted_train
		else:
			dataset = self.sorted_test

		sid = {}
		sid[0] = random.randrange(0,10)
		sid[1] = random.randrange(0,10)

		while sid[1]==sid[0]:
			sid[1] = random.randrange(0,10)

		sids = np.random.randint(low=0, high=10, size=(2, batch_size))
		
		data = np.zeros(shape=(2, batch_size, self.num_channels, self.height, self.width), dtype=np.float32)

		for i in xrange(2):
			for j in xrange(batch_size):
				temp = dataset[sid[i]]
				temp = temp[random.randrange(0,temp.shape[0])]
				
				data[i][j] = temp
				sids[i][j] = sid[i]

		data_1 = data[0,:,:,:,:]
		sids_1 = sids[0,:]

		data_2 = data[1,:,:,:,:]
		sids_2 = sids[1,:]

		return data_1, sids_1, data_2, sids_2


	def normalize(self, data):
		minn = data.min(axis=0)
		maxx = data.max(axis=0)
		data = (data - minn)/((maxx - minn + 1e-5)/2) - 1
		return data


	def sortDataset(self, data, labels):
		n_samples_per_class = {}

		for i in xrange(0, 10):
			n_samples_per_class[i] = 0

		for i in labels:
			n_samples_per_class[i-1] += 1

		dataset = {}
		for i in xrange(0, 10):
			dataset[i] = np.zeros(shape=(n_samples_per_class[i], self.num_channels, self.height, self.width), dtype=np.float32)
			#print(dataset[i].shape)

		for i in xrange(labels.shape[0]):
			label = labels[i] - 1
			sample = data[i]

			dataset[label][n_samples_per_class[label] - 1] = sample
			n_samples_per_class[label] -= 1

		return dataset


	def sortDatasetMod(self, data, labels):
		n_samples_per_class = []

		for i in xrange(10):
			n_samples_per_class.append(0)

		for i in labels:
			n_samples_per_class[i - 1] += 1

		dataset = []
		for i in xrange(10):
			dataset.append(np.zeros(shape=(n_samples_per_class[i], self.num_channels, self.height, self.width), dtype=np.float32))
			#print(dataset[i].shape)

		for i in xrange(labels.shape[0]):
			label = labels[i] - 1
			sample = data[i]

			dataset[label][n_samples_per_class[label] - 1] = sample
			n_samples_per_class[label] -= 1
		

def main():
	dataset = Dataset()
	data, sids = dataset.getBatch(16, data_type="train")
	data_1, sids_1, data_2, sids_2 = dataset.getBatchDiscriminator(16, data_type="train")

	print(sids_1)
	print(sids_2)
	

	"""
	for i in xrange(3):
		temp = np.reshape(data[i][0], (28,28))
		img = toimage(temp)
		img.save('my_{}.png'.format(i))
		print(sids[i][0])
	"""

if __name__ == '__main__':
	main()