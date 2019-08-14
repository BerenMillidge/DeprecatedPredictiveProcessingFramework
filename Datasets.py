import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import keras

class Dataset():

	def __init__(self, data):
		self.data = data
		self.current_index =0

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		
		return self.data[i]

	def next(self):
		self.current_index +=1
		return self.__getitem(self.current_index)

	def shuffle(self):
		return np.random.shuffle(self.data)

	def unsqueeze(self):
		return np.reshape(self.data, (1, self.data.shape))

	def extend(self):
		return np.reshape(self.data, (self.data.shape,1))

	def normalize(self, batched=True):
		if batched:
			d = []
			for i in xrange(len(self.data)):
				d.append(self.data[i] / np.sum(self.data[i]))
			return np.array(d)
		else:
			return self.data / np.sum(self.data)

	def whiten(self, batched=True):
		if batched:
			d  =[]
			for i in xrange(len(self.data)):
				whit = (self.data[i] - np.mean(self.data[i])) / np.std(self.data[i])
				d.append(whit)
			return np.array(d)
		else:
			return (self.data[i] - np.mean(self.data[i])) / np.std(self.data[i])

	def resize(self, newshape, batched=True, interpolation=None):
		if batched:
			d = []
			for i in xrange(len(self.data)):
				img = scipy.ndimage.imresize(d[i], newshape, interpolation = interpolation)
				d.append(img)
			return np.array(d)
		else:
			return scipy.ndimage.imresize(d, newshape, interpolation = interpolation)



	def split(self):
		pass # figure out how to do this


class MNISTDataset(Dataset):

	def __init__(self, save_name=None):



		if isinstance(save_name, str):
			try:
				self.xtrain = np.load(save_name + "_xtrain.npy")
				self.xtest = np.load(save_name + "_xtest.npy")
				self.ytrain = np.load(save_name + "_ytrain.npy")
				self.ytest = np.load(save_name + "y_test.npy")
			except Exception as e:
				pass

		if not self.xtrain:
			print("Loading data from keras")
			(self.xtrain, self.ytrain), (self.xtest, self.ytest) = keras.datasets.mnist.load_data()



	def xtrain(self):
		return self.xtrain
	def xtest(self):
		return self.xtest

	def ytrain(self):
		return self.ytrain
	def ytest(self):
		return self.ytest
