import numpy as np
import pickle
from utils import *
import scipy.ndimage
import scipy.misc
from Exceptions import *

def isDataProcessor(obj):
	return hasattr(obj, "process")



class BaseDataPipeline(object):

	def __init__(self, data, stages):
		self.data = data
		self.stages = stages


	def process(self):
		for stage in stages:
			# verify it is a valid object
			if callable(obj):
				self.data = obj(self.data)
			if isDataProcessor(obj):
				self.data = obj.process(self.data)
			else:
				raise AttributeError("Stage of data processing pipeline must be either a data processor object or a functoin to apply to data. Found: " + str(type(obj)))

	def get_data(self):
		return self.data

	def save(self, save_name):
		pickle_save(save_name, self.data)

class BaseDataProcessor(object):

	def __init__(self, data, funs_plus_args= None, init_args=None):

		self.data = data
		self.functions_plus_args = functions_plus_args
		self.init_args = init_args
		self.results = None

	def initialize(self, args):
		self.init_args = args

	def process(self, args):
		self.results = data
		return data 

	def save(self, save_name, results=None):
		if not results:
			pickle_save(save_name, self.results)
		else:
			pickle_save(save_name, results)


class Spratling2016DataPreprocessor(BaseDataProcessor):
	# A data processor to process and cluster the data in accordance with the paper: https://link.springer.com/article/10.1007/s12559-016-9445-1

	def __init__(self, data, blur_sigma = 4, padding_coefficient = 1, resize_sizes = [], ZMNNC_k = 0, ZMMNC_L = 0):
		self.data = data
		self.blur_sigma = blur_sigma
		self.padding_coefficient = padding_coefficient
		self.resize_sizes = resize_sizes
		self.ZMMNC_L = ZMMNC_L
		self.ZMNNC_k = ZMNNC_k

		def _process_img(self, img):
			h,w = img.shape
			padding_px = self.blur_sigma * self.padding_coefficient
			img = mirror_padding(img,padding_px)
			img = gaussian_blur(img, self.blur_sigma)
			#crop back to original size to remove padding!
			img = img[padding_px:h+padding_px, padding_px:w+padding_px]
			on, off = on_off_channels(img)


	def process(self, data=None):
		if data:
			self.data = data

		if not isinstance(data, np.ndarray):
			raise ValueError("Data must be in numpy array format, for now!")
		sh = data.shape
		if len(sh) == 2:
			on, off =  _process_img(data)
			return np.reshape(on, (1, on.shape)), np.reshape(off, (1, off.shape))

		if len(sh) == 3:
			ons = []
			offs = []
			for img in data:
				on, off = _process_img(img)
				ons.append(on)
				offs.append(off)
			return np.array(ons), np.array(offs)

		else:
			raise DimensionException("Dimensions of data not recognised. Input must either be two dimensional - a single image, or three dimensional - a list of images. This data is: " + str(sh))

			




def gaussian_blur(img, sigma):
	return scipy.ndimage.filters.gaussian_filter(img, sigma)

def mirror_padding(img, padding_px):
	h,w = img.shape
	arr = np.zeros((h+(2*padding_px), w+ (2* padding_px)))
	print(img.shape)
	print(arr.shape)
	for i in range(padding_px):
		# top edge!
		arr[padding_px - i - 1 , padding_px : h + padding_px] = img[padding_px + i,0:h]
		# bottom edge!
		arr[padding_px + h + i, padding_px: h + padding_px] = img[h - i - 1, 0:h]
		# left edge
		#arr[padding_px : w + padding_px, padding_px - i - 1] = img[0:w, padding_px + i]
		#right edge
		#arr[padding_px: w + padding_px, padding_px + w + i] = img[0:w, w-i-1]

	return img


def on_off_channels(img):
	if len(img.shape != 2):
    	raise ValueError("Function only works for 2d inputs  TODO: Generalize later!")

	on = np.zeros(img.shape)
	off = np.zeros(img.shape)
	for index, val in np.ndenumerate(img):
		if val > 0:
			on[index] = val
		else:
			off[index] = val

	return on, off


if __name__ == '__main__':
	cat = scipy.misc.imread('cat.jpg')
	cat = cat[:,:,0]
	cat = scipy.misc.imresize(cat,(50,50))
	print(type(cat))
	print(cat.shape)
	pad = mirror_padding(cat, 4)
	print(type(pad))
	scipy.misc.imshow(pad)
