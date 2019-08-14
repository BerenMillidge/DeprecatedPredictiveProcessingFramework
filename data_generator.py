
from __future__ import division
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from utils import *
import matplotlib.image as mpimg
import scipy.signal as signal
import scipy
from Exceptions import *
import os

def load_save_mnist(savename):
	(xtrain, xtest), (ytrain, ytest) = mnist.load_data()
	np.save(savename + '_xtrain', xtrain)
	np.save(savename + '_xtest', xtest)
	np.save(savename + '_ytrain', ytrain)
	np.save(savename + '_ytest', ytest)
	return xtrain, xtest, ytrain, ytest


#load_save_mnist('mnist')
def check_mnist_order():
	xtrain = np.load('mnist_xtrain.npy')
	ytrain = np.load('mnist_ytrain.npy')
	print xtrain.shape
	print ytrain.shape
	print ytrain[0]
	plt.imshow(xtrain[0])
	plt.show()
	print ytrain[1]
	plt.imshow(xtrain[1])
	plt.show()

def get_N_mnist_of_val(N, val, save_name=None):
	xtrain = np.load('mnist_xtrain.npy')
	ytrain = np.load('mnist_ytrain.npy')
	maxNum = len(xtrain)
	if N == -1:
		N = len(xtrain)
	i = 0
	arr = []
	while len(arr) < N and i < maxNum:
		if ytrain[i] == val:
			arr.append(xtrain[i])
		i = i+1
	a = np.array(arr)
	if save_name is not None:
		np.save(save_name, a)
	return a

def create_N_image_patches(img,N, shape, return_format):
	h,w = shape
	ih, iw= img.shape
	height = int(np.random.uniform(low=0+h, high=ih-h))
	width = int(np.random.uniform(low=0+(N*w), high=iw-(N*w)))
	if return_format == 'list':
		patches = []
		for i in xrange(N):
			patches.append(img[height:height+h, width+(i*w):width+((i+1)*w)])
		return patches
	if return_format == 'array':
		patches = img[height:height+h, width:width+w]
		for i in xrange(N-1):
			patch = img[height:height+h, width+((i+1)*w):width+((i+2)*w)]
			print patches.shape
			print patch.shape
			patches = np.concatenate(patches, patch)
		return patches


def create_N_image_patches(img, N, shape, return_format):
	h,w = shape
	h = h//2
	w = w//2
	ih, iw = img.shape
	patches = []
	#print img.shape
	if return_format == 'list':
		patches = []
		for i in xrange(N):
			rh = int(np.random.uniform(low=0+h, high=ih-h))
			rw = int(np.random.uniform(low=0+w, high=iw-w))
			patch = img[rh-h:rh+h, rw-w:rw+w]
			print patch.shape
			patches.append(patch)
		return patches
	if return_format == 'array':
		#rh = int(np.random.uniform(low=0+h, high=ih-h))
		#rw = int(np.random.uniform(low=0+w, high=iw-w))
		#patches = img[rh-h:rh+h, rw-w:rw+w]
		patches  = []
	#	print patches.shape
		for i in xrange(N-1):
			rh = int(np.random.uniform(low=0+h, high=ih-h))
			rw = int(np.random.uniform(low=0+w, high=iw-w))
			patch = img[rh-h:rh+h, rw-w:rw+w]
			patches.append(patch)
		patches= np.array(patches)
		#print patches.shape
		return patches



def create_image_patch_dataset(images, num_patches, patch_shape, return_format='array'):

	# check the return format is okay
	if return_format != 'array' and return_format != 'list':
		raise ValueError("Return format is not recognised. Possible formats are 'list', and 'array'.")
	# if type images is a str load them 
	if type(images) is str:
		if images.split('.')[-1] == 'npy':
			# thus it is an npy file 
			images = np.load(images)
		else:
			images = load(images)
	if len(images.shape)==2:
		#assume two dimensional
		return create_N_image_patches(images, num_patches, patch_shape, return_format=return_format)

	if len(images.shape)==3:
		patches = []
		for i in xrange(len(images)):
			patches.append(create_N_image_patches(images[i], num_patches, patch_shape, return_format=return_format))
		if return_format == 'array':
			dataset = []
			for image in patches:
				print "in image loop!"
				for patch in image:
					dataset.append(patch)
					print patch.shape
			patches =  np.array(dataset)
			print "ending in image patch dataset"
			print patches.shape
			return patches
		return patches 
	if len(images.shape) > 3 or len(images.shape)<=1:
		raise ValueError('Input data must be either 2 dimensional (a single image), or three dimensional (a list of images)')
		return None



def create_bar(image,start_width=None, start_height=0, bar_width = 5, bar_height=15):
	if start_width is None:
		start_width = image.shape[1]//2

	bar_extent = bar_width //2


	img = np.copy(image) # create a copy of the image
	img[start_height : start_height + bar_height, start_width-bar_extent : start_width + bar_extent] = np.full((bar_height, 2*bar_extent), 255.)
	return img

def create_bar_dataset(images, bar_length, bar_width =5):
	res = []
	for image in images:
		res.append(create_bar(image, bar_height = bar_length,bar_width = bar_width))
	return np.array(res)


def resize_dataset(dataset, newshape):
	d = []
	for i in xrange(len(dataset)):
		d.append(scipy.misc.imresize(dataset[i], newshape))
	return np.array(d)


def gaussian2D(x,y,sigma):
	return (1.0 /1 * np.pi * (sigma**2)) * np.exp(-(1.0/(2*(sigma**2))) * (np.square(x) + np.square(y)))

def receptiveFieldMatrix(func):
	h = 30
	g = np.zeros((h,h))
	for xi in range(0,h):
		for yi in range(0,h):
			x = xi - h / 2
			y = yi - h / 2
			g[xi, yi] = func(x,y)
	return g

def plotFilter(fun):
	g = receptiveFieldMatrix(fun)
	plt.imshow(g, cmap='gray')
	plt.show()

#plotFilter(lambda x,y: gaussian2D(x,y,4))

def DOG(x,y,sigma1, sigma2):
	return gaussian2D(x,y,sigma1) - gaussian2D(x,y,sigma2)
# next thing is difference of gaussians
# an alias!
def _mexicanHat(x,y, sigma1, sigma2):
	return DOG(x,y,sigma1, sigma2)

def oddGabor2D(x,y,sigma, orientation):
	return np.sin(x + orientation * y) * np.exp(-(x**2 + y**2)/(2*sigma))

def evenGabor2D(x,y,sigma, orientation):
	return np.cos(x + orientation * y) * np.exp(-(np.square(x) + np.square(y))/ (2*sigma))


def edgeEnergy(x,y,sigma, orientation):
	g_even = evenGabor2D(x,y, sigma, orientation)
	g_odd = oddGabor2D(x,y,sigma, orientation)
	return np.square(g_even) + np.square(g_odd)


def _applyFilter(img, func, func_params):
	return signal.convolve(img, receptiveFieldMatrix(lambda x, y: func(x,y, *func_params)), mode='same')

def filterImage(img, func, func_params):
	if isinstance(img, str):
		# assume it's an image
		if img.split('.')[-1] == 'npy':
			# it's an npy file
			img = np.load(img)
		else:
			img = load(img)
	if isinstance(img, list):
		img = np.array(list)

	if isinstance(img, np.ndarray):
		l = len(img.shape)
		if l ==3:
			# assume list of images
			imglist = []
			for i in xrange(len(img)):
				imglist.append(_applyFilter(img[i], func, func_params))
			return np.array(imglist)
		if l == 2:
			return _applyFilter(img, func, func_params)
		else:
			raise TypeError('Image must either be 2D or 3D meaning a list of images. Current shape is: ' + str(img.shape))
	raise TypeError('Input type not recognised')

def mexican_hat(dataset, sigma1, sigma2):
	return filterImage(dataset, _mexicanHat, (sigma1, sigma2))

def center_surround(dataset, sigma1, sigma2):
	return mexican_hat(dataset, sigma1, sigma2)

def create_RB_data(fname, num_images, sigma1, sigma2, num_patches, patch_shape, save_name=None):
	if not isinstance(fname, str):
		raise TypeError('Data input filename must be a string')

	if fname.split('.')[-1] =='npy':
		imgs = np.load(fname)
	else:
		imgs = load(fname)
	print "in create RB data"
	print imgs.shape
	imgs = imgs[:,:,:,0]
	imgs = imgs[0:num_images]
	print imgs.shape
	# explicitly mexican hats the input data
	filt = mexican_hat(imgs, sigma1, sigma2)
	print filt.shape
	patches =create_image_patch_dataset(filt, num_patches, patch_shape)
	if save_name is not None:
		np.save(save_name, patches)
	return patches

def create_sequential_mnist_dataset(save_name=None):
	# okay, just hack this out
	splits = []
	for i in xrange(10):
		splits.append(get_N_mnist_of_val(-1, i))
	# okay, now create the dataset
	lengths = []
	for split in splits:
		lengths.append(len(split))
	l = np.min(np.array(lengths))
	dataset = []
	for i in xrange(l):
		for j in xrange(10):
			dataset.append(splits[j][i])

	if save_name is not None:
		np.save(save_name, dataset)

	return np.array(dataset)

def load_dataset(fname, flatten=False, normalise=False):
	try:
		dataset = np.load(fname)
	except Exception as e:
		raise DatasetException('Loading dataset failed: , ' + str(e))

	if normalise:
		dataset = normalize_dataset(dataset)
	if flatten:
		dataset = flatten_dataset(dataset)

	return dataset

def mnist_sequential(path='datasets/mnist_sequential'):
	loaded = False
	try:
		dataset = np.load(path)
		loaded = True
		return dataset
	except Exception as e:
		pass

	if not loaded:
		# then create from scratch and save to path
		dataset = create_sequential_mnist_dataset()
		pathsplits = path.split('/') # assumes a linux directory
		if len(pathsplits >1):
			pathname = ""
			for i in xrange(len(pathsplits)-1):
				pathname += pathsplits[i]
			if not os.path.exists(pathname):
				os.makedirs(pathname)
			np.save(pathname+ '/' + pathsplits[-1], dataset)
		else:
			np.save(pathname, dataset)
	return dataset
