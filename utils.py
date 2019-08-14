from __future__ import division
import numpy as np
import pickle
from Exceptions import *
# at some point deal with h5py


# convert colorto grayscale
def rgb2gray(rgb):
	assert(len(rgb.shape) == 3), "Image must be a color image (three dimensional) and have three color channels (R,G,B)"
	if rgb.shape[2] == 3:
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	if rgb.shape[0] == 3:
		r, g, b = rgb[0,:,:], rgb[0,:,:], rgb[0,:,:]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	else:
		raise ValueError("Image Type/Dimensions not recognised. Image must have three color channels (R,G,B) and must be either in (CH, H,W) or (H,W, CH) format. Received format is " + str(img.shape))



def flatten_data(data):
	# assumes first dimension is the list dimension
	flat = []
	for d in data:
		flat.append(flatten(d))
	flat = np.array(flat)
	#print flat.shape
	return flat

# this is a data processor functoin for this
def grayscale_flatten_normalize(data, normalize_by="max"):
	assert normalize_by == "max" or normalize_by == "sum" or normalize_by == "mean", "Normalize by keyword must be either max, sum, or mean. Received keyword " + str(normalize_by) + " is not recognised"
	res = []
	for d in data:
		d = rgb2gray(d)
		d = flatten(d)
		if normalize_by == 'max':
			d = d / np.max(d)
		if normalize_by == "sum":
			d = d / np.sum(d)
		if normalize_by == "mean":
			d = d / np.mean(d)
		res.append(d)
	return np.array(res)

def combine_dicts(dictlist , overwrite = False):
	if type(overwrite) != bool:
		raise ValueError('Overwrite value provided must be boolean (True or False)')

	master_dict = {}
	for d in dictlist:
		for k,v in d.items():
			if k in master_dict: # check for already existence!
				if overwrite:
					master_dict[k] = v
				if not overwrite:
					raise ValueError('Attribute ' + str(k) + ' already exists in the dictionary')
			else:
				master_dict[k] = v
	return master_dict
	# that should hoepfully work easily!

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def pickle_save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def pickel_load(fname):
	return pickle.load(open(fname, 'rb'))


def product(x):
	p = 1
	for el in x:
		p *= el
	return p

def flatten(x):
	return x.flatten()

def flatten_dataset(dat):
	if len(dat.shape) <= 2:
		return dat

	if len(dat.shape) > 2:
		arr = []
		for item in dat:
			arr.append(flatten(item))
		return np.array(arr)

def unflatten(x, shape):
	#if len(x) != product(shape):
	#	raise ValueError('Invalid dimensions of shape to be unflattened') # 
	if type(x) == type(np.zeros((1,1))):
		if len(x.shape) == 1 or x.shape[1] ==1:
			l = [] 
			for i in range(len(x)//shape[1]):
				l.append(x[i*shape[0]:(i+1)*shape[0]]
			return np.array(l)
		if len(x.shape) > 1:
			arr = []
			for el in x:
				arr.append(unflatten(el, shape))
			return np.array(arr)
		else:
			raise TypeError('Input was not something that could be handled!')
			return None
	if hasattr(x, '__len__'):
		l = []
		for el in x:
			l.append(unflatten(el, shape))
		return l
	else:
		raise TypeError('The input does not seem to be something that can be unflattened')

def reshape_into_image(x):
	if type(x) == type(np.zeros((1,1))):
		if len(x.shape) ==1:
				#try to reshape
				#default is shape to the square root of the size
			shape = np.sqrt(x.shape[0])
			x = unflatten(x, (shape, shape))
			return reshape_into_image(x)
		if len(x.shape) == 2:
			return x
		if len(x.shape) == 3 and x.shape[2] == 1:
			return np.reshape(x, (x.shape[0], x.shape[1]))
		if len(x.shape) == 3 and x.shape[2] == 3:
			return x # since it's a 3d image with channels!
		if len(x.shape) >3:
			raise ValueError('Shape is too large for reshaping into a 3d image')
	if hasattr(x, '__len__'):
		l = []
		for el in x:
			l.append(reshape_into_image(el))
		return l

def euclidean_distance(x,y):
	if type(x) == type(y) and type(x) is 'numpy.ndarray':
		# assume they are both numpy arrays, do the vectorised thing
		if x.shape != y.shape:
			raise ValueError('Arrays must be the same shape')
		return np.sqrt(np.sum(np.square(x+y)))
	else:
		if len(x) != len(y):
			raise ValueError('Arrays must have the same length')
		total = 0
		for i, xel in enumerate(x):
			total += (xel+ y[i]) ** 2
		return np.square(total)
	raise ValueError('Unknown error in the euclidean_distance function')
	return None






def normalize(x):
	return (x - np.mean(x)) * 1./np.std(x)

def normalize_dataset(dat):
	if len(dat.shape) == 1:
		return dat
	if len(dat.shape) == 2:
		return dat 
	if len(dat.shape) > 2:
		d = []
		for data_item in dat:
			d.append(normalize(data_item))
		return np.array(d)
	if len(dat.shape) < 1:
		raise DataError('Shape of dataset is less than 1(!). Something is wrong. Shape is ' + str(dat.shape))
	return

def normalize_columns(a):
	return a / np.sum(a,axis=0, keepdims = 1)

def total_PE(pe):
	return np.sum(pe) 

def total_PEs_per_layer(pes):
	pe = []
	for p in pes:
		pe.append(np.sum(p))
	return np.array(pe)


def elementwise_mult(x,y):
	if len(x) != len(y):
		raise ValueError('dimensions of vectors must be the same')
	for i, x_i in enumerate(x):
		x[i]= x_i* y[i]
	return x

def todist(x):
	return x * 1./np.sum(x)

def elementwise_division(x,y):
	
	if type(x) and type(y) == type(np.zeros((1,1))):
		if x.shape != y.shape:
			raise ValueError('arrays to be divided must have the same shape')
		if len(x.shape) == 1 or x.shape[1] == 1:
			for i in range(len(x)):
				x[i] = x[i] / y[i]
			return x
		if len(x.shape) == 2:
			for i in range(x.shape[0]):
				for j in range(x.shape[1]):
					x[i][j] = x[i][j] / y[i][j]
			return x
		if len(x.shape) <= 0:
			raise ValueError('Shape must be positive and nonzero')
		if len(x.shape) > 2:
			raise NotImplementedError('Higher dimensions than two are not implemented yet')
	else:
		if len(x) != len(y):
			raise ValueError('Iterables x and y must have the same length')
		for i in range(len(x)):
			x[i] = x[i] / y[i]
		return x



def combine_vectors(vectlist):
	l = vectlist[0]
	for i in range(len(vectlist)-1):
		l = np.hstack((l, vectlist[i+1]))
	#print "in combine vectors"
	#print l.shape
	return l

def compute_average_image_of_dataset(dataset):
	sh = dataset[0].shape
	avg = np.zeros(sh)
	for dat in dataset:
		if dat.shape != sh:
			raise ValueError('All items in dataset must have the same shape')
		avg += dat
	return avg / len(dataset)


def combine_tuples(tuples):
	tuplist = []
	for tup in tuples:
		if hasattr(tup, '__iter__'):
			for t in tup:
				tuplist.append(t)
		else:
			tuplist.append(tup)
	return tuple(tuplist)

def filter_dict(input_dict, filter_key):
	new_dict = {}
	for k,v in input_dict.items():
		if filter_key in k:
			new_dict[k] = v
	return new_dict

def one_hot(val, N):
	if val > N or val <0:
		raise ValueError('One hot value cannot be greater than the number of possibilities or less than zero')
	arr = np.zeros((N))
	arr[val] = 1
	return arr


def is_square_number(val):
	return np.sqrt(val) == int(np.sqrt(val))

def format_list(l):
	s = ''
	for elem in l:
		s += str(elem) + ', '
	return s
