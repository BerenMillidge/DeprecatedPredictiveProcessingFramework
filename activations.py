
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def tanh(x):
	return np.tanh(x)
def tanhderiv(x):
	return 1 - np.square(np.tanh(x))

def linear(x):
	return x

def linearderiv(x):
	return x

def sigmoid(x):
	return 1 / (1 + np.exp(-1*x))

def sigmoidderiv(x):
	return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
	if x <= 0:
		return 0
	else:
		return x

def reluderiv(x):
	if x <=0:
		return 0
	return 1	


def clipped_tanh(x, baseline=0.1, maxval=10):
	if baseline == 0:
		baseline = 1e-8
	if x < 0:
		return baseline * np.tanh(x / baseline)
	if x >= 0:
		return (maxval - baseline) * np.tanh(x / (maxval - baseline))

def clipped_tanhderiv(x, baseline=0.1, maxval=10):
	if baseline == 0:
		baseline = 1e-8
	if x < 0:
		return tanhderiv(x / baseline)
	if x >=0:
		return tanhderiv(x/ (maxval - baseline))


def clip(x, low=0, high=1):

	if len(x.shape) == 1 or x.shape[1] == 1:
		for i in xrange(len(x)):
			if x[i] < low:
				x[i] = low
			if x[i] > high:
				x[i] = high
	if len(x.shape) == 2:
		h,w = x.shape
		for i in xrange(h):
			for j in xrange(w):
				if x[i][j] < low:
					x[i][j] = low
				if x[i][j] > high:
					x[i][j] = high
	else:
		raise ValueError('Shape not implemented/recognised here')
	return x

	

def clipderiv(x, low=0, high=1):
	return clip(x, low=low, high=high)

def normalize_activations(x, eps = 1e-7):
	if type(x) is not 'numpy.ndarray':
		raise TypeError('Input must be a numpy vector')

	mu = np.mean(x)
	var = np.var(x)
	return (x - mu) / np.sqrt(var + eps)

func_dict = {'tanh': [tanh, tanhderiv],
			'linear': [linear, linearderiv],
			'sigmoid': [sigmoid, sigmoidderiv],
			'relu': [relu, reluderiv],
			'clip': [clip, clipderiv]
			}

def _get_func_dict():
	return func_dict

def parse_input_function(name):
	# just a big if function!
	if not isinstance(name, str):
		raise TypeError('Activation name input must be a string. You inputted type: ' + str(type(name)))

	if name not in func_dict.keys():
		raise ValueError('Activation function name not recognised. You inputted: ' + str(name) + '. Possible activation functions are ' + format_list(func_dict.keys()) + '.')
	return func_dict[name]


def plot_activation_function(func, minval=-5, maxval=5, num=100):
	xs = [i for i in range(minval, maxval, num)]
	vals = [func(x) for x in xs]
	fig = plt.figure()
	plt.plot(xs, vals)
	plt.xlabel("Input Value")
	plt.ylabel("Activation Value")
	plt.title("Effect of activatoin function")
	fig.tight_layout()
	plt.show()
	return fig
