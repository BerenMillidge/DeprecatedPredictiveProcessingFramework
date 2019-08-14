import numpy as np 
from Exceptions import *

def default_gaussian_initializer(shape, mu=0, sigma=0.1):
	return np.random.normal(loc=mu, scale=sigma, size=shape)

def zeroes_initializer(shape):
	return np.zeros(shape)

def normalized_initializer(N):
	return np.random.randn(N) * np.sqrt(2/N)

def normalized_uniform_initializer(N):
	return np.array([1/N for i in xrange(N)])


class BaseInitializer(object):

	def __init__(self, initialize_func, dims):
		self.initialize_func = initialize_func
		self.dims = dims


	def initialize(self):
		try:
			return self.initialize_func(self.dims)
		except Exception as e:
			raise InitializerException(e)


class GaussianInitializer(BaseInitializer):

	def __init__(self, dims):
		self.initialize_func = default_gaussian_initializer
		self.dims = dims


class ZerosInitializer(BaseInitializer):

	def __init__(self, dims):
		self.initialize_func = zeroes_initializer
		self.dims = dims
