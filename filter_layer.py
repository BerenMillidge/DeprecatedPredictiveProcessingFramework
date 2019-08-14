from __future__ import division
import numpy as np
from utils import *

class FilterLayer():

	def __init__(self, input_data, filter_function, function_params, train_function = None, train_function_params=None):

		#assume input data is either data or a generator
		if type(input_data) == 'object' and not hasattr(input_data, 'next'):
			raise TypeError('If an object the input data must be a data generator with a next function')
		self.input_data  = input_data
		if type(filter_function) is not 'function':
			raise TypeError('Filter function provided must be a function; you provided: ' + str(type(filter_function)))
		self.filter_function = filter_function
		self.function_params = function_params
		self.__trainable = False
		if train_function is not None:
			if type(train_function) is not 'function':
				raise TypeError('Train function provided must be a function; you provided: ' + str(type(train_function)))
			self.train_function = train_function
			self.__trainable = True
		self.train_function_params = train_function_params


	def _next_data(self):
		if type(self.input_data) == 'object':
			return self.input_data.next()
		return self.input_data

	def call(self):
		# this applies the input functions to the data
		return filter_function(self._next_data() self.function_params)

	def train(self):
		if train_function is not None and train_function_params is not None:
			self.train_function_params = self.train_function(self._next_data(), self.train_function_params)
			return self.train_function_params
		else:
			pass

	def get_layer_info(self):
		info = {}
		info['input_data'] = self.input_data
		info['filter_function'] = self.filter_function
		info['function_params'] = self.function_params
		info['train_function'] = self.train_function
		info['train_function_params'] = self.train_function_params
		return info

