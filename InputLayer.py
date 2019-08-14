from __future__ import division
import numpy as np

class BaseInputLayer():

	def __init__(self, name,input_dim, bottom_up_projections):
		self.name = name
		self.input_dim = input_dim
		self.bottom_up_projections = bottom_up_projections
		self.input_data = None
		self._layer_type = 'Input'
		self._trainable=  False

	def feed(self, input_data):
		if isinstance(input_data, numpy.ndarray) or isinstance(input_data, list):
			self.input_data = input_data
		elif hasattr(input_data, next):
			# so it's a dataprovider object
			self.input_data = input_data.next()
		else:
			raise TypeError('Type of input data not recognised in input layer: ' + name + '. You inputted type: ' + str(type(input_data)) + '. Input data must be a numpy array, a list, or data provider object with a next() method.')

	def call(self):
		return self.input_data

	def _get_bottom_up_output(self):
		return self.call()

	def _get_top_down_output(self):
		return None

	def _get_bottom_up_projections(self):
		return self.bottom_up_projections

	def _get_top_down_projections(self):
		return None

class InputLayer(BaseInputLayer):

	def __init__(self, name, bottom_up_projections, normalize=True):
		self.name = name
		self.bottom_up_projections = bottom_up_projections
		self.normalize = normalize

	def _normalize_data(self):
		if isinstance(self.input_data, np.ndarray):
			return (1/np.var(self.input_data)) * (self.input_data - np.mean(input_data))
		return self.input_data


	def call(self):
		return self._normalize_data()



