import numpy as np
from utils import *

class ActivationsFilterLayer():

	def __init__(self, activations, filter_function, filter_params=None, train_function = None, train_params = None):
		self.activations = activations
		self.filter_function = filter_function
		self.filter_params = filter_params
		self.train_params = train_params
		self.train_function = train_function
		self.__trainable = False
		if train_function is not None and train_params is not None:
			self.__trainable = True


	def call(self):
		return filter_function(self.activations, self.filter_params)

	def train(self):
		self.filter_params = train_function(self.filter_params, self.train_params)
		return self.filter_params
