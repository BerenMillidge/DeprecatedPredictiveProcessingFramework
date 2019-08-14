import numpy as np
from utils import *


class LPC_Layer():

	def __init__(self, data_generator, epochs, learning_rate= 0.015, optimize_function = None, optimize_functions = None):

		self._check_data_generator(data_generator) # check the data generator
		self.data_generator = data_generator
		self.learning_rate = learning_rate
		self.epochs = epochs 
		self.function_dict = self.initialize_function_dict()

		if optimize_functions is not None:
			if type(optimize_functions) is 'function':
				self.add_optimize_function(optimize_functions)
			if type(optimize_functions) is 'list':
				for func in optimize_functions:
					if type(optimize_functions) is 'function':
						self.add_optimize_function(optimize_functions)
			if type(optimize_functions) is 'dict':
				self.function_dict = combine_dicts((self.function_dict, optimize_functions))
			else:
				raise TypeError('Type of optimize functions is not recognised. You inputted '
								 + str(type(optimize_functions)) + 
								 '; requires a function, a list of functions, or a dictionary of funtions and' +
								 ' their names')


		if optimize_function in self.function_dict and type(optimize_function) is 'function':
			self.optimize_function = optimize_function
		else:
			if type(optimize_function) is 'function':
				self.function_dict[optimize_function.__name__] = optimize_function
				self.optimize_function = optimize_function
			else:
				raise TypeError('Optimize function must be a function; you gave a : ' + str(type(optimize_function)))

		self.__transform = False
		self.__trainable = True

	def _check_data_generator(self, data_generator):
		if not hasattr(data_generator, 'next') and type(data_generator.next) is 'function':
			#make sure the data generator has a next functino
			raise TypeError('Data generator must have a next function giving the next datapoint in the time series')

	def add_optimize_function(function, name=None):
		if type(function) is not 'function':
			raise TypeError('Optimization function added must be a function. You inputted type ' + str(type(function)) + '.')
		self.function_dict[name or function.__name__] = function


	def _initialize_function_dict(self):
		pass

	def run(self, learning_rate = None, data=None, epoch=None):
		if learning_rate is not None:
			if type(learning_rate) == 'float' or type(learning_rate) == 'int':
				self.learning_rate = learning_rate
			else:
				raise TypeError('Type of learning rate is incorrect. Must be a float or integer. You gave a: ' + str(type(learning_rate)))
		if data is not None:
			self._check_data_generator(data)
			self.data_generator = data

		if epochs is not None:
			if type(epochs) != 'int':
				raise TypeError('Epochs must be an integer. You inputted: ' + str(type(epochs)))
			self.epochs = epochs

		#now run the thing for each epoch!
		for i in xrange(self,epochs):
			data = self.data_generator.next()
			self.currents = self.optimize_function(data, self.currents)
		return self.currents
	



