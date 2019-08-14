# Implmentation of Spratling Predictive-Coding-Divisive-Normalization layer
import numpy as np
from utils import * 
from initializers import *


class PC_DN_Layer():

	def __init__(self, input_dimension, N_neurons, learning_rate =1, weights_initializer=gaussian_initializer, eps1=1e-7, eps2=1e-7, internal_variance=1, top_down_variance=1,noise_variance =1, activation_function=linear):
		self.input_dimension = input_dimension
		self.N_neurons = N_neurons
		self.weights_initializer = weights_initializer
		if eps1 <= 0:
			raise ValueError('Eps1 needs to be nonzero and nonnegative')
		if eps2 <= 0:
			raise ValueError('Eps2 needs to be nonzero and nonnegative')
		self.eps1 = eps1
		self.eps2 = eps2
		self.internal_variance = internal_variance
		self.noise_variance = noise_variance
		self.top_down_variance = top_down_variance
		self.activation_function = activation_function
		self.activation_function_derivative = linearderiv


		self.forward_weights = self.weights_initializer((input_dimension, N_neurons))
		self.backward_weights = self.weights_initializer((N_neurons, input_dimension))

		#initialize activations
		self.activations = self.weights_initializer((N_neurons))

		self.predictions = None
		self.prediction_errors = None

		self.__transform__ = False
		self.__trainable__ = True


	def calculate_prediction(self):
		self.predictions = (np.dot(self.forward_weights, self.activations))
		return self.predictions

	def calculate_prediction_error(self, input_data):
		self.prediction_errors = elementwise_division(input_data, (eps2 + self.predictions))
		return self.prediction_errors

	#the update activations are necessary
	def update_activations(self, top_down_predictions):
		self.top_down_error = elementwise_division(self.activations, top_down_predictions)
		bottom_up_term = np.dot(self.backward_weights, self.prediction_errors)
		self.activations = elementwise_division(((eps1 + self.activations) * bottom_up_term), self.top_down_error)
		return self.activations

	def update_weights(self):
		pass 

	def run(self. bottom_up_input, top_down_input, learning_rate=None):
		if learning_rate is not None:
			self.learning_rate = learning_rate

		self.predictions = self.calculate_prediction()
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input
		self.prediction_errors = self.calculate_prediction_error(bottom_up_input)
		self.update_activations(self.top_down_input)
		self.update_weights()
		return self.predctions, self.prediction_errors, self.activations, self.forward_weights, self.backward_weights

	def get_predictions(self):
		return self.predictions

	def get_prediction_errors(self):
		return self.prediction_errors

	def get_activations(self):
		return self.activations

	def get_forward_weights(self):
		return self.forward_weights

	def get_backward_weights(self):
		return self.backward_weights

	def get_internal_variance(self):
		return self.internal_variance

	def get_noise_variance(self):
		return self.noise_variance

	def get_top_down_variance(self):
		return self.top_down_variance

	def get_top_down_error(self):
		return self.top_down_error or None

	def get_bottom_up_input(self):
		return self.bottom_up_input

	# setters as well
	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self.prediction_errors = prediction_errors

	def _set_activations(self, activations):
		self.activations = activations

	def _set_forward_weights(self, weights):
		self.forward_weights = weights

	def _set_backward_weights(self, weights):
		self.backward_weights = weights

	def _set_internal_variance(self, internal_variance):
		self.internal_variance = internal_variance

	def _set_noise_variance(self, noise_variance):
		self.noise_variance = noise_variance

	def _set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def _set_activation_function(self, activation_function):
		self.activation_function = activation_function

	def _set_activation_function_derivative(self, activation_function_derivative):
		self.activation_function_derivative = activation_function_derivative

	def _set_top_down_error(self, top_down_error):
		self.top_down_error = top_down_error

	def get_layer_info(self):
		info = {}
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.activations
		info['forward_weights'] = self.forward_weights
		info['backward_weights'] = self.backward_weights
		info['internal_variance'] = self.internal_variance
		info['noise_variance'] = self.noise_variance
		info['top_down_variance'] = self.top_down_variance
		info['learning_rate'] = self.learning_rate
		info['activation_function'] = self.activation_function
		info['activation_function_derivative'] = self.activation_function_derivative
		return info


	def set_layer_info(self, info):
		
		if type(info) != dict:
			raise ValueError('Information must be a dict with keys as the attributes and the values as the values to set them to in the layer')

		for key, v in info:
			if hasattr(self, key):
				getattr(self, key) = v
			else:
				raise ValueError('Layer does not have attribute: ' + str(key))
		return
	 # 

