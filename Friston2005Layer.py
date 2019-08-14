import numpy as np
from utils import *
from initializers import * 

class Friston2005Layer():

	def __init__(self, input_dimension=None, N_neurons = None, top_down_input_dimension = None, forward_weights_initializer=gaussian_initializer, lateral_weights_initializer=gaussian_initializer,activations_initializer=gaussian_initializer, errors_initializer=gaussian_initializer, activation_function=sigmoid):
		
		if input_dimension is None:
			assert N_neurons is not None, 'If not the first layer, needs a preset dimension'
		if N_neurons is not None:
			assert input_dimension is None, 'Set dimensions of the layers can only occur if it is not the input layer#
		self.input_dimension = input_dimension
		self.N_neurons = N_neurons
		self.top_down_input_dimension = top_down_input_dimension
		self.forward_weights_initializer = forward_weights_initializer
		self.lateral_weights_initializer = lateral_weights_initializer
		self.activations_initializer = activations_initializer
		self.errors_initializer = errors_initializer
		self.activation_function = activation_function
		
		self.input_dim = self.input_dimension or self.N_neurons
		self.forward_weights = self.forward_weights_initializer((input_dim, top_down_input_dimension))
		self.lateral_weights= self.lateral_weights_initializer((input_dim, input_dim))
		self.representation_units = self.activations_initializer((input_dim, 1))
		self.error_units = self.errors_initializer((input_dim, 1))

		self.lwinv = None 
		self.__trainable = True
		self.__tranform = False


	def calculate_prediction(self):
		return self.activation_function(np.dot(self.representation_units, self.forward_weights))

	def calculate_prediction_errors(self, top_down_input):
		error = self.representation_units - top_down_input
		# then add in the lateral inhibition effect
		return error - np.dot(self.lateral_weights, self.error_units)

	def _calculate_lwinv(self):
		if self.lwinv is None:
			return np.inv(1 + self.lateral_weights)
		return self.lwinv

	def update_representation_units(self):


	def update_forward_weights(self):
		lwinv = self._calculate_lwinv()
		crossprod = np.dot(self.error_units, self.representation_units.T)
		return np.dot(lwinv, crossprod)


	def update_lateral_weights(self):
		lwinv = self._calculate_lwinv()
		crossprod = np.dot(self.error_units, self.error_units.T) - 1.
		return np.dot(lwinv, crossprod)
		

	def get_representation_units(self):
		return self.representation_units

	#aliases
	def get_predictions(self):
		return self.calculate_prediction()

	def get_error_units(self):
		return self.error_units

	#alias:
	def get_prediction_errors(self):
		return self.get_error_units()

	def get_activations(self):
		return self.get_representation_units()

	def get_forward_weights(self):
		return self.forward_weights

	def get_lateral_weights(self):
		return self.lateral_weights

	def get_weights(self):
		return self.get_forward_weights(), self.get_lateral_weights()

	def get_bottom_up_input(self):
		return self.bottom_up_input or None

	# setters as well
	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self._set_error_units(prediction_errors)

	def _set_error_units(self, prediction_errors):
		self.error_units = prediction_errors

	def _set_representation_units(self, activations):
		self.representation_units = activations

	def _set_activations(self, activations):
		self.representation_units = activations

	def _set_forward_weights(self, weights):
		self.forward_weights = weights

	def _set_lateral_weights(self, weights):
		self.lateral_weights = weights


	def _set_activation_function(self, activation_function):
		self.activation_function = activation_function


	def get_layer_info(self):
		info = {}
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.error_units
		info['activations'] = self.representation_units
		info['forward_weights'] = self.forward_weights
		info['lateral_weights'] = self.lateral_weights
		info['activation_function'] = self.activation_function
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

	
