import numpy as np
from initializers import *
from BaseLayer import *

class SpratlingReviewLayer(BaseLayer):

	def __init__(self, bottom_up_dimension, layer_dimension, weights_initializer=default_gaussian_initializer, activations_initializer = default_gaussian_initializer, eps_error = 1e-4, eps_weights = 1e-4, update_weights=False, weights_learning_rate = 0.001, activations_learning_rate = 0.001):

		self.bottom_up_dimension = bottom_up_dimension
		self.layer_dimension = layer_dimension
		self.weights_initializer = weights_initializer
		self.activations_initializer = activations_initializer
		self.eps_error = eps_error
		self.eps_weights = eps_weights
		self.weights_learning_rate = weights_learning_rate
		self.activations_learning_rate = activations_learning_rate

		self.weights = weights_initializer([self.layer_dimension, self.bottom_up_dimension])
		self.activations = activations_initializer([self.layer_dimension,1])

		self.loss = 0

		self._trainable = True
		self._callable = False
		self.update_weights = update_weights

	def _compute_predictions(self):
		print("In cmopute predictoins")
		print(self.weights.shape)
		print(self.activations.shape)
		return self.eps_error + np.dot(self.weights.T, self.activations)

	def _compute_errors(self):
		return np.divide(self.bottom_up_dimension, self.predictions) 

	def _update_activations(self):
		self.activations += self.activations_learning_rate * ((self.eps_weights + self.activations) * np.dot(self.weights, self.errors))

	def _update_weights(self):
		self.weights += self.weights_learning_rate * np.dot(self.activations, self.errors.T)

	def _compute_loss(self):
		return np.dot(self.errors.T, self.errors)

	def run(self, bottom_up_input, top_down_input):
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input 
		self.predictions = self._compute_predictions()
		print("Spratling predictoin shape: ", self.predictions.shape)
		self.errors = self._compute_errors()
		self._update_activations()
		self._update_weights()
		self.loss = self._compute_loss()
		return self.activations, None, self.loss

	def get_representation_units(self):
		return self.activations
	def get_cause_units(self):
		return self.activations

	def get_activations(self):
		return self.activations

	def get_predictions(self):
		if self.predictions is None:
			return self._compute_predictions()
		return self.predictions

	def get_prediction_errors(self):
		if self.errors is None:
			return self._compute_errors()
		return self.errors

	def get_forward_weights(self):
		return self.weights


	def get_weights(self):
		return self.weights

	def get_loss(self):
		return self.loss

	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self.errors = prediction_errors

	def _set_error_units(self, prediction_errors):
		self.errors = prediction_errors

	def _set_representation_units(self, activations):
		self.activations = activations

	def _set_activations(self, activations):
		self.activations = activations

	def _set_forward_weights(self, weights):
		self.weights = weights

	def _set_weights(self, weights):
		self.weights = weights


	def get_layer_info(self):
		info = {}
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.errors
		info['activations'] = self.activations
		info['weights'] = self.weights
		return infos
