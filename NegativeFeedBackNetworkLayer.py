# implementation of layer  described first in https://www.frontiersin.org/articles/10.3389/neuro.10.004.2008/full


from utils import *
from initializers import *
from activations  import *

class NegativeFeedbbackNetworkLayer(BaseLayer):

	def __init__(self, bottom_up_dimension, top_down_dimension, layer_dimension, weights_initializer = gaussian_initializer, activations_initializer = gaussian_initializer, top_down_learning_rate = 0.001, layer_learning_rate = 0.001, weights_learning_rate = 0.001):
		self.bottom_up_dimension = bottom_up_dimension
		self.top_down_dimension = top_down_dimension
		self.layer_dimension = layer_dimension
		self.weights_initializer = weights_initializer
		self.activations_initializer = activations_initializer
		self.weights_learning_rate = weights_learning_rate
		self.top_down_learning_rate = top_down_learning_rate
		self.layer_learning_rate = layer_learning_rate

		self.weights = self.weights_initializer(self.bottom_up_dimension, self.layer_dimension)
		self.activations = self.activations_initializer(self.layer_dimension)


	def _compute_prediction(self):
		return np.dot(self.weights.T, self.activations)

	def _compute_errors(self):
		return self.bottom_up_input - self.predictions

	def _update_activations(self):
		self.activations += self.layer_learning_rate * np.dot(self.weights, self.errors) + self.top_down_learning_rate * self.top_down_input

	def _update_weights(self):
		self.weights += self.weights_learning_rate * np.dot(self.activations, self.errors.T)

	def _compute_loss(self):
		return np.dot(self.errors, self.errors) # just the squared error!

	def run(self, bottom_up_input, top_down_input):
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input
		self.predictions = self._compute_prediction()
		self.errors = self._compute_errors()
		self._update_activations() 
		self._update_weights()
		return self.activations, self._compute_prediction(), self._compute_loss() 

	def get_representation_units(self):
		return self.activatiosn
	def get_cause_units(self):
		return self.activations

	def get_activations(self):
		return self.activations

	def get_prediction(self):
		if self.prediction is None:
			return self._compute_prediction()
		return self.prediction

	def get_prediction_errors(self):
		if self.errors is None:
			return self._compute_errors()
		return self.errors

	def get_forward_weights(self):
		return self.weights


	def get_weights(self):
		return self.weights

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
		return info
