import numpy as np 
from utils import * 
from initializers import *
from BaseLayer import *

class Friston2003NoLateralInteractions(BaseLayer):

	def __init__(self, bottom_up_input_dim, top_down_input_dim, learning_rate =0.01, weights_initializer = default_gaussian_initializer, activations_initializer = default_gaussian_initializer):

		if not isinstance(bottom_up_input_dim, int):
			raise ValueError("Button up input dimension must be an interger")

		if not isinstance(top_down_input_dim, int):
			raise ValueError('Top down input dimension must be an integer')

		#if not isinstance(layer_dim, int):
		#	raise ValueError('Layer dimension must be an integer')

		self.bottom_up_input_dim = bottom_up_input_dim
		self.top_down_input_dim = top_down_input_dim
		self.layer_dim = bottom_up_input_dim 
		self.weights_initializer = weights_initializer
		self.learning_rate = learning_rate
		self.activations_initializer = activations_initializer

		self.weights = self.weights_initializer([self.layer_dim, self.top_down_input_dim])
		self.cause_units = self.activations_initializer([self.layer_dim,1])

		self._trainable = True
		self._callable = False



	def _calculate_top_down_predictions(self):
		if self.top_down_input is None:
			self.top_down_input = np.zeros([self.top_down_input_dim,1])
		return np.dot(self.weights, self.top_down_input) # 

	def _compute_upward_projection(self):
		return np.dot(self.weights.T, self.prediction_errors)

	def _compute_prediction_errors(self):
		return self.cause_units - self.top_down_predictions

	def _update_cause_units(self):
		#print("In update caues units")
		#print(self.bottom_up_input.shape)
		#print(self.prediction_errors.shape)
		self.cause_units -= self.learning_rate * (self.bottom_up_input + self.prediction_errors) 

	def _update_weights(self):
		self.weights -= self.learning_rate * (np.dot(self.prediction_errors, self.top_down_input.T))
		self.weights /= np.sum(self.weights)

	def _compute_loss(self):
		#print("In compute loss: mean predictino errors: ", np.mean(self.prediction_errors))
		return np.dot(self.prediction_errors.T, self.prediction_errors)

	def run_without_update(self, bottom_up_input, top_down_input):
		def run(self, bottom_up_input, top_down_input):
		self.bottom_up_input = bottom_up_input
		if len(self.bottom_up_input.shape) == 1:
			self.bottom_up_input = np.reshape(self.bottom_up_input, (len(self.bottom_up_input),1))


		self.top_down_input = top_down_input
		self.top_down_predictions = self._calculate_top_down_predictions()
		self.prediction_errors = self._compute_prediction_errors()
		self._update_cause_units()
		self.loss = self._compute_loss()
		return self._compute_upward_projection(), self.cause_units, self.loss

	def update(self):
		self._update_weights()

	def run(self, bottom_up_input, top_down_input):
		#print("In run function!")
		self.bottom_up_input = bottom_up_input
		if len(self.bottom_up_input.shape) == 1:
			self.bottom_up_input = np.reshape(self.bottom_up_input, (len(self.bottom_up_input),1))


		self.top_down_input = top_down_input
		self.top_down_predictions = self._calculate_top_down_predictions()
		self.prediction_errors = self._compute_prediction_errors()
		#print(self.prediction_errors)
		self._update_cause_units()
		self._update_weights()
		self.loss = self._compute_loss()
		#print("Weights normal layer: ", np.mean(self.weights))
		#print("activations normal layer: ", np.mean(self.cause_units))
		#print("Loss: ", self.loss)
		return self._compute_upward_projection(), self.cause_units, self.loss


	def get_representation_units(self):
		return self.cause_units
 
	def get_cause_units(self):
		return self.cause_units

	def get_activations(self):
		return self.cause_units

	def get_predictions(self):
		if self.top_down_predictions is None:
			return self._calculate_top_down_predictions()
		return self.top_down_predictions

	def get_prediction_errors(self):
		if self.prediction_errors is None:
			return self._compute_prediction_errors()
		return self.prediction_errors

	def get_forward_weights(self):
		return self.weights

	def get_weights(self):
		return self.get_forward_weights()

	def get_loss(self):
		#print("In get loss; loss = ", self.loss)
		return self.loss

	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self._set_error_units(prediction_errors)

	def _set_error_units(self, prediction_errors):
		self.prediction_errors = prediction_errors

	def _set_representation_units(self, activations):
		self.cause_units = activations

	def _set_activations(self, activations):
		self.cause_units = activations

	def _set_forward_weights(self, weights):
		self.forward_weights = weights


	def get_bottom_up_shape(self):
		return self.bottom_up_input_dim

	def calculate_prediction(self):
		return np.dot(self.weights.T, self.cause_units)

	def _initialize_weights_activations(self):
		pass

	def get_layer_info(self):
		info = {}
		info['predictions'] = self.cause_units 
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.cause_units
		info['forward_weights'] = self.weights
		info['activation_function'] = None
		return info

class Friston2003NoLateralInteractionsInputLayer(BaseLayer):

	def __init__(self, data_dim, top_down_dim, weights_initializer = default_gaussian_initializer, learning_rate =0.01):

		self.data_dim = data_dim
		self.top_down_dim = top_down_dim
		self.weights_initializer = weights_initializer
		self.weights = default_gaussian_initializer([self.data_dim, self.top_down_dim])
		self.cause_units = np.zeros([self.data_dim, 1])
		self.learning_rate = learning_rate

		self._trainable = True
		self._callable = False

	def _compute_predictions(self):
		if self.top_down_input is None:
			self.top_down_input = np.zeros([self.top_down_dim,1])
		return np.dot(self.weights, self.top_down_input)

	def _compute_upward_projection(self):
		return np.dot(self.weights.T, self.prediction_errors)

	def _compute_prediction_errors(self):
		return np.subtract(self.cause_units, self.top_down_predictions)

	def _update_weights(self):
		self.weights -= self.learning_rate * (np.dot(self.prediction_errors, self.top_down_input.T))
		self.weights /= np.sum(self.weights) 

	def _compute_loss(self):
		return np.dot(self.prediction_errors.T, self.prediction_errors)

	def run_without_update(self, bottom_up_input, top_down_input):
		self.top_down_input = top_down_input
		self.cause_units = bottom_up_input
		self.top_down_predictions = self._compute_predictions()
		self.prediction_errors = self._compute_prediction_errors()
		up = self._compute_upward_projection()
		self.loss = self._compute_loss()
		return up, self.cause_units, self.loss

	def update(self):
		self._update_weights()

	def run(self, bottom_up_input, top_down_input):
		#print("In input layer run func:")
		#print(top_down_input)
		self.top_down_input = top_down_input
		self.cause_units = bottom_up_input
		self.top_down_predictions = self._compute_predictions()
		self.prediction_errors = self._compute_prediction_errors()
		self._update_weights()
		up = self._compute_upward_projection()
		self.loss = self._compute_loss()
		#print("Loss, input layer: " , self.loss) 
		#print("predictions, input layer: " , np.mean(self.top_down_predictions))
		#print("upward, input layer", np.mean(up))
		#print("weights, input layer:" , np.mean(self.weights))
		#print("cause untis, activatoin, layer", np.mean(self.cause_units))
		##print("Upward projection shape: ", up.shape)
		return up, self.cause_units, self.loss

	def get_representation_units(self):
		return self.cause_units
 
	def get_cause_units(self):
		return self.cause_units

	def get_activations(self):
		return self.cause_units

	def get_predictions(self):
		if self.top_down_predictions is None:
			return self._calculate_top_down_predictions()
		return self.top_down_predictions

	def get_prediction_errors(self):
		if self.prediction_errors is None:
			return self._compute_prediction_errors()
		return self.prediction_errors

	def get_forward_weights(self):
		return self.weights

	def get_weights(self):
		return self.get_forward_weights()

	def get_loss(self):
		#print("In get loss; loss = ", self.loss)
		return self.loss

	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self._set_error_units(prediction_errors)

	def _set_error_units(self, prediction_errors):
		self.prediction_errors = prediction_errors

	def _set_representation_units(self, activations):
		self.cause_units = activations

	def _set_activations(self, activations):
		self.cause_units = activations

	def _set_forward_weights(self, weights):
		self.forward_weights = weights


	def get_bottom_up_shape(self):
		return self.bottom_up_input_dim

	def calculate_prediction(self):
		return np.dot(self.weights.T, self.cause_units)

	def _initialize_weights_activations(self):
		pass

	def get_layer_info(self):
		info = {}
		info['predictions'] = self.cause_units
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.cause_units
		info['forward_weights'] = self.weights
		info['activation_function'] = None
		return info


