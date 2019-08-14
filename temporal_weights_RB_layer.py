from __future__ import division 
from activations import *
from initializers import *
import numpy as np
import collections

class TemporalWeightsRBLayer():

	def __init__(self, N_neurons, bottom_up_shape=None, internal_variance=1, top_down_variance=1, learning_rate=0.01,
				activation_function='sigmoid',initializer=None, weight_decay_rate = 0.002):

		self.N_neurons = N_neurons
		self.bottom_up_shape = bottom_up_shape
		self.internal_variance = internal_variance
		self.top_down_variance = top_down_variance
		self.learning_rate = learning_rate
		self.activation_function, self.activation_function_derivative = parse_input_function(activation_function)
		self.initializer = initializer or default_gaussian_initializer
		self.weight_decay_rate = weight_decay_rate

		#initialize the things
		if bottom_up_shape is not None:
			self.weights = self.initializer((self.bottom_up_shape, self.N_neurons))
			self.activations = self.initializer((self.N_neurons, 1))

		self.predictions = None
		self.prediction_errors = None
		self.top_down_predictions = None
		self._trainable =True
		self._callable = False
		self.losses = []
		self.loss = 0


	def _initialize_weights_activations(self):
		self.weights = self.initializer((self.bottom_up_shape, self.N_neurons))
		self.activations = self.initializer((self.N_neurons, 1))



	def on_epoch_begin(self):
		self.losses = []

	def on_epoch_end(self):
		self.loss =np.mean(np.array(self.losses))

	def on_training_begin(self):
		self.losses = []

	def on_training_end(self):
		pass

	def _gaussian_prior(self):
		return self.activations_prior_strength * np.sum(np.square(self.activations))

	def _gaussian_prior_deriv(self):
		return 2 * self.activations_prior_strength * self.activations

	def _kurtotic_prior(self):
		return self.activations_prior_strength * np.sum(np.log(1 + np.square(self.activations)))

	def _kurtotic_prior_deriv(self):
		return 2 * self.activations / (1 + np.square(self.activations))

	def gaussian_weight_prior(self):
		return self.weight_decay_rate * np.sum(self.weights)

	def gaussian_weight_prior_deriv(self):
		return self.weight_decay_rate * self.weights

	def calculate_prediction(self, activations):
		return self.activation_function(np.dot(self.weights, activations))

	def calculate_prediction_errors(self, bottom_up_input, predictions):
		bottom_up_input = np.reshape(bottom_up_input, (len(bottom_up_input),1))
		return bottom_up_input - predictions

	def calculate_activations(self, bottom_up_input):
		acts =  self.activation_function(np.dot(self.weights.T, bottom_up_input))
		return np.reshape(acts, (len(acts),1))

	def calculate_top_down_residual(self, activations, top_down_predictions):
		#print "in calculate top down residual"
		# I think/am sure this is the issue!
		return np.subtract(activations, top_down_predictions)

	def update_weights(self, prediction_errors, activations):
		update_rate = self.learning_rate / self.internal_variance
		# reshape activations
		wupdate = update_rate * np.dot(self.activation_function_derivative(prediction_errors), activations.T)
		#print "wupdate",  np.mean(wupdate)
		wprior = self.learning_rate * self.gaussian_weight_prior_deriv()
		#print "wprior", np.mean(wprior)
		return self.weights + wupdate - wprior

	def weighted_mean_incorporation(self, top_down_influence):
		return 1/2 * (1/self.internal_variance * self.activations) + (1/self.top_down_variance * top_down_influence)

	def incorporation_replace(self, top_down_influence):
		return top_down_influence

	def incorporate_top_down_influence(self, top_down_influence, incorporation_function=incorporation_replace):
		return incorporation_function(self, top_down_influence)

	def calculate_layer_loss(self):
		return np.dot(self.prediction_error.T, self.prediction_error) + np.dot(self.top_down_residual.T, self.top_down_residual)
	def forward_pass(self, inp):
		return self.calculate_activations( inp)

	def backward_pass(self, imp):
		return self.calculate_prediction(imp)

	def run(self, bottom_up_input, top_down_input,verbose=False):
		self.bottom_up_input = bottom_up_input
		self.top_down_predictions = top_down_input
		if self.top_down_predictions is None:
			self.top_down_predictions = np.zeros((len(self.activations),1))


		self.predictions = self.calculate_prediction(self.activations)
		#print "initial predictions ", self.predictions.shape

		self.prediction_error = self.calculate_prediction_errors(self.bottom_up_input, self.predictions)
		#print "prediction error ", self.prediction_error.shape
		self.activations = self.calculate_activations(self.bottom_up_input)
		#print "activations ", self.activations.shape 
		self.weights = self.update_weights(self.prediction_error, self.activations)
		#print "weights ", self.weights.shape 
		self.top_down_residual = self.calculate_top_down_residual(self.activations, self.top_down_predictions)
		#print "topdown residual " , self.top_down_residual.shape
		self.updated_prediction = self.calculate_prediction(self.activations)
		#print "updated predictions , ", self.updated_prediction.shape
		self.layer_loss = self.calculate_layer_loss()
		#print "layer loss , ", self.layer_loss.shape
		if verbose:
			print("initial predictions: ", np.mean(self.predictions))
			print("prediction error: " , np.mean(self.prediction_error))
			print("activations: ", np.mean(self.activations))
			print("weights: ", np.mean(self.weights))
			print("top down residual", np.mean(self.top_down_residual))
			print("updated prediction", np.mean(self.updated_prediction))
			print("layer loss: " , np.mean(self.layer_loss))

		return self.predictions, self.prediction_error, self.layer_loss,self.updated_prediction, self.top_down_residual

	def get_predictions(self):
		return self.predictions

	def get_prediction_errors(self):
		return self.prediction_errors

	def get_activations(self):
		return self.activations

	def get_representation_units(self):
		return self.predictions

	def get_weights(self):
		return self.weights

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

	def get_bottom_up_shape(self):
		return self.bottom_up_shape

	def get_N_neurons(self):
		return self.N_neurons

	def get_loss(self):
		return self.layer_loss

	# setters as well
	def _set_predictions(self, predictions):
		self.predictions = predictions

	def _set_prediction_errors(self, prediction_errors):
		self.prediction_errors = prediction_errors

	def _set_activations(self, activations):
		self.activations = activations

	def _set_weights(self, weights):
		self.weights = weights

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

	def _set_bottom_up_shape(self, bottom_up_shape):
		self.bottom_up_shape = bottom_up_shape



	def get_layer_info(self):
		info = collections.OrderedDict()
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.activations
		info['weights'] = self.weights
		info['internal_variance'] = self.internal_variance
		info['top_down_variance'] = self.top_down_variance
		info['learning_rate'] = self.learning_rate
		info['activation_function'] = self.activation_function
		info['activation_function_derivative'] = self.activation_function_derivative
		info['loss'] = self.layer_loss
		self.info = info
		return info

	def set_layer_info(self, info):
		if type(info) != dict:
			raise ValueError('Information must be a dict with keys as the attributes and the values as the values to set them to in the layer')

		for key, v in info:
			if hasattr(self, key):
				setattr(self, key, v)
			else:
				raise ValueError('Layer does not have attribute: ' + str(key))
		return

