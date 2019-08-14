#Biased competition layer - ala spratling

import numpy as np
from utils import *
from initializers import *
# simplest kind of competition - normalizer filter
def softmax_filter(activations):
	return np.exp(activations) / np.sum(np.exp(activations))

def exponential_dropoff(distance):
	return np.exp(-1*np.abs(distance))

def surround_inhibition(activations, inhibition_radius, drop_off_function = exponential_dropoff):
	#assume activations is a vector
	if type(activations) is not 'numpy.ndarray' and len(activations.shape) != 2 and activations.shape[1] != 1:
		raise TypeError('Activations must be a numpy vector')
	for i in xrange(len(activations)):
		activation = activations[i]
		for j in xrange(inhibition_radius*2):
			pos = i - inhibition_radius + j
			if pos >= 0 or pos < len(activations) or pos != i:
				# then do the dropoff
				distance = pos - inhibition_radius
				activations[pos] -= drop_off_function(distance)*activations[i]
	return activations


# Desimone and Duncan (Corchs and Deco/ deco and rolls - classic linear BC layer)

class ClassicBCLinearLayer():

	def __init__(self, input_dim, N_neurons, weights_initializer, internal_variance, top_down_variance,learning_rate=0.01):
		self.input_dim =  input_dim
		self.N_neurons = N_neurons
		self.weights_initializer = weights_initializer
		self.internal_variance = internal_variance
		self.top_down_variance = top_down_variance
		self.learning_rate = learning_rate

		self.weights = weights_initializer((input_dim, N_neurons))
		self.activations = weights_initializer((N_neurons))

		self.predictions = None
		self.prediction_errors = None

	def calculate_prediction(self):
		self.predictions = np.dot(self.weights, self.activations)
		return self.predictions

	def calculate_prediction_errors(self, input_data):
		if self.predictions is None:
			self.predictions = self.calculate_prediction()
		self.prediction_errors = input_data - self.predictions
		return self.prediction_errors

	def update_activations(self, top_down_input):
		bottom_up = self.internal_variance * np.dot(self.weights.T, self.prediction_errors)
		top_down = top_down_variance * top_down_input
		self.activations = self.activations + bottom_up + top_down
		return self.activations

	def update_weights(self):
		self.weights = self.weighs + (self.learning_rate * np.dot(self.activations, self.prediction_errors.T))
		return self.weights

	def run(self, input_data, top_down_input):
		self.calculate_prediction_errors(input_data)
		self.update_activations(top_down_input)
		self.update_weights()
		return self.predictions, self.prediction_errors, self.activations, self.weights

