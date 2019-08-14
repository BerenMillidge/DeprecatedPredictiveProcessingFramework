
from __future__ import division 
from activations import *
from initializers import *
import numpy as np
import collections
class RB_Layer():

	def __init__(self, N, bottom_up_input, activations_learning_rate=0.1, weights_learning_rate = 0.005,
				internal_variance=1, top_down_variance=1, top_down_predictions=None, 
				activation_function='sigmoid', noise_variance=0, initializer=None, 
				activations_prior_strength=1, weight_decay_rate=0.02, learning_rate = 0):
			
		self.N = N
		self.bottom_up_input = bottom_up_input
		self.internal_variance = internal_variance
		self.top_down_predictions = top_down_predictions
		self.top_down_variance = top_down_variance
		self.activation_function = activation_function
		self.activation_function, self.activation_function_derivative = parse_input_function(activation_function)
		self.initializer = initializer or default_gaussian_initializer
		self.learning_rate = learning_rate
		print(self.activation_function)
 
		
		self.weights = self.initializer((len(self.bottom_up_input), self.N))
		if len(self.bottom_up_input.shape) == 2:
			self.weights = self.initializer((self.bottom_up_input.shape[1], self.N))
		#self.weights = np.random.normal(loc=0, scale=0.1, size=(len(self.bottom_up_input), self.N))
		

		self.noise_variance = noise_variance

		self.activations_learning_rate = activations_learning_rate
		#self.weights_learning_rate = weights_learning_rate
		self.weights_learning_rate = learning_rate
		self.activations_prior_strength = activations_prior_strength
		self.weight_decay_rate = weight_decay_rate
		# initialise the activations similarly!
		self.activations = self.initializer((self.N, 1))
		#self.activations = np.random.normal(loc=0, scale=0.1, size=(N,1)) 
		self.predictions = None
		self.prediction_errors = None


		self._trainable = True

	def on_epoch_begin(self):
		#print "in layer on epoch begin"
		self.losses = []

	def on_epoch_end(self):
		#print "in layer on epoch end"
		self.loss =np.mean(np.array(self.losses))

	def on_training_begin(self):
		#print "in layer on training begin"
		self.losses = []

	def on_training_end(self):
		#print "in layer on training end"
		pass


	def _gaussian_prior(self):
		return self.activations_prior_strength * np.sum(np.square(self.activations))

	def _gaussian_prior_deriv(self):
		return 2 * self.activations_prior_strength * self.activations

	def _kurtotic_prior(self):
		return self.activations_prior_strength * np.sum(np.log(1 + np.square(self.activations)))

	def _kurtotic_prior_deriv(self):
		return 2 * self.activations / (1 + np.square(self.activations))
	# let's hope these actually work!

	
	def gaussian_weight_prior(self):
		return self.weight_decay_rate * np.sum(self.weights)

	def gaussian_weight_prior_deriv(self):
		return self.weight_decay_rate * self.weights

	def calculate_prediction(self, activations = None):
		if activations is None:
			#print "updating real activations"
			activations = self.activations
		#print "in calculate prediction"
		#print "weights ", self.weights.shape
		return self.activation_function(np.dot(self.weights, activations))

	def calculate_prediction_errors(self):
		self.bottom_up_input = np.reshape(self.bottom_up_input, (len(self.bottom_up_input),1))
		preds = np.subtract(self.bottom_up_input, self.predictions)
		return self.bottom_up_input - self.predictions
	
	def calculate_top_down_error(self):
		if self.top_down_predictions is None:
			return np.zeros((self.N))
		return self.top_down_predictions - self.activations
		

	def update_activations(self):
		#self.activations_learning_rate = 0.001 # 
		#self.activations_prior_strength = 1
		update_rate = self.activations_learning_rate / self.internal_variance
		top_down_update_rate = self.activations_learning_rate / self.top_down_variance 
		bottom_up_term = np.dot(self.weights.T, self.activation_function_derivative(self.prediction_errors)*self.prediction_errors)
		#top_down_term = top_down_update_rate * self.top_down_error
		#print bottom_up_term.shape
		update =  (update_rate*bottom_up_term) + (top_down_update_rate* np.reshape(self.top_down_error, (len(self.top_down_error),1))) - ((self.activations_learning_rate/2) * self._gaussian_prior_deriv())
		
		self.activations = self.activations + update
	
		return self.activations

	def update_weights(self):
		update_rate = self.weights_learning_rate / self.internal_variance
		#update = np.dot(self.activation_function_derivative(self.prediction_errors)* self.prediction_errors, self.activations.T)
		update = (update_rate * (np.dot(self.prediction_errors, self.activations.T))) - (self.weights_learning_rate * self.gaussian_weight_prior_deriv())
		self.weights = self.weights + update
	
		return self.weights
	
	def weighted_mean_incorporation(self,top_down_influence):
		top_down_influence = np.reshape(top_down_influence, (len(top_down_influence),1))
		#print "in weighted mean incorporation"
		#print self.activations.shape
		#print top_down_influence.shape
		
		return (self.activations * (1/self.internal_variance) + (top_down_influence * (1/self.top_down_variance)) /2)

	def incorporation_replace(self, top_down_influence):
		return top_down_influence

	def incorporate_top_down_influence(self, top_down_influence, incorporation_function=incorporation_replace):
		self.incorporation_function = incorporation_function
		activations = incorporation_function(self,top_down_influence)
		return activations
		#print self.activations.shape
		#print self.weights.shape

	def incorporate_multiple_top_down_influences(self, top_down_influences, incorporation_function):
		for i in xrange(len(top_down_influences)):
			self.activations = incorporation_function(top_down_influences[i], self.activations, i)
		return self.activations
		

	def top_down_predict(self, top_down_influence, incorporation_function = incorporation_replace):
		activations = self.incorporate_top_down_influence(top_down_influence, incorporation_function)
		return self.calculate_prediction(activations=activations)
		

	def calculate_activation_prediction_error(self, input_data):
		temp_input = self.bottom_up_input
		self.bottom_up_input = input_data
		pes = self.calculate_prediction_errors()
		activation_pes = (np.dot(self.weights.T, pes)) - self.activations
		self.bottom_up_input = temp_input
		return activation_pes, pes

	def calculate_layer_loss(self):
		new_pes = self.calculate_prediction_errors()
		bottom_up = (1/self.internal_variance) * np.dot(new_pes.T, new_pes)
		top_down = (1/self.top_down_variance) * np.dot(self.top_down_residual.T, self.top_down_residual)
		#print bottom_up
		#print top_down

		loss = (bottom_up + top_down)[0][0]
		#print loss
		return loss

	def forward_pass(self, input_data):
		inp = input_data
		return self.activation_function(np.dot(self.weights.T, inp))
		# hopefully this will work!

	def backward_pass(self, backward_input):
		return self.activation_function(np.dot(self.weights, backward_input))


	def run(self, run_input, top_down_input, callbacks=None, learning_rate = None, train=True):
		if learning_rate is not None:
			self.learning_rate = learning_rate
		#print "in layer run:"
		#print callbacks
		self.bottom_up_input = run_input
		self.top_down_predictions = top_down_input
		self.predictions = self.calculate_prediction()
	#	print "predictions shape" , self.predictions.shape
		self.prediction_errors = self.calculate_prediction_errors()
		#print "prediction errors shape" , self.prediction_errors.shape
		self.top_down_error = self.calculate_top_down_error()
		#print "activations shape ", self.activations.shape
		self.update_activations()
	
		if train:
			self.update_weights()
		self.top_down_residual = np.zeros((self.N,1))
		if self.top_down_predictions is not None:
			self.top_down_residual = self.top_down_predictions
		#print "self losses!"
		#print self.losses
	
		#print "losses before calculation" , self.losses
		loss = self.calculate_layer_loss()
		#print "loss from self calculate layer loss" , loss
		self.losses.append(loss) # that's the problem, very silly!
		#print "loss after calculation ", self.losses
		
		return self.predictions,self.prediction_errors,self.top_down_error, self.activations -self.top_down_residual, self.loss
	
	def get_predictions(self):
		return self.predictions

	def get_prediction_errors(self):
		return self.prediction_errors

	def get_activations(self):
		return self.activations

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



	def get_layer_info(self):
		# replace all of these things with ordered dicts
		info = collections.OrderedDict()
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.activations
		info['weights'] = self.weights
		info['internal_variance'] = self.internal_variance
		info['noise_variance'] = self.noise_variance
		info['top_down_variance'] = self.top_down_variance
		info['learning_rate'] = self.learning_rate
		info['activation_function'] = self.activation_function
		info['activation_function_derivative'] = self.activation_function_derivative
		info['loss'] = self.loss
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
