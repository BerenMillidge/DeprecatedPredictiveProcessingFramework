from utils import *
from initializers import *
import  activations


class Friston2003Layer(BaseLayer):

	def __init__(self, bottom_up_input_dim, top_down_input_dim, layer_dim,learning_rate = 0.001, weights_initializer=gaussian_initializer, lateral_weighs_initializer= gaussian_initializer, cause_units_initializer = gaussian_initializer, biases = False, bias_initialier = gaussian_initializer, activation_function  = "relu"):

		if not isinstance(bottom_up_input_dim, int):
			raise ValueError('Bottom up input dimension must be an integer')

		if not isinstance(top_down_input_dim, int):
			raise ValueError('Top down input dimension must be an integer')

		if not isinstance(layer_dim, int):
			raise ValueError('Layer dimension must be an integer')

		self.bottom_up_input_dim = bottom_up_input_dim
		self.top_down_input_dim = top_down_input_dim
		self.layer_dim = layer_dim
		self.weights_initializer = weights_initializer
		self.lateral_weighs_initializer = lateral_weighs_initializer
		self.learning_rate = learning_rate
		self.phi_initializer = cause_units_initializer
		self.biases_flag = biases
		self.bias_initialier = bias_initialier

		if activation_function not in activations.funcdict.keys();
			raise ValueError("Activation function not found. Valid activations functions are : " +  str(activations.funcdict
				.keys()) ".")

		else:
			self.activation_function, self.activation_function_derivative = activations.funcdict[activation_function]


		self.weights = self.weights_initializer([self.top_down_input_dim, self.layer_dim])
		self.lateral_weights = self.lateral_weighs_initializer([self.layer_dim, self.layer_dim])
		self.cause_units = self.phi_initializer(self.layer_dim)

		if self.biases_flag:
			self.biases =self.bias_initialier(self.layer_dim)
		else:
			self.biases = np.zeros(self.layer_dim)


		self._trainable = True
		self._callable = False

	def _calculate_linear_prediction(self):
		return np.dot(self.weights, self.top_down_input) + self.biases

	def _calculate_top_down_prediction(self):
		return self.activation_function(self.linear_prediction)

	def _calculate_prediction_derivative(self):
		return self.activation_function_derivative(self.linear_prediction)

	def _calculate_lwinv(self):
		return np.inv(1+self.lateral_weights)

	def _compute_upward_projection(self):
		upward_modifier = np.dot(self.lwinv, self.weights)
		return np.dot(upward_modifier.T, self.prediction_errors * self.prediction_derivative )

	def _compute_prediction_errors(self):
		return np.dot(self.lwinv, self.cause_units - self.top_down_prediction)

	def _update_cause_units(self):
		self.cause_units += self.learning_rate * (-1 * self.bottom_up_input - np.dot(self.lwinv, self.prediction_errors))

	def _update_weights(self):
		self.weights += self.learning_rate * (np.dot(self.lwinv, np.dot((self.prediction_errors * self.prediction_derivative), self.top_down_input.T)))

	def _update_lateral_weights(self):
		self.lateral_weights += self.learning_rate * (np.dot(self.lwinv, np.dot(self.prediction_errors, self.prediction_errors.T) -1))

	def _update_biases(self):
		self.biases += self.learning_rate * (np.dot(self.lwinv, (self.prediction_derivative * self.prediction_errors)))

	def _compute_loss(self):
		return np.dot(self.prediction_errors.T, self.prediction_errors)

	def run(self, bottom_up_input, top_down_input):
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input
		self.lwinv = self._calculate_lwinv()
		self.linear_prediction = self._calculate_linear_prediction()
		self.top_down_prediction = self._calculate_top_down_prediction()
		self.prediction_derivative = self._calculate_prediction_derivative()
		self.prediction_errors = self._compute_prediction_errors()
		self._update_cause_units()
		self._update_weights()
		self._update_lateral_weights()
		if self.biases_flag:
			self._update_biases()
		self.loss = _compute_loss()
		return self._compute_upward_projection(), self.cause_units, self.loss 

	def get_representation_units(self):
		return self.cause_units
 
	def get_cause_units(self):
		return self.cause_units

	def get_activations(self):
		return self.cause_units

	def get_prediction(self):
		if self.top_down_prediction is None:
			return self._calculate_top_down_prediction()
		return self.top_down_prediction

	def get_prediction_errors(self):
		if self.prediction_errors is None:
			return self._compute_prediction_errors()
		return self.prediction_errors

	def get_forward_weights(self):
		return self.weights

	def get_lateral_weights(self):
		return self.lateral_weights

	def get_weights(self):
		return self.get_forward_weights(), self.get_lateral_weights()

	def get_biases(self):
		return self.biases

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

	def _set_biases(self, biases):
		self.biases = biases

	def get_layer_info(self):
		info = {}
		info['predictions'] = self.predictions
		info['prediction_errors'] = self.error_units
		info['activations'] = self.representation_units
		info['forward_weights'] = self.forward_weights
		info['lateral_weights'] = self.lateral_weights
		info['activation_function'] = self.activation_function
		info['biases'] = self.biases
		info['activation_function'] = self.activation_function
		info['activation_function_derivative'] = self.activation_function_derivative
		return info

