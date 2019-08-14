from utils import *
from initializers import *
from BaseLayer import *



class Friston2003LinearHebbianLayer(BaseLayer):
	def __init__(self, bottom_up_input_dim, top_down_input_dim,learning_rate = 0.01, weights_initializer=default_gaussian_initializer, lateral_weighs_initializer= default_gaussian_initializer, cause_units_initializer = default_gaussian_initializer):

		if not isinstance(bottom_up_input_dim, int):
			raise ValueError('Bottom up input dimension must be an integer')

		if not isinstance(top_down_input_dim, int):
			raise ValueError('Top down input dimension must be an integer')

		#if not isinstance(layer_dim, int):
		#	raise ValueError('Layer dimension must be an integer')

		self.bottom_up_input_dim = bottom_up_input_dim
		self.top_down_input_dim = top_down_input_dim
		self.layer_dim = bottom_up_input_dim #
		self.weights_initializer = weights_initializer
		self.lateral_weighs_initializer = lateral_weighs_initializer
		self.learning_rate = learning_rate
		self.phi_initializer = cause_units_initializer

		self.weights = self.weights_initializer([self.layer_dim, self.top_down_input_dim])
		self.lateral_weights = self.lateral_weighs_initializer([self.layer_dim, self.layer_dim])
		self.cause_units = self.phi_initializer([self.layer_dim,1])

		self._trainable = True
		self._callable = False


	def _calculate_top_down_prediction(self):
		if self.top_down_input is None:
			self.top_down_input = np.zeros((self.top_down_input_dim,1))
		return np.dot(self.weights, self.top_down_input)

	def _calculate_lwinv(self):
		return np.linalg.inv(1+self.lateral_weights)

	def _compute_upward_projection(self):
		upward_modifier = np.dot(self.weights.T, self.lwinv.T)
		return np.dot(upward_modifier, self.prediction_errors)

	def _compute_prediction_errors(self):
		return np.dot(self.lwinv, self.cause_units - self.top_down_prediction)

	def _update_cause_units(self):
		self.cause_units +=  ((-1 * self.bottom_up_input) - np.dot(self.lwinv, self.prediction_errors))

	def _update_weights(self):
		self.weights -= self.learning_rate * (np.dot(self.lwinv, np.dot(self.prediction_errors, self.top_down_input.T)))

	def _update_lateral_weights(self):
		self.lateral_weights -= self.learning_rate * (np.dot(self.lwinv, np.dot(self.prediction_errors, self.prediction_errors.T) -1))

	def _compute_loss(self):
		return np.dot(self.prediction_errors.T, self.prediction_errors)


	def run(self, bottom_up_input, top_down_input):
		print("In run function")
		#print(bottom_up_input)
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input
		self.lwinv = self._calculate_lwinv()
		self.top_down_prediction = self._calculate_top_down_prediction()
		self.prediction_errors = self._compute_prediction_errors()
		self._update_cause_units()
		self._update_weights()
		self._update_lateral_weights()
		self.loss = self._compute_loss()
		self.losses.append(self.loss)
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

	def get_loss(self):
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

	def _set_lateral_weights(self, weights):
		self.lateral_weights = weights

	def get_bottom_up_shape(self):
		return self.bottom_up_input_dim

	def _initialize_weights_activations(self):
		pass

	def get_layer_info(self):
		info = {}
		info['predictions'] = self.cause_units 
		info['prediction_errors'] = self.prediction_errors
		info['activations'] = self.cause_units
		info['forward_weights'] = self.weights
		info['lateral_weights'] = self.lateral_weights
		info['activation_function'] = None
		return info


