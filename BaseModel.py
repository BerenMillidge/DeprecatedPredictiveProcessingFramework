from __future__ import division
import numpy as np
from activations import *
import pickle
from utils import *
#from Rao_ballard import *
import jsonpickle
import pdb
import json

# extend jsonpickle for numpy
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


class BaseModel():

	def __init__(self, input_data, epochs=None, learning_rate=None):
		self.input_data = input_data
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.prediction_list = []
		self.prediction_error_list = []
		self.residual_list = []
		self.model = []
		self._stop_training=False


	def _total_loss(self):
		total = 0
		for layer in self.model:
			if hasattr(layer, 'loss'):
				total += layer.loss
		return total

	def _verify_logger(self, logger):
		function_names = ['on_training_begin','on_training_end','on_epoch_begin','on_epoch_end']
		for function in function_names:
			if not hasattr(logger, function):
				raise AttributeError('Logger ' + str(logger) + ' must have function ' + function + 'as an attribute.')

	def _verify_loggers(self, loggers):
		if isinstance(loggers, list):
			for logger in loggers:
				self._verify_logger(logger)
		else:
			self._verify_logger(loggers)
		return True

	def _verify_callback(self, callback):
		function_names = ['on_training_begin','on_training_end','on_epoch_begin','on_epoch_end']
		for function in function_names:
			if not hasattr(callback, function):
				raise AttributeError('Callback ' + str(callback) + ' must have function ' + function + 'as an attribute.')


	def _verify_callbacks(self, callbacks):
		if isinstance(callbacks, list):
			for callback in callbacks:
				self._verify_callback(callback)
		else:
			self._verify_callback(callbacks)
		return True

	def on_training_begin(self):

		for layer in self.model:
			layer.on_training_begin()

		if self.loggers is not None:
			if self._verify_loggers(self.loggers):
				#self.loggers = loggers
				if isinstance(self.loggers, list):
					for logger in self.loggers:
						logger.on_training_begin(self)
				else:
					self.loggers.on_training_begin(self)
		if self.callbacks is not None:
			print("callbacks initialized")
			if self._verify_callbacks(self.callbacks):
				if isinstance(self.callbacks, list):
					for callback in self.callbacks:
						callback.on_training_begin(self)
			else:
				self.callbacks.on_training_begin(self)


	def on_training_end(self):

		for layer in self.model:
			layer.on_training_end()

		if self.loggers is not None:
			if isinstance(self.loggers, list):
				for logger in self.loggers:
					logger.on_training_end(self)
			else:
				self.loggers.on_training_end(self)

		if self.callbacks is not None:
			if isinstance(self.callbacks, list):
				for callback in self.callbacks:
					callback.on_training_end(self)
			else:
				self.callbacks.on_training_end(self)

	def on_epoch_begin(self):
		# notify layers
		for layer in self.model:
			layer.on_epoch_begin()

		# notify loggers
		if self.loggers is not None:
			if isinstance(self.loggers, list):
				for logger in self.loggers:
					logger.on_epoch_begin(self)
			else:
				self.loggers.on_epoch_begin(self)
		# notify callbacks
		if self.callbacks is not None:
			#print "in callback on epoch begin!"
			if isinstance(self.callbacks, list):
				for callback in self.callbacks:
					callback.on_epoch_begin(self)
			else:
				self.callbacks.on_epoch_begin(self)



	def on_epoch_end(self):

		#notify layers
		for layer in self.model:
			layer.on_epoch_end()

		#notify loggers
		if self.loggers is not None:
			if isinstance(self.loggers, list):
				for logger in self.loggers:
					logger.on_epoch_end(self)
			else:
				self.loggers.on_epoch_end(self)

		#notify callbacks!
		if self.callbacks is not None:
			#print "in callback on epoch end"
			if isinstance(self.callbacks, list):
				for callback in self.callbacks:
					callback.on_epoch_end(self)
			else:
				#print "calling individual callback"
				self.callbacks.on_epoch_end(self)

	def save(self):
		raise NotImplementedError('Save function is not yet implemented!')

	def get_model_config(self):
		raise NotImplementedError('Get model config function is not yet implemented!')

	def train(self):
		raise NotImplementedError('Train function is not yet implemented!')

	def forward_pass(self):
		raise NotImplementedError('Forward pass is not yet implemented!')

	def backward_pass(self):
		raise NotImplementedError('Backward pass is not yet implemented!')

	def predict(self):
		raise NotImplementedError('Predict function is not yet implemented!')

	def dream(self):
		raise NotImplementedError('Dream function is not yet implemented!')

