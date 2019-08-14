from __future__ import division
import numpy as np
from activations import *
import pickle
from utils import *
#from Rao_ballard import *
import jsonpickle
import pdb
import json
from Exceptions import *
from BaseModel import *
import matplotlib.pyplot as plt

# extend jsonpickle for numpy
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

def _deserialize_model(model):
	Model = SequentialModel(model['Input_data'], 
					model['learning_rate'],
					model['epochs'],
					model['verbose'],
					model['debug']
					)
	Model._set_model(model['model'])
	Model._set_prediction_list(model['prediction_list'])
	Model._set_prediction_error_list(model['prediction_error_list'])
	Model._set_current_epoch(model['current_epoch'])
	return Model


def load_model(fname):
	m = load(fname)
	m = jsonpickle.decode(m)
	model = _deserialize_model(m)
	return model




class SequentialModel(BaseModel):

	def __init__(self, data=None, learning_rate=None, epochs=None, verbose=True, debug=False, convergence_runs=200):
		self.input_data = data
		self.learning_rate  = learning_rate
		self.epochs = epochs
		self.verbose = verbose
		self.debug=debug

		self.current_epoch = 0
		self.model = []
		self.prediction_list = []
		self.prediction_error_list = []
		self.stop_training=  False
		self.N_layers = 0
		self.loggers = None
		self.callbacks = None
		self.convergence_runs = convergence_runs

	def _verify_layer(self, layer):
		if not hasattr(layer, '_trainable') and not hasattr(layer,'_transform'):
			raise AttributeError('Layers should extend the base trainable layer and therefore have _trainable and _transform attibutes')
		if hasattr(layer, '_trainable') and not hasattr(layer, 'run'):
			raise AttributeError("Trainable layers should all implement a 'run' function, which will be called whenever the layer is called in a training step")
		if hasattr(layer, '_transform') and not hasattr(layer, 'call'):
			raise AttributeError("Transformer layers should all implement a 'call' function which will be called whenever the layer is called by the model.")

		if not hasattr(layer, 'get_layer_info'):
			raise AttributeError('All layers should implement a get_layer_info function giving details to their internal variables for loggers on the model to use.')
		return True


	def add(self, layer):
		if self._verify_layer(layer):
			self.model.append(layer)
			self.N_layers +=1
		return

	def _automated_shape_inference(self):
		# just for vectors
		bottom_up_shape = len(self.input_data)
		for layer in self.model:
			if layer.get_bottom_up_shape() is not None:
				layer._set_bottom_up_shape(bottom_up_shape)
			layer._initialize_weights_activations()
			bottom_up_shape = layer.get_N_neurons()
		return


	def initialize_prediction_prediction_error_list(self, model):

		ps = []
		pes = []
		for i in range(len(self.model) + 1):
			ps.append(None)
			pes.append(0)
	
		return ps, pes

	def run_train_step(self):
		for i,layer in enumerate(self.model):
			if layer._trainable is True:
				if not self.debug:
					# run the standard thing
					upward, downward, loss = layer.run(self.prediction_error_list[i], self.prediction_list[i+1])
					self.prediction_list[i] = downward
					self.prediction_error_list[i+1] = upward
				if self.debug:
					#setup infinite loop
					while True:
						r = raw_input()
						upward, downward, loss= layer.run(self.prediction_error_list[i], self.prediction_list[i+1], verbose=True)
						self.prediction_list[i] = downward
						self.prediction_error_list[i+1] = upward


			if layer._callable is True:
				bottom_up, top_down = layer.call(self.prediction_error_list[i], self.prediction_list[i+1])
				self.prediction_list[i] = bottom_up
				self.prediction_error_list[i+1] = top_down

			if self.cascade:
				# run twice!
					upward, downward, loss = layer.run_without_update(self.prediction_error_list[i], self.prediction_list[i+1])
					self.prediction_list[i] = downward
					self.prediction_error_list[i+1] = upward
					# then run again and actually update
					upward, downward, loss = layer.run(self.prediction_error_list[i], self.prediction_list[i+1])
					self.prediction_list[i] = downward
					self.prediction_error_list[i+1] = upward

	def run_train_epoch(self):
		print("In run train epoch! length of input data: ", len(self.input_data.shape))
		self.on_epoch_begin()
		if len(self.input_data.shape) ==1:
			print("Going down dimension 1 train epoch!")
			self.prediction_error_list[0] = self.input_data
			for i in range(self.convergence_runs):
				self.run_train_step()

		if  len(self.input_data.shape)== 2 and self.input_data.shape[1] ==1:
			
			print("Two dimensinos but with 1 in second dimensino, so treating as first!")
			self.prediction_error_list[0] = self.input_data
			for i in range(self.convergence_runs):
				self.run_train_step()

		if len(self.input_data.shape) == 2 and self.input_data.shape[1] != 1:
			for data in self.input_data:
				self.prediction_error_list[0] = np.reshape(data, (len(data),1)) 
				for i in range(self.convergence_runs):
					self.run_train_step()
		if len(self.input_data.shape) > 2:
			raise DataError('Input data can only be two dimensional at the moment... Matrices are not supported')

		self.on_epoch_end()


	def collect_activations(self):
		activations = []
		for layer in self.model:
			activations.append(layer.get_activations())
		return activations

	def collect_weights(self):
		weights = []
		for layer in self.model:
			weights.append(layer.get_weights())
		return weights

	def collect_predictions(self):
		predictions = []
		for layer in self.model:
			predictions.append(layer.get_predictions())
		return predictions

	def collect_prediction_errors(self):
		pes = []
		for layer in self.model:
			pes.append(layer.get_prediction_errors())
		return pes

	def collect_internal_variance(self):
		vals = []
		for layer in self.model:
			vals.append(layer.get_internal_variance())
		return vals

	def collect_top_down_variance(self):
		vals = []
		for layer in self.model:
			vals.append(layer.get_top_down_variance())
		return vals

	def collect_noise_variance(self):
		vals = []
		for layer in self.model:
			vals.append(layer.get_noise_variance())
		return vals

	def collect_top_down_errors(self):
		vals = []
		for layer in self.model:
			vals.append(layer.get_top_down_error())
		return vals

	def collect_weights_and_activations(self):
		return self.collect_weights(), self.collect_activations()

	def get_prediction_list(self):
		return self.prediction_list

	def get_prediction_error_list(self):
		return self.prediction_error_list

	def get_prediction_and_prediction_error_list(self):
		return self.get_prediction_list(), self.get_prediction_error_list()

	def _set_prediction_list(self, prediction_list):
		self.prediction_list = prediction_list

	def _set_prediction_error_list(self, prediction_error_list):
		self.prediction_error_list = prediction_error_list

	def _set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def _set_epochs(self, epochs):
		self.epochs = epochs

	def _set_current_epoch(self, current_epoch):
		self.current_epoch = current_epoch

	def _set_current_batch(self, current_batch):
		self.current_batch = current_batch

	def _set_model(self, model):
		self.model = model

	def _set_stop_training(self, stop):
		self.stop_training = stop

	def get_model_metadata(self):
		metadata['Input_data'] = self.input_data
		metadata['learning_rate'] = self.learning_rate
		metadata['epochs'] = self.epochs
		return metadata

	def get_model_information(self):
		info = {}
		info['model'] = self.model
		info['prediction_list'] = self.prediction_list
		info['prediction_error_list'] = self.prediction_error_list
		info['current_epoch'] = self.current_epoch
		info['loggers'] = self.loggers
		info['callbacks'] = self.callbacks
		return info

	def get_layers_info(self):
		info = {}
		for i, layer in enumerate(self.model):
			#print [key for key in layer.__dict__.keys()]
			if layer._trainable == True:
				# only do this for trainable layers etc
				if hasattr(layer, 'get_layer_info'):
					layer_info_func = getattr(layer, 'get_layer_info')
					if layer_info_func is callable or 1 ==1:
						info['layer_'+str(i)] = layer_info_func()
					else:
						raise TypeError('Get layer info function of the layer must be a function. Actual type received is: ' + str(type(layer_info_func)))
				else:
					raise AttributeError('Trainable layers should have a get_layer_info function which summarizes the crucial parameters of the layer. Layer ' +str(i) + 'does not.')
		return info

	def get_model_config(self):
		return combine_dicts((self.get_model_metadata(), self.get_model_information()))

	def _serialize_model_dict(self, model_info):
		for k,v in model_info.iteritems():

			if v is None:
				print("replaced: ", k)
				model_info[k] = 'None'
		return model_info
		
	def save(self, save_name):
		if not isinstance(save_name, str):
			raise ValueError('Save name must be a string')

		s = jsonpickle.encode(self.get_model_config())
		#print len(self.get_model_config().keys())
		save(s, save_name)

	def train(self, input_data=None, epochs=None, loggers=None, callbacks=None, plot_per_epoch=False, cascade = False):
		self.cascade = cascade
		if epochs is not None:
			self.epochs = epochs
		if input_data is not None:
			self.input_data = input_data

		if self.epochs is None:
			raise ValueError('You must provide a value for number of epochs either in the model constructor or in the train function')

		if self.input_data is None:
			raise ValueError('You must provide input data, either in the model constructor or in the train function')


		if loggers is not None and self._verify_loggers(loggers):
			self.loggers = loggers
#
		if callbacks is not None and self._verify_callbacks(callbacks):
			self.callbacks = callbacks

		self.prediction_list, self.prediction_error_list = self.initialize_prediction_prediction_error_list(self.model)

		self.on_training_begin()

		for i in range(self.epochs):
			if self.stop_training is False:
				self.run_train_epoch()
				if plot_per_epoch:
					"""
					#np.reshape(plt.imshow(self.model[0].get_representation_units()), (28,28))
					l = self.model[0]
					#pred = self.model[0].get_representation_units()
					pred = l.get_predictions()
					print(pred.shape)
					a = l.get_activations()
					print("Actiations: ", a.shape)
			
					print("Mean Prediction: ", np.mean(pred))
					print("Loss: ", self.model[0].get_loss())
					w = l.get_weights()
					print("Weight shape", w.shape)
					#print("Lateral Weight shape", lat.shape)
					print("Mean weights: ",np.mean(w))
					#print("Mean lateral weights", np.mean(lat))
					pred = np.reshape(pred, (28,28))
					plt.imshow(pred)
					plt.show()
					"""
					for i,layer in enumerate(self.model):
						print("Layer: ", i)
						#print("mean top down input: ", layer.get_top_down_input())
						print("mean predictoin_errors :", np.mean(layer.get_prediction_errors()))
						print("Mean predictions: ", np.mean(layer.get_predictions()))
						print("mean activations: ", np.mean(layer.get_representation_units()))
						print("Mean weights: ", np.mean(layer.get_weights()))
						if i == 0:
							print("In grahing predictions")
							pred = layer.get_predictions()
							pred = np.reshape(pred, (28,28))
							plt.imshow(pred)
							plt.show()

			if self.verbose:
				print("Epochs: " + str(i))
			self.current_epoch +=1
			if self.stop_training is True:
				pass
		self.on_training_end()

	def generate_prediction(self):
		return self.model[0].calculate_prediction()

	def top_layer_gaussian_stats(self, plot_hist = False):
		acts = self.model[-1].get_activations()
		if plot_hist:
			xs = np.arange(0, acts)
			plt.hist(acts, xs, w=0.4, align='center')
			plt.show()
		return np.mean(acts), np.var(acts)


	def top_layer_length(self):
		return len(self.model[-1].get_activations())

	def sample_top_layer_activations(self):
		mu, var = self.top_layer_gaussian_stats()
		l = self.top_layer_length()
		samp = np.random.normal(loc=mu, scale=var, size=l)
		return samp

	def forward_pass(self, forward_input):
		acts = []
		layer_input = forward_input
		for layer in self.model:
			layer_input = layer.forward_pass(layer_input)
			acts.append(layer_input)
		return acts

	def backward_pass(self, backward_input):
		preds = []
		layer_input = backward_input
		for layer in reversed(self.model):
			layer_input = layer.backward_pass(layer_input)
			preds.append(layer_input)
		return preds

	def forward_backward(self, inp):
		acts = self.forward_pass(inp)
		back = self.backward_pass(acts[-1])
		return back

	def dream(self, N, initial_input):
		forward_input = initial_input
		backward_input = None
		preds = []
		for i in range(N):
			back = forward_backward(forward_backward)[-1]
			forward_input = back
			preds.append(back)

		return preds



