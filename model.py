from __future__ import division
from RB_layer import RB_Layer
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


def _deserialize_model(model):
	#create the model
	Model = RB_Model(model['Input_data'], 
					model['Neurons_per_layer'],
					model['Number_of_Layers'],
					model['learning_rate'],
					model['epochs'],
					model['batch_size'],
					model['dataprovider'],
					model['shuffle'],
					model['layer_type'],
					model['activation_function'],
					model['activation_function_deriv'],
					model['learning_rate_scheduler'],
					model['verbose'],
					model['shuffle_algorithm'],
					model['initializer'],
					model['batch_processing']
					)
	Model._set_model(model['model'])
	Model._set_prediction_list(model['prediction_list'])
	Model._set_prediction_error_list(model['prediction_error_list'])
	Model._set_current_epoch(model['current_epoch'])
	Model._set_current_batch(model['current_batch'])
	return Model

def load_model(fname):
	#jsonmodel = load(fname)
	#print type(jsonmodel)
	#print jsonmodel
	#try:
	##	model = json.load(jsonmodel)
	#	return model
	#except Exception as e:
	#	raise ValueError('Depickling failed: ' + str(e))
	m = load(fname)
	m = jsonpickle.decode(m)
	#print type(m)
	#print m.keys()
	model = _deserialize_model(m)
	#print type(model)
	return model
"""
def load_model(fname):
	if type(fname) != 'str':
		raise ValueError('Filename of model to be loaded must be a string')

	#try to load
	model = None
	try:
		if fname.splits['.'][-1] == 'npy':
			model = np.load(fname)
		else:
			model = load(fname)
	except Exception as e:
		raise ValueError('Cannot load model file provided:  ', e)

	if model is None:
		raise ValueError('Model failed to load correctly')

	#create the model
	Model = RB_Model(model['Input_data'], 
					model['N_per_layer'],
					model['N_layers'],
					model['learning_rate'],
					model['epochs'],
					model['batch_size'],
					model['dataprovider'],
					model['shuffle'],
					model['layer_type'],
					model['activation_function'],
					model['activation_function_deriv'],
					model['learning_rate_scheduler'],
					model['verbose'],
					model['shuffle_algorithm'],
					model['initializer'],
					model['batch_processing']
					)
	Model._set_model(model['model'])
	Model._set_prediction_list(model['prediction_list'])
	Model._set_prediction_error_list(model['prediction_error_list'])
	Model._set_current_epoch(model['current_epoch'])
	Model._set_current_batch(model['current_batch'])
	return Model
"""



class RB_Model():

	def __init__(self, input_data, N_per_layer, N_layers, learning_rate, epochs,batch_size, 
		dataprovider=None, shuffle=True,layer_type=RB_Layer,activation_function = 'sigmoid', 
		activation_function_deriv = sigmoidderiv, learning_rate_scheduler=None, verbose=True,
		shuffle_algorithm= None, initializer=None, batch_processing = False, callbacks=None, plot_per_epoch=False):
		#initialize everything
		self.input_data = input_data
		#print "in init"
		#print self.input_data
		self.N_layers = N_layers
		self.N_per_layer = N_per_layer
		# turn into a nidentical list if it is a float
		if isinstance(N_per_layer, int):
			# i.e. if int 
			self.N_per_layer = [self.N_per_layer for i in range(self.N_layers)]
	
		self.learning_rate = learning_rate
		self.epochs = epochs 
		self.dataprovider = dataprovider
		self.layer_type = layer_type
		self.activation_function = activation_function
		self.activation_function_deriv = activation_function_deriv
		self.current_epoch = 0 # initialize as 0
		self.learning_rate_scheduler = learning_rate_scheduler 
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_batches = len(self.input_data) // self.batch_size
		self.current_batch = 0
		self.verbose = verbose
		self.initializer = initializer
		self.batch_processing = batch_processing
		self.plot_per_epoch = plot_per_epoch


		#initialize the model!
		self.prediction_list = []
		self.prediction_error_list = []
		self.model = self.initialize_model()


		# perhaps get an algorithm for shuffling
		self.shuffle_algorithm = None

		#stop training thing
		self.stop_training = False




	def initialize_model(self):
		model = [] # currently just a list of layers
		for i in range(self.N_layers):
			#print self.N_per_layer[i]
			if i == 0:
				layer_input = self.input_data
			#create fake input data for other layers
			if i >0:
				layer_input = np.zeros((self.N_per_layer[i-1]))
			layer = self.layer_type(self.N_per_layer[i], layer_input,activation_function = self.activation_function, learning_rate = self.learning_rate, initializer = self.initializer)
			model.append(layer)
		self.prediction_list, self.prediction_error_list = self.initialize_prediction_predictoin_error_list(model)
		return model

	def _verify_layer(self, layer):
		#just check general layer attributes
		if not hasattr(layer, '_trainable') or not hasattr('_transform'):
			raise AttributeError('Layers should extend the base trainable layer and therefore have _trainable and _transform attibutes')
		if hasattr(layer, '_trainable') and not hasattr(layer, 'run'):
			raise AttributeError("Trainable layers should all implement a 'run' function, which will be called whenever the layer is called in a training step")
		if hasattr(layer, '_transform') and not hasattr(layer, 'call'):
			raise AttributeError("Transformer layers should all implement a 'call' function which will be called whenever the layer is called by the model.")

		if not hasattr(layer, 'get_layer_info'):
			raise AttributeError('All layers should implement a get_layer_info function giving details to their internal variables for loggers on the model to use.')
		return True

	def add(self, layer):
		if _verify_layer(layer):
			self.model.append(layer)
		return

	def _total_loss(self):
		total = 0
		for layer in self.model:
			if hasattr(layer, 'loss'):
				total += layer.loss
		return total

	def on_training_begin(self):

		#notify layers
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
				#self.callbacks = callbacks
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

		# calculate the total loss


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

		if self.plot_per_epoch:
			print("Inside plot per epoch!")
			pred = np.reshape(self.model[0].get_predictions(), (28,28))
			plt.imshow(pred)
			plt.show()


	def initialize_prediction_predictoin_error_list(self, model):
		ps = []
		pes = []
		N = len(model)
		for i in range(N+1):
			ps.append(None)
			pes.append(0)
	
		return ps, pes

	def default_get_next(self):

		data =  self.input_data[self.current_batch*self.batch_size: (self.current_batch+1)*self.batch_size]
		self.current_batch +=1
		return data

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



	def run_train_step(self, train):
		for j, layer in enumerate(self.model):
			#if self.verbose:
				#print "in layer: " + str(j)
				#print "in layer ", j
			#if type(self.prediction_list[j+1]) != type(0):
			#	print self.prediction_list[j+1].shape
			##print self.prediction_list[j+1]
			#if type(self.prediction_list[j]) != type(0):
			#	print self.prediction_list[j].shape
			#print "training layer: " + str(j)
			#layer = self.model[j]
			#print("In run train step: layer ",j)
			ps, pes, tde,tdr,loss = layer.run(self.prediction_error_list[j], self.prediction_list[j+1], self.callbacks,learning_rate = self.learning_rate, train=train)
			#print(ps.shape)
			#ps, pes, loss = layer.run(self.prediction_error_list[j], self.prediction_list[j+1], self.callbacks,learning_rate = self.learning_rate, train=train)
			#print "max predictions: " , np.max(ps)
			#print " max prediction errors: " , np.max(pes)
			#print "max top down errors: " , np.max(tde)
			#print "max top down residual: " , np.max(tdr)

			#print "predictions output shape ", ps.shape
			#print "predictions error output shape", pes.shape
			self.prediction_list[j] = ps
			self.prediction_error_list[j+1] = tdr



	def run_train_epoch(self, train):

		if self.batch_processing is True:
			for i in range(self.num_batches):
				if self.dataprovider is None:
					input_data = self.default_get_next()
				else:
					input_data = self.dataprovider.next() 
		else:
			# just use the whole data
			input_data = self.input_data

		#notify epoch beginning
		self.on_epoch_begin()

		# if it is one dimensional do this:
		if len(input_data.shape) ==1 or (len(input_data.shape) == 2 and input_data.shape[1] == 1):

			self.prediction_error_list[0] = input_data # set the data as the first PE
			self.run_train_step(train)

			
		if len(input_data.shape)== 2 and not input_data.shape[1] ==1:
			# do the following algorithm, just iterate through it
			for i in range(len(input_data)):
				self.prediction_error_list[0] = input_data[i]
				#print input_data[i].shape
				self.run_train_step(train)

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

	# various setters too!

	def _set_prediction_list(self, prediction_list):
		self.prediction_list = prediction_list

	def _set_prediction_error_list(self, prediction_error_list):
		self.prediction_error_list = prediction_error_list

	def _set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def _set_epochs(self, epochs):
		self.epochs = epochs

	def _set_dataprovider(self, dataprovider):
		self.dataprovider = dataprovider

	def _set_shuffle(self, shuffle):
		self.shuffle = shuffle

	def _set_layer_type(self, layer_type):
		self.layer_type = layer_type

	def _set_activation_function(self, activation_function):
		self.activation_function = activation_function

	def _set_activation_function_deriv(self, activation_function_deriv):
		self.activation_function_deriv = activation_function_deriv

	def _set_learning_rate_scheduler(self, learning_rate_scheduler):
		self.learning_rate_scheduler = learning_rate_scheduler

	def _set_verbose(self, verbose):
		self.verbose = verbose

	def _set_shuffle_algorithm(self, shuffle_algorithm):
		self.shuffle_algorithm = shuffle_algorithm

	def _set_initializer(self, initializer):
		self.initializer = initializer

	def _set_batch_processing(self, batch_processing):
		self.batch_processing = batch_processing

	def _set_current_epoch(self, current_epoch):
		self.current_epoch = current_epoch

	def _set_current_batch(self, current_batch):
		self.current_batch = current_batch

	def _set_model(self, model):
		self.model = model

	def _set_stop_training(self, stop):
		self.stop_training = stop

	def get_model_metadata(self):
		metadata = {}# empty dict, there are better ways of doing this
		metadata['Input_data'] = self.input_data
		metadata['Number_of_Layers'] = self.N_layers
		metadata['Neurons_per_layer'] = self.N_per_layer
		metadata['learning_rate'] = self.learning_rate
		metadata['epochs'] = self.epochs
		metadata['dataprovider'] = self.dataprovider
		metadata['shuffle'] = self.shuffle
		metadata['layer_type'] = self.layer_type
		metadata['activation_function'] = self.activation_function
		metadata['activation_function_deriv'] = self.activation_function_deriv
		metadata['learning_rate_scheduler'] = self.learning_rate_scheduler
		metadata['verbose'] = self.verbose
		metadata['shuffle_algorithm'] = self.shuffle_algorithm
		metadata['initializer'] = self.initializer
		metadata['batch_processing'] = self.batch_processing
		metadata['batch_size'] = self.batch_size
		return metadata

	def get_model_information(self):
		info = {}
		info['model'] = self.model
		info['prediction_list'] = self.prediction_list
		info['prediction_error_list'] = self.prediction_error_list
		info['current_epoch'] = self.current_epoch
		info['current_batch'] = self.current_batch
		info['loggers'] = self.loggers
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

	def __getstate__(self):
		return self.__dict__()

	def __setstate__(self, d):
		self.__dict__.update(d)

	def get_model_config(self):
		print("in get model config")
		print(type(self.get_model_metadata()))
		print(type(self.get_model_information()))
		print(type(combine_dicts((self.get_model_metadata(), self.get_model_information()))))
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
		print(len(self.get_model_config().keys()))
		#print type(s)
		#print s
		save(s, save_name)
		#m = load(save_name)
		#m = jsonpickle.decode(s)
		#print len(m.keys())

		#model_info = jsonpickle.encode(self._serialize_model_dict(self.get_model_config()))
		#m = jsonpickle.decode(model_info)
		# what is this dict object is not callable nonsense... argh!
		#print type(m)
		#print type(model_info)
		#print type(save_name)
		#print model_info.keys()
		#print self.__getattr__()

			#print type(model_info)
			#print type(save_name)
		
		#print model_info
		#save(model_info, save_name)
	
		#except Exception as e:
		#	raise ValueError('Failed to save model: ' , str(e))


	"""
	def save(self, save_name):
		try:
			pdb.set_trace()
			jsonstr= jsonpickle.encode(self.__dict__())
			save(jsonstr, save_name)
		except Exception as e:
			print e
			raise
	"""



	def run(self, input_data=None, epochs=None, loggers=None, callbacks=None, train=False):
		return self.train(input_data, epochs,loggers, callbacks, train)


	def train(self, input_data=None, epochs=None, loggers = None, callbacks=None, train=True, plot_per_epoch=None):

		if plot_per_epoch:
			self.plot_per_epoch = plot_per_epoch

		if epochs is not None:
			self.epochs = epochs
		if input_data is not None:
			print("input data is not none so...")
			self.input_data = input_data
			#printself.input_data


		self.loggers = loggers
		self.callbacks = callbacks

		#notify training start
		self.on_training_begin()


		if self.verbose:
			print("Training model...")

		for i in range(self.epochs):
			if self.stop_training == False:
				if self.learning_rate_scheduler is not None:
					self.learning_rate = self.learning_rate_scheduler(self.learning_rate, self.current_epoch)
				
				self.run_train_epoch(train)
				if self.verbose:
					print("Epochs: " + str(i))
				self.current_epoch +=1
			if self.stop_training == True:
				return
			
		##set logger to save
		#notify end of treaining
		self.on_training_end()


		return self.collect_weights_and_activations()

	def generate_average_prediction(self):
		return self.model[0].calculate_prediction()

	def L1_prediction_error(self, data):
		return self.model[0].calculate_prediction_error(data)


	def propagate_top_down_prediction(self, top_layer_activations):
		#set the top layer of activations
		self.model[-1]._set_activations(top_layer_activations)
		prediction  = self.model[-1].calculate_prediction()
		for layer in reversed(self.model[0:len(self.model)-1]):
			prediction = layer.top_down_predict(prediction)
		return prediction


	def forward_pass(self, forward_input):
		pes = []
		activation_errors =[]
		layer_input = forward_input
		for layer in layer:
			activation_pe, pe = layer.calculate_activation_prediction_error(layer_input)
			#replace the current layer input with the activation PE
			layer_input = activation_pe
			#and add the errors
			pes.append(pe)
			activation_errors.append(activation_pe)
		return pes, activation_errors

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

	def forward_pass_to_final_activations(self, forward_input):
		layer_input = forward_input
		print("forward input" , forward_input.shape)
		top_down_input = self.model[1].calculate_prediction()
		acts = []
		for i,layer in enumerate(self.model):
			p, pe, tde, tdr, loss = layer.run(layer_input, top_down_input, train=False)
			layer_input = tdr
			#print "tdr!"
			#print type(tdr)
			#print len(tdr)
			if i < len(self.model) -2:
				top_down_input = self.model[i+2]
			else:
				top_down_input = None
			acts.append(layer.get_activations())
		return acts[-1]

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
		
	def dream(self, N, initial_input=None, plot=False, shape=None):
		sh = initial_input.shape
		if initial_input is None and shape is None:
			raise ValueError('Shape must be supplied if no initial input is supplied since the model must know the shape of the input you want.')

		layer_input = initial_input
		if initial_input is None:
			layer_input = np.zeroes(shape)

		preds = []
		for i in range(N):
			# forward pass 
			acts = self.forward_pass_to_final_activations(layer_input)
			pred = self.propagate_top_down_prediction(acts)
			if plot:
				plt.imshow(np.reshape(pred, shape))
				plt.show()
			preds.append(pred)
			layer_input = pred

		return np.array(preds)

	def dream_no_change(self, N, initial_input):
		forward_input = initial_input
		backward_input = None
		preds = []
		for i in range(N):
			back = forward_backward(forward_backward)[-1]
			forward_input = back
			preds.append(back)

		return preds


