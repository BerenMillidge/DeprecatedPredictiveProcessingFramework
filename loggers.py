
import numpy as np
from utils import *
import collections

def convert_logs_to_numpy_arrays(logs):
	if isinstance(logs, np.ndarray):
		return logs
	if isinstance(logs, dict):
		return convert_logs_to_lists(logs)
	if isinstance(logs, list):
		return np.array(logs)
	else:
		raise TypeError('Type not recognised. Should be dict or list or numpy array; got: ' + str(type(logs)))


def convert_logs_to_lists(logs):
	if isinstance(logs, list):
		return logs
	if isinstance(logs, dict):
		l = []
		for v in logs.values():
			l.append(convert_logs_to_lists(v))
		return l
	if isinstance(logs, int) or isinstance(logs, float):
		return [logs]

	if isinstance(logs, np.ndarray):
		return logs

	if logs is None:
		return [None]
	else:
		raise TypeError('Something went wrong with types here. Expected list or dict. Got: ' + str(type(logs)))


class BaseLogger():

	def __init__(self, save_name=None):
		self.save_name = save_name
		self.logs = collections.OrderedDict()
		self.current_epoch = 0

	def get_save_name(self):
		return self.save_name

	def _set_save_name(self, save_name):
		self.save_name = save_name

	def save(self):
		if self.save_name is not None:
			print("saving logs...")
			if isinstance(self.logs, np.ndarray):
				np.save(self.save_name, self.logs)
			else:
				save(self.logs, self.save_name)
			return True

	def get_logs(self):
		return self.logs

	def _set_logs(self, logs):
		self.logs = logs

	def get_current_epoch(self):
		return self.current_epoch

	def _set_current_epoch(self, epoch):
		self.current_epoch = epoch

	def on_training_begin(self, input_class):
		pass

	def on_training_end(self, input_class):
		pass

	def on_epoch_begin(self, input_class):
		pass

	def on_epoch_end(self, input_class):
		self.log(input_class)


	def log(self, input_class):
		pass

	def on_training_end(self, input_class):
		if self.return_type == 'dict' or self.return_type is None:
			self.save()
			return
		if self.return_type == 'list':
			self.logs = convert_logs_to_lists(self.logs)
			self.save()
			return
		if self.return_type == 'numpy.ndarray':
			self.logs = convert_logs_to_numpy_arrays(self.logs)
			self.save()
			return
		else:
			self.save()
			return

	def write_to_file(self,fname, max_line_width=None, precision=None, suppress_small=None):
		#print fname
		#print type(fname)
		with open(fname, 'w+') as f:
			f.write('flibblejib!!')
			#assume it's an epoch thing
			for key, epoch_dict in self.logs.items():
				f.write(key + " \n")
				for k, layer_dict in epoch_dict.items():
					f.write(k + "\n")
					for k2, arr in layer_dict.items():
						f.write(k2 + "\n")
						f.write(np.array_str(arr, max_line_width = max_line_width, precision = precision, suppress_small = suppress_small))
						f.write("\n")
		return



class BasicLogger(BaseLogger):

	# okay the aim here is to figure out what it wants you to log
	# and try to log it generally
	# as a dict which it then saves, so let's figure it out
	def __init__(self, save_name,log_list = None, epoch_per_log=1, return_type=None):
		print(type(save_name))
		if type(save_name) is not type('str'):
			raise TypeError('Save name should be a string')
		self.save_name = save_name

		if log_list is not None:
			if isinstance(log_list, list) or isinstance(log_list, str):
				if isinstance(log_list, list):
					self.log_list = log_list
				if isinstance(log_list, str):
					self.log_list = [log_list]
			else:
				raise TypeError('Log list must either be a list of loggable attributes or a string of a single loggable attribute. You inputted: ' + str(type(log_list)))
			
		else:
			self.log_list = log_list
		print(self.log_list)
		if not isinstance(epoch_per_log, int):
			raise TypeError('Number of epochs per log must be an integer. You inputted: ' + str(type(epoch_per_log)))

		#now do the value error
		if epoch_per_log <=0:
			raise ValueError('Epoch per log must be a positive nonzero number')

		# initizliase the logs
		self.logs = collections.OrderedDict()
		self.current_epoch = 0
		self.epoch_per_log = epoch_per_log

		# sort out the return type
		self.return_type = return_type
		if return_type is not None and return_type not in ['list', 'dict','numpy.ndarray']:
			print("Return type is not recognised. Will return in the defualt dict of dicts format")
			self.return_type = None


	def log(self, input_class):
		
		if self.current_epoch % self.epoch_per_log == 0:
			# only log after the correct nmber of epochs
			l = collections.OrderedDict()

			if hasattr(input_class, 'model'):
				layers = getattr(input_class, 'model')
				for i in xrange(len(layers)):
					layer = layers[i]
					#print([key for key in layer.__dict__.keys()])
					vals = collections.OrderedDict()
					if self.log_list is not None:
						for attr in self.log_list:
							# should try this and if it fails then raise
							if hasattr(layer, attr):
								vals[attr] = getattr(layer, attr)
							else:
								raise AttributeError('Layer ' + str(i) + ' does not have the attribute: ' + str(attr))
					#after the list add it to l
					if self.log_list is None:
						if hasattr(input_class, 'get_layers_info'):
							l =  getattr(input_class, 'get_layers_info')()
	
						else:
							raise AttributeError('Model must have get_layers_info function')

					l['Layer_'+str(i)] = vals
			else:
				raise AttributeError('Model class should have model attribute containing the layers of the modell')

			self.logs['epoch_' + str(self.current_epoch)] = l
		# increment the epochs!
		self.current_epoch += 1

	def on_training_end(self, input_class):
		if self.return_type == 'dict' or self.return_type is None:
			self.save()
		if self.return_type == 'list':
			self.logs = convert_logs_to_lists(self.logs)
			self.save()
		if self.return_type == 'numpy.ndarray':
			self.logs = convert_logs_to_numpy_arrays(self.logs)
			self.save()
		else:
			self.save()

class ModelLogger(BaseLogger):

	def __init__(self, save_name, epoch_per_log=1, return_type=None):
		
		if not isinstance(save_name, str):
			raise TypeError('Save name should be a string')
		self.save_name = save_name

		if not isinstance(epoch_per_log, int):
			raise TypeError('Number of epochs per log must be an integer. You inputted: ' + str(type(epoch_per_log)))

		if epoch_per_log <=0:
			raise ValueError('Epoch per log must be a positive nonzero number')

		# initizliase the logs
		self.logs = collections.OrderedDict()
		self.current_epoch = 0
		self.epoch_per_log = epoch_per_log

		# sort out the return type
		self.return_type = return_type
		if return_type is not None and return_type not in ['list', 'dict','numpy.ndarray']:
			print("Return type is not recognised. Will return in the defualt dict of dicts format")
			self.return_type = None

	def on_training_begin(self, input_class):
		# get the model metadata
		if hasattr(input_class, 'get_model_metadata'):
			self.logs = input_class.get_model_metadata()
		else:
			print("No model metadata found for the logger")

	def log(self, input_class):
		if self.current_epoch % self.epoch_per_log == 0:
			if hasattr(input_class, 'get_layers_info'):
				layer_info_func = getattr(input_class, 'get_layers_info')
				if layer_info_func is callable:
					l = input_class.layer_info_func()
				else:
					raise TypeError('Layer information function from the model must be a callable function')
			else:
				raise AttributeError('Model has no get_layers_info function')
			self.logs['epoch_' + str(self.current_epoch)] = l
		self.current_epoch +=1

class FilterLogger(BaseLogger):

	def __init__(self, save_name,filter_key, epoch_per_log=1, return_type=None):
		
		if not isinstance(save_name, str):
			raise TypeError('Save name should be a string')
		self.save_name = save_name

		if not isinstance(epoch_per_log, int):
			raise TypeError('Number of epochs per log must be an integer. You inputted: ' + str(type(epoch_per_log)))

		if epoch_per_log <=0:
			raise ValueError('Epoch per log must be a positive nonzero number')

		# initizliase the logs
		self.logs = collections.OrderedDict()
		self.current_epoch = 0
		self.epoch_per_log = epoch_per_log
		if not isinstance(filter_key, str):
			raise TypeError('Filter key should be a string')
		self.filter_key = filter_key

		# sort out the return type
		self.return_type = return_type
		if return_type is not None and return_type not in ['list', 'dict','numpy.ndarray']:
			print("Return type is not recognised. Will return in the defualt dict of dicts format")
			self.return_type = None

	def log(self, input_class):
		if self.current_epoch % self.epoch_per_log ==0:
			# this is meant to filter dictionaries!
			if hasattr(input_class, 'get_layers_info'):
				layer_info_func = getattr(input_class, 'get_layers_info')
				new_dict = collections.OrderedDict()
				if callable(layer_info_func):
					l = getattr(input_class, 'get_layers_info')()
					for k,v in l.items():
						filt = filter_dict(v, self.filter_key)
						new_dict[k] = filt
						#print filt.keys()
				else:
					raise TypeError('Layer info function is not callable function. Is actually of type: ' + str(type(layer_info_func)))

				self.logs['epoch_' + str(self.current_epoch)] = new_dict
			else:
				raise AttributeError('Model has no get layers info function')
		self.current_epoch +=1


#aliases for quick shortcuts to loggers
def WeightsLogger(save_name, epoch_per_log=1, return_type=None):
	return FilterLogger(save_name, 'weights', epoch_per_log, return_type)

def ActivationsLogger(save_name, epoch_per_log=1, return_type=None):
	return FilterLogger(save_name, 'activations', epoch_per_log, return_type)

def PredictionsLogger(save_name, epoch_per_log=1, return_type=None):
	return FilterLogger(save_name, 'predictions', epoch_per_log, return_type)

def PredictionErrorLogger(save_name, epoch_per_log=1, return_type=None):
	return FilterLogger(save_name, 'prediction_errors', epoch_per_log, return_type)

def LossLogger(save_name, epoch_per_log=1, return_type=None):
	return FilterLogger(save_name, 'loss', epoch_per_log, return_type)



def combine_logs(logs):
	
	if not isinstance(logs, list):
		raise TypeError('List of logs must be a list. You inputted: ' + str(type(logs)))
	combined_dict = logs[0]
	for log in logs:
		if isinstance(log, str):
			#try loading it as if a filename
			log = load(log)
		if hasattr(log, 'get_logs'):
			# i.e. it's a logger instance
			log = log.get_logs()
		if hasattr(log, 'items'):
			# i.e. it's a dictionary of some kind treat it as such!
			for epoch_num, epoch_dict in log.items():
				for layer_num, layer_dict in epoch_dict.items():
					for attr, arr in layer_dict.items():
						combined_dict[epoch_num][layer_num][attr] = arr
		else:
			raise TypeError('Type of log not understood. You inputted: ' + str(type(log)))

	return combined_dict

def write_logs_to_file(logs, fname,max_line_width=None, precision=None, suppress_small=None):
	logger = BaseLogger()
	logger._set_logs(logs)
	logger.write_to_file(fname, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)
	return


def generate_default_logger(save_base):
	if not isinstance(save_base, str):
		raise TypeError('Save base name should be a string since it is where the logs files will be saved to')
	plogger = PredictionsLogger(save_name=save_base + '_prediction_logs')
	pelogger = PredictionErrorLogger(save_name=save_base + '_prediction_error_logs')
	alogger = ActivationsLogger(save_name=save_base + '_activations_logs')
	wlogger = WeightsLogger(save_name=save_base + '_weights_logs')
	losslogger = LossLogger(save_name=save_base + '_loss_logs')
	return [plogger, pelogger, alogger, wlogger, losslogger]
