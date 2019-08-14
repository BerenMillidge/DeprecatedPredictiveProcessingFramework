
import numpy as np
from utils import *

class BaseCallback():

	def __init__(self):
		pass

	def on_model_initialization(self, input_class):
		pass

	def on_training_begin(self, input_class):
		pass

	def on_epoch_begin(self, input_class):
		pass

	def on_epoch_end(self, input_class):
		#print("in base callback on epoch end!")
		pass

	def on_training_end(self, input_class):
		pass


class TerminateOnNaN(BaseCallback):

	def __init__(self, filter_keys=['weights','activations'], verbose=False):
		if isinstance(filter_keys, str):
			self.filter_keys = [filter_keys]
		else:
			self.filter_keys = filter_keys

		self.verbose = verbose

	def on_epoch_end(self, input_class):
		layers_dict = getattr(input_class,'get_layers_info')()
		for k,v in layers_dict.items():
			for key, value in v.items():
				if key in self.filter_keys:
					if np.any(np.isnan(value)) or np.any(np.isinf(value)):
						if self.verbose:
							print("NAN or infinity detected in: " + str(k) + " " + str(key) + ".")
						input_class._set_stop_training(True)
						raise ValueError('Callback stopped training since NaN or infinity detected')
		return



class ModelCheckpointer(BaseCallback):

	def __init__(self, save_base, epoch_per=1):
		if not isinstance(save_base, str):
			raise TypeError('Save base must be a string as its a filename to save the file')
		self.save_base = save_base
		if not isinstance(epoch_per, int):
			raise TypeError('Epoch per must be an integer number of epochs to checkpoint the model after')
		self.epoch_per = epoch_per

	def on_epoch_end(self, input_class):
		if input_class.current_epoch % self.epoch_per == 0:
			input_class.save(save_base + '_epoch_' + str(input_class.current_epoch))
		return

	def on_training_end(self, input_class):
		# if epoch per is -1 save at the end
		if self.epoch_per == -1:
			input_class.save(save_base+'_epoch_' + str(input_class.current_epoch))
		else:
			pass

class BaseStopMonitor(BaseCallback):

	def __init__(self, monitor_attr, monitor_func, result_func=None, layer_num=None):
		self.monitor_attr = monitor_attr
		self.monitor_func = monitor_func
		self.result_func = result_func or self._stop_training_func
		self.layer_num = layer_num

	def _stop_training_func(self, input_class):
		input_class._set_stop_training(True)



	def on_epoch_end(self, input_class):
		layers_dict = getattr(input_class, 'get_layers_info')()
		for k, v in layers_dict.items():
			if hasattr(v, monitor_attr):
				if monitor_func(v[monitor_attr], self, input_class):
					result_func(input_class)
			else:
				raise AttributeError('Monitor attribute ' + str(monitor_attr) + ' not found on layer.')


def NormMonitor(norm, attr):

	def _check_norm(arr, monitor_class, model_class):
		if np.linalg.norm(arr) > norm:
			return True
		return False

	return BaseStopMonitor(attr, _check_norm)

def WeightNormMonitor(norm):
	return NormMonitor(norm, 'weights')


def ActivationsNormMonitor(norm):
	return NormMonitor(norm, 'activations')

def LossEarlyStopper(required_decrease, epoch_leeway):

	def _stop_training_func(self, input_class):
		print("Loss Early Stopper callback stopping training since loss has increased.")
		input_class._set_stop_training(True)

	def _check_loss(arr, monitor_class, model_class):
		if model_class.current_epoch % epoch_leeway == 0:
			# i.e. if at the epoch leeway check limit
			if not hasattr(monitor_class, 'prev_total_loss') or monitor_class.prev_total_loss is None:
				monitor_class.prev_total_loss = model_class._total_loss()

			prev_loss = monitor_class.prev_total_loss
			curr_loss = model_class._total_loss()
			if curr_loss - required_decrease >= prev_loss:
				return True
			else:
				return False
		return False

	return BaseStopMonitor('loss', _check_loss, _stop_training_func)


class ModelSaver(BaseCallback):


	def __init__(self, save_name, verbose=True):
		if not isinstance(save_name, str):
			raise TypeError('File name to save model must be a string')

		self. save_name = save_name
		self.verbose = verbose

	def on_training_end(self, input_class):
		if self.verbose:
			print("Saving model...")
		input_class.save(self.save_name)
		return

class DefaultLossPrinter(BaseCallback):

	def on_epoch_end(self, input_class):
		print("Loss: " + str(input_class._total_loss()))

def generate_default_callbacks():
	return [TerminateOnNaN(), DefaultLossPrinter()]

