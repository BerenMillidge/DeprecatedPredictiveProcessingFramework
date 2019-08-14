import sys
import traceback

def get_stack_trace(e):
	type_, value_, traceback_ = sys.exc_info()
	tb = traceback.format_tb(traceback_)
	return str(type) + " " + str(value_) + " " + str(tb)

class ShapeError(Exception):

	def __init__(self, message, shape):
		self.message = message
		self.shape = shape

	def __str__(self):
		return "Incorrect shape:" + str(shape)

class LayerError(Exception):
	pass

class ModelError(Exception):
	pass

class TopologyError(Exception):
	pass 

class DataError(Exception):
	pass 

class DatasetError(Exception):
	pass

class InitializerException(Exception):

	def __init__(self, e):
		self.e = e
		self.tb = get_stack_trace(e)

	def __str__(self):
		return "Initializer Exception: " + self.tb
