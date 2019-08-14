import numpy as np

class BaseCombination_Layer():

	def __init__(self, input_vect, comb_function, params=None):
		self.input_vect = input_vect
		self.comb_function = comb_function
		self.params = params

		# set transformer attribute
		self.__transformer = True
		#set trainable attribute
		self.__trainable = False

	def _call(self):
		# the general key function
		return comb_function(input_vect, **params)

	def get_input_vector(self):
		return self.input_vect

	def get_comb_function(self):
		return self.comb_function

	def get_params(self):
		return self.params

	def get_layer_info(self):
		info = {}
		info['input_vect'] = self.input_vect
		info['comb_function'] = self.comb_function
		info['params'] = self.params
		return info




def matrix_stack(vectlist):
	return np.vstack(vectlist)

def matrix_concatenate(vectlist, axis):
	return np.concatenate(vectlist, axis=axis)

def vector_append(vectlist):
	return np.hstack(vectlist)

# and so forth!
