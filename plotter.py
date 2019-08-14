import numpy as np
import matplotlib.pyplot as plt 

class BasePlotter():

	def __init__(self):
		pass

	def on_epoch_begin(self):
		pass

	def on_epoch_end(self):
		pass

	def on_training_begin(self):
		pass

	def on_training_end(self):
		pass

	def on_model_initialization(self):
		pass

	def plot(self):
		pass
