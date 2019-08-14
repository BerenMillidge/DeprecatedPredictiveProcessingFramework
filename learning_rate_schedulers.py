from __future__ import division
import numpy as np

def RB_scheduler(learning_rate, epoch):
	#divide by 1.05 every 40 epochs - RB paper https://www.cs.utexas.edu/users/dana/nn.pdf
	if epoch % 40 == 0:
		return learning_rate/1.015
	else:
		return learning_rate

class BaseLearningRateScheduler():

	def __init__(self, learning_rate, epoch):
		self.learning_rate = learning_rate
		self.epochs = epoch

	def get_learning_rate(self):
		return self.learning_rate

	def get_epoch(self,epoch):
		return self.epoch

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def set_epoch(self, epoch):
		self.epoch = epoch

	def call(self):
		return self.learning_rate


class RB_Scheduler(BaseLearningRateScheduler):

	def call(self):
		if epochs % 40 == 0:
			return self.learning_rate / 1.015
		else:
			return self.learning_rate


class ConstantDivisorScheduler(BaseLearningRateScheduler):

	def __init__(self, learning_rate, epoch, divisor, per_epoch):
		self. learning_rate = learning_rate
		self.epoch = epoch
		self.divisor = divisor
		self.per_epoch = per_epoch

	def call(self):
		if epoch % per_epoch == 0:
			return self.learning_rate / divisor
		else:
			return self.learning_rate




