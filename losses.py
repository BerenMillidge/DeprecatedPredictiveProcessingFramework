
import numpy as np
from utils import *


EPS = 1e-8 # for numerical stability

def subtraction(act, pred):
	return act - pred

def normalised_subtraction(act, pred):
	act = todist(act)
	pred = todist(pred)
	return subtraction(act, pred)

def division(act, pred):
	return act/pred

def multiplication(act, pred):
	return act * pred

def LSQ(act, pred):
	return np.square(act - pred)

def Kldiv(act, pred):
	act = todist(act)
	pred = todist(pred)
	return act * (np.log(eps + act,2) - np.log(eps + pred, 2))

def unnormalisedKLdiv(act, pred):
	return act * (np.log(eps+act, 2) - np.log(eps + pred, 2))

def crossentropy(act, pred):
	return act * np.log(eps + pred, 2)

def normalised_crossentropy(act, pred):
	act = todist(act)
	pred = todist(pred)
	return crossentropy(act, pred)

def absolute(act, pred):
	return np.abs(act - pred)

def hinge(act, pred):
	return np.max(0, 1 - act * pred)

def squared_hinge(act, pred):
	return np.square(hinge)

def log_error(act, pred):
	return np.log(eps+act, 2) - np.log(eps + pred, 2)

def huber(act, pred, delta=1):
	diff = np.abs(act - pred)
	if diff <= delta:
		return 0.5 * np.square(diff)
	else:
		return delta * diff - 0.5 * np.square(delta)




