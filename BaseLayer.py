
import numpy as np
from initializers import *

class BaseLayer():

	def __init__(self, name, bottom_up_input, top_down_input):
		self.name = name
		self.bottom_up_input = bottom_up_input
		self.top_down_input = top_down_input

	

		self._layer_type = 'trainable'

		self._run = False

	def _get_bottom_up_input(self):
		return self.bottom_up_input

	def _get_top_down_input(self):
		return self.top_down_input

	def _get_bottom_up_output(self):
		pass

	def get_top_down_output(self):
		pass

	def get_bottom_up_projections(self):
		pass

	def get_bottom_up_projectors(self):
		pass

	def get_top_down_projections(self):
		pass

	def get_top_down_projectors(self):
		pass

	def _add_bottom_up_projector(self):
		pass

	def _add_top_down_projector(self):
		pass 

	def _add_bottom_up_projection(self):
		pass

	def _add_top_down_projection(self):
		pass

	def get_name(self):
		return self.name

	def run(self, bottom_up_input, top_down_input):
		pass

	def on_training_begin(self):
		pass

	def on_training_end(self):
		pass
	#def on epoch_begin(self):
	#	self.losses = []

	def on_epoch_begin(self):
		self.losses = []

	def on_epoch_end(self):
		self.loss = np.mean(self.losses)


"""
class BasicLinearPPLayer():
    
    def __init__(self,learning_rate, bottom_up_size, top_down_size, input_layer=False, weight_update_ratio = 1):
        self.learning_rate = learning_rate
        self.bottom_up_size = bottom_up_size
        self.top_down_size = top_down_size
        self.weights = np.random.normal(0, 0.1, [bottom_up_size, top_down_size])
        self.latents = np.random.normal(0, 0.1, [bottom_up_size,1])
        self.input_layer = input_layer
        self.weight_update_ratio = weight_update_ratio
               
        
    def upward_projection(self,pe, w):
        return np.dot(w.T, pe)

    def update_weight(self,pe, u):
        return learning_rate * np.dot(pe, u.T)

    def update_cause_unit(self,bu , pe):
        return learning_rate * np.subtract(bu, pe)

    def top_down_prediction(self,w, td):
        return np.dot(w, td)

    def prediction_error(self,u, pred):
        return np.subtract(u, pred)

    def get_loss(self):
    	return np.dot(self.pe1.T, self.pe1)
    
    
    def run(self, bottom_up, top_down, training=True):
    	self.bottom_up = bottom_up
    	self.top_down = top_down
		if self.input_layer:
			self.latents = bottom_up
		if self.top_down is None:

			self.pe1 = prediction_error(self.latents, 0)
            #print(pe1.shape)
            #print(bottom_up.shape)
            self.latents += self.update_cause_unit(self.bottom_up, self.pe1)
            self.preds = self.pe1
            return None, self.latents, self.weights, self.pe1, self.pe1
        else:
            self.preds = self.top_down_prediction(self.weights, self.top_down)
            self.pe1 = self.prediction_error(self.latents, self.preds)            
            self.up = self.upward_projection(self.pe1, self.weights)
            #print(np.sum(pred))
            #print(np.sum(pe1))
            #print(np.sum(up))
            if not self.input_layer:
                self.latents += self.update_cause_unit(self.bottom_up, self.pe1)
            if i % weight_update_ratio == 0 and training is True:
                self.weights += self.update_weight(self.pe1, self.top_down)
            
            return self.up, self.latents, self.weights, self.preds, self.pe1


class SigmoidLayer():
    
    def __init__(self,learning_rate, bottom_up_size, top_down_size, input_layer=False, weight_update_ratio = 1):
        self.learning_rate = learning_rate
        self.bottom_up_size = bottom_up_size
        self.top_down_size = top_down_size
        self.weights = np.random.normal(0, 0.1, [bottom_up_size, top_down_size])
        self.latents = np.random.normal(0, 0.1, [bottom_up_size,1])
        self.input_layer = input_layer
        self.weight_update_ratio = weight_update_ratio
               
        
    def sigmoidderiv(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def upward_projection(pe, w):
        return sigmoidderiv(np.dot(w.T, pe)) 
    def update_weight(predderiv, pe, u):
        return learning_rate * np.dot(pe * predderiv,  u.T)

    def update_cause_unit(bu , pe):
        return learning_rate * np.subtract(bu, pe)

    def top_down_prediction(w, td):
        return sigmoid(np.dot(w, td))

    def prediction_error(u, pred):
        return np.subtract(u, pred)

    def prediction_derivative(w,u):
        return sigmoidderiv(np.dot(w, u))
    
    pred = top_down_prediction(w1, u2)
        predderiv = prediction_deriv(w1, u2)
        pe1 = prediction_error(u1, pred)
        up1 = upward_projection(predderiv,pe1, w1)
        pe2 = prediction_error(u2, 0)
        w1 += update_weight(predderiv, pe1, u2)
        u2 += update_cause_unit(up1, pe2)
    
    
    def run(self, bottom_up, top_down, training=True):
        self.bottom_up = bottom_up
        self.top_down = top_down
        if self.input_layer:
            self.latents = self.bottom_up
        if self.top_down is None:
            self.prediction_errors = prediction_error(self.latents, 0)
            self.latents += self.update_cause_unit(self.bottom_up, self.prediction_errors)
            self.preds = self.prediction_errors
            return None, self.latents, self.weights, self.prediction_errors, self.prediction_errors
        else:
            self.preds = self.top_down_prediction(self.weights, self.top_down)
            self.prediction_errors = self.prediction_error(self.latents, self.preds)  
            self.predderiv = self.prediction_deriv(self.weights, top_down)
            self.up = self.upward_projection(self.predderiv,self.prediction_errors, self.weights)
            #print(np.sum(pred))
            #print(np.sum(pe1))

            #print(np.sum(up))
            if not self.input_layer:
                self.latents += self.update_cause_unit(self.bottom_up, self.prediction_errors)
            if i % weight_update_ratio == 0 and training is True:
                self.weights += self.update_weight(delf.predderiv,self.prediction_errors, self.top_down)
            
            return self.up, self.latents, self.weights, self.preds, self.prediction_errors

"""
class BasicDynamicalLayer(object):
    
    def __init__(self,sensory_dimension, top_down_dimension, layer_dimension,learning_rate = 0.01, sensory_variance=1, dynamical_variance=1, initializer= default_gaussian_initializer): # not that difficult to be honest... larning rate can be defined per lay or per network probably
        self.sensory_dimension = sensory_dimension
        self.top_down_dimension = top_down_dimension        
        self.layer_dimension = layer_dimension
        self.sensory_variance = sensory_variance
        self.dynamical_variance = dynamical_variance
        self.mu= np.random.normal(0, 0.1, [self.layer_dimension, 1])
        self.mus = []
        self.ezs = []
        self.ews = []
        self.sensory_weights = np.random.normal(0, 0.1, [self.sensory_dimension, self.layer_dimension])
        self.dynamical_weights = np.random.normal(0, 0.1, [self.top_down_dimension, self.layer_dimension])
        self.thetazs = []
        self.thetaws = []
        self.sensory_pe = 0
        self.dynamical_pe = 0
        self.learning_rate = learning_rate

        
    
    def run(self, sense_data, bottom_up_pe, top_down_mu, learning=False):
        #calculate prediction errors
        if top_down_mu is None:
            top_down_mu = np.zeros([self.layer_dimension,1])
        sense_data = np.reshape(sense_data, [len(sense_data), 1])
        #print(self.sensory_weights.shape)
        #print(self.mu.shape)
        #print(sense_data.shape)
        #print(sense_data)
        #print(type(sense_data))
        #print("\n")
        pred = np.dot(self.sensory_weights, self.mu)
        #print(pred.shape)
        #print(pred)
        #print(type(pred))
        self.sensory_pe = np.subtract(sense_data, pred)
        #print(self.sensory_pe.shape)
        #print(self.sensory_pe)
        self.dynamical_pe = top_down_mu - np.dot(self.dynamical_weights, self.mu)
        # update states
        #print(self.sensory_weights.T.shape)
        #print(self.sensory_pe.shape)
        #print(self.dynamical_weights.T.shape)
        #print(self.dynamical_pe.shape)
        #print(bottom_up_pe.shape)
        state_grad = np.dot(self.sensory_weights.T, self.sensory_pe) + np.dot(self.dynamical_weights.T, self.dynamical_pe) + bottom_up_pe
        self.mu += self.learning_rate * state_grad
        if learning:
            # update weights
            self.sensory_weights += self.learning_rate * (np.dot(self.sensory_pe, self.mu.T))
            self.dynamical_weights += self.learning_rate * (np.dot(self.dynamical_pe, self.mu.T))
            
        self.mus.append(self.mu)
        self.ezs.append(np.sum(self.sensory_pe))
        self.ews.append(np.sum(self.dynamical_pe)) # these are only for record keeping functions
        self.thetazs.append(self.sensory_weights)
        self.thetaws.append(self.dynamical_weights)
        pred = np.dot(self.sensory_weights, self.mu)
        return pred, self.dynamical_pe, self.mu  down!
