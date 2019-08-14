from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import cPickle as pickle
from RB_layer import *
from model import *
from data_generator import *
from utils import *
from loggers import *
from callbacks import *
import pdb

def load_image(fname):
	if fname.split('.')[-1] == 'npy':
		#is npy file
		return np.load(fname)
	else:
		return pickle.load(open(fname, 'rb'))

def get_image_patch(img,shape):
	h,w = shape
	ih, iw,ch= img.shape
	height = int(np.random.uniform(low=0+h, high=ih-h))
	width = int(np.random.uniform(low=0+w, high=iw-w))
	patch = img[height:height+h, width:width+w,0]
	return patch



def get_image_patches(img, N, shape):
	patches = []
	for i in xrange(N):
		patch = get_image_patch(img, shape)
		patches.append(patch)
	return np.array(patches)



def plot_PE_per_layer(pe,shape):
	for i in xrange(len(pe)//3):
		fig = plt.figure()
		fig.add_subplot(131)
		plt.title(str(i*3))
		plt.imshow(np.reshape(unflatten(pe[(i*3)], shape), shape), label=str(i*3), cmap='gray')
		fig.add_subplot(132)
		plt.title(str((i*3)+1))
		plt.imshow(np.reshape(unflatten(pe[(i*3)+1], shape), shape), label=str((i*3)+1),cmap='gray')
		fig.add_subplot(133)
		plt.title(str((i*3)+2))
		plt.imshow(np.reshape(unflatten(pe[(i*3)+2], shape), shape), label=str((i*3)+2),cmap='gray')
		plt.legend()
		fig.tight_layout()
		plt.show()


#put this in the data generator eventually!
def create_mnist_data(val):
	a = get_N_mnist_of_val(500,val)
	ar =[] 
	for i in xrange(len(a)):
		ar.append(flatten(a[i]))
	ar = np.array(ar)
	print ar.shape
	return ar


imgs = load_image('test_images')
print imgs.shape
#print imgs.shape
#plt.imshow(imgs[0])
#plt.show()


#patches = get_image_patches(imgs[0], 10, (20,20))
#print patches.shape

def plot_correct_prediction_errors(c,p,pe):
	fig = plt.figure()
	fig.add_subplot(131)
	plt.title('Original image')
	plt.imshow(c, label='correct')
	fig.add_subplot(132)
	plt.title('Predicted image')
	plt.imshow(p, label='prediction')
	fig.add_subplot(133)
	plt.title('Prediction errors')
	plt.imshow(pe, label='prediction error')
	plt.legend()
	fig.tight_layout()
	plt.show()


#simple_pattern = np.array([1,0,1,0])
#simple_pattern = patches[0]
simple_pattern = imgs[0]
#print simple_pattern.shape
#simple_pattern = simple_pattern[:,:,0]
#plt.imshow(simple_pattern)
#plt.show()
#p#rint simple_pattern.shape
#print simple_pattern[0].shape
#print flatten(simple_pattern).shape
#plt.imshow(simple_pattern)
#plt.show()
patch = normalize(flatten(simple_pattern))
#patch = np.reshape(patch, (len(patch),1))
#plt.imshow(simple_pattern[0])
#plt.show()
#patch = np.array([1,1,1,1,1,1])
#patch = np.array([0,0,0,0,0,0,0])
# yeah, linear learning works really well. nonlinear, not at al!
#patch = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#patch = np.array([1,1])
"""
layer = RB_Layer(10, patch)
for i in xrange(200):
	p,pe = layer.run(patch, None)

#	print "epoch: " + str(i)
	#print p.shape
	#print pe.shape
	#print p
	p2 = np.reshape(patch, (len(patch),1))
	#if i % 10 == 0:
	#	plot_correct_prediction_errors(p2, p,pe)
"""

def create_mnist_val_average(val):
	ar =create_mnist_data(val)
	patch = normalize(ar)

	model = RB_Model(patch, N_per_layer=[16,10,5], N_layers = 3, learning_rate = 0.0001, epochs=100, batch_size=1)
	plogger = PredictionsLogger(save_name='logs/prediction_logger_test_3_linear')
	pelogger = PredictionErrorLogger(save_name='logs/prediction_errors_logger_test_3_linear')
	alogger = ActivationsLogger(save_name='logs/activations_logger_test_3_linear')
	wlogger = WeightsLogger(save_name='logs/weights_logger_test_3_linear')
	losslogger = LossLogger(save_name='logs/loss_logger_test_3_linear')
	loggers = [plogger, pelogger, alogger, wlogger, losslogger]
	cb = TerminateOnNaN()
	#pdb.set_trace()
	weights, activations = model.train(loggers=loggers, callbacks=cb)
	
	p,pe = model.get_prediction_and_prediction_error_list()
	prediction = model.generate_average_prediction()
	#wlogger.write_to_file('logs/weight_logs_test.txt')

	comb = combine_logs([alogger.get_logs(), wlogger.get_logs()])
	write_logs_to_file(comb, 'logs/combined_logs_test')
	print patch.shape
	print prediction.shape
	plt.imshow(np.reshape(prediction, (28,28)))
	plt.show()
	#alogs = alogger.get_logs()
	#print type(alogs)
	#print len(alogs)
	#print np.max(alogs['epoch_99']['layer_1']['activations'])
	#prediction = model.propagate_top_down_prediction(np.array([-0.1,-0.1,0.1,0.1,0.1]))
	#print prediction.shape
	#prediction = np.reshape(prediction, (28,28))
	#plt.imshow(prediction)
	#plt.show()
	#model.save('logs/save_model_test')

	#patch = unflatten(patch[0], (28,28))
	#p = np.reshape(unflatten(np.reshape(p[0], (784,1)), (28,28)), (28,28))
	#pe = unflatten(pe[0], (28,28))
	#plot_correct_prediction_errors(patch, p, pe)
	#m = load_model('logs/save_model_test')
	
	#p, pe = m.get_prediction_and_prediction_error_list()
	#p = np.reshape(p[0], (28,28))
	#plt.imshow(p)
	#plt.show()
	
	#prediction = m.propagate_top_down_prediction(np.array([-0.1,-0.1,0.1,0.1,0.1]))
	#print prediction.shape
	#prediction = np.reshape(prediction, (28,28))
	#plt.imshow(prediction)
	#plt.show()

	#model2 = load_model('logs/save_model_test')
	#p2, pe2 = model.get_prediction_and_prediction_error_list()
	#p2 = np.reshape(unflatten(np.reshape(p2[0], (784,1)), (28,28)), (28,28))
	#pe2 = unflatten(pe2[0], (28,28))
	#plot_correct_prediction_errors(patch, p2, pe2)
	#p2 = np.reshape(patch[0], (784,1))
	# p2
	#print prediction
	#print len(p)
	#print len(pe)
	#print type(p)
	##print p[0].shape
	#print pe[0].shape
#	pe3 = np.reshape(pe[0], (len(pe[0]),1))

#	pe2 = p2 - prediction
	#print pe2
	#print pe3
	#print prediction
	#print prediction.shape
	#print weights[0].shape
	#print pe2
#	#print total_PE(pe2)
#	shape = (28,28)

	#img = np.reshape(unflatten(p2, shape), shape)
	#pred2 = np.reshape(unflatten(prediction, shape), shape)
	#plt.imshow(pred2)
	#plt.show()
	#pe3 = np.reshape(unflatten(pe2, shape), shape)
	#pred2 = np.reshape(pred2, (20,20))
	#print pred2.shape 
	#plt.imshow(pred2)
	#plt.show()
	#print img
	#print pred2
	#print pe3
	#print p[0]
	#print patch.shape
	##print p.shape
	#print pe.shape
	
	
	##plot_PE_per_layer(pe, (20,20))
	#plot_PE_per_layer(p, (20,20))

	#we = weights[0]
	##weights = np.reshape(weights, (len(weights),))
	#w = np.reshape(unflatten(we[:,0], shape),shape)
	#print w.shape
	#plt.imshow(w)
	#plt.show()
	#w2 = np.reshape(unflatten(we[:,1], shape),shape)
	#print w.shape
	##plt.imshow(w)
	#plt.show()
	#print activations[0]
	#plt.imshow(weights[0])
	#plt.show()
	#p#rint activations[0].shape
	#plt.imshow(activations[0])
	#lt.show()
	# what will it do on the miage files?
	#print len(pe)
	#print pe[0].shape
	#print pe[1].shape
	# okay, that's perfect that's precisely what I want!
	#plt.imshow(np.reshape(unflatten(pe[0],shape),shape))
	#plt.show()
	#plt.imshow(pe[1])
	#plt.show()

def all_mnist_test(save_name=None):
	xtrain = np.load('mnist_xtrain.npy')
	mnist = normalize(flatten_data(xtrain))
	model = RB_Model(mnist, N_per_layer=[20,10], N_layers = 2, learning_rate = 0.0001, epochs=100, batch_size=1)
	plogger = PredictionsLogger(save_name='logs/all_mnist_predictions_linear')
	pelogger = PredictionErrorLogger(save_name='logs/all_mnist_prediction_errors_linear')
	alogger = ActivationsLogger(save_name='logs/all_mnist_activations_linear')
	wlogger = WeightsLogger(save_name='logs/all_mnist_weights_linear')
	losslogger = LossLogger(save_name='logs/all_mnist_loss_linear')
	loggers = [plogger, pelogger, alogger, wlogger, losslogger]
	cb = TerminateOnNaN()
	weights, activations = model.train(loggers = loggers, callbacks = cb)
	p,pe = model.get_prediction_and_prediction_error_list()
	if save_name is not None:
		model.save(save_name)
	prediction = model.generate_average_prediction()
	#reshape prediction to seem reasonable
	shape = (28,28)
	print prediction.shape
	prediction = np.reshape(unflatten(prediction, shape),shape)
	plt.imshow(prediction)
	plt.show()
	#plot_PE_per_layer(pe,shape)
	#plot_PE_per_layer(p,shape)
	print p[0]
	print p[1]

#all_mnist_test('models/all_mnist_test')
#model = load_model('models/all_mnist_test')
#p = model.propagate_top_down_prediction(np.array([0.1,-0.1,0.1,-0.1,0.1,0.1,-0.1,0.1,0.1,0.1]))
#p = model.generate_average_prediction()
#p = np.reshape(p, (28,28))
#plt.imshow(p)
#plt.show()

def train_rao_ballard_epoch(input_patches, l1s, l2, l2_p, learning_rate,save_all = False):

	l1pes = [0 for i in xrange(len(l1s))]
	l1ps = [0 for i in xrange(len(l1s))]
	l1tdes = [0 for i in xrange(len(l1s))]
	l2_pe = 0
	l2_tde = 0
	if save_all:
		all_l1pes = []
		all_l1ps = []
		all_l2_ps = []
		all_l2pes = []
		all_l1_tdes = []
		all_l2tdes = []
	for patchlist in input_patches:
		assert len(patchlist) == len(l1s), 'Number of patches in patch list must be the same as number of l1 neuron groups'
		for i in xrange(len(l1s)):
			#print "in layer 1 : " , i
			# run the first layer

			l1p, l1pe, l1tde, l1_tdr, l1_loss = l1s[i].run(patchlist[i],l2_p[i], learning_rate = learning_rate)
			#print l1p
			l1ps[i] = l1p
			l1pes[i] = l1pe
			l1tdes[i] = l1tde
			# run the second layer
			#print "running layer 2"
		l2_p, l2_pe, l2_tde, l2_tdr, l2_loss = l2.run(flatten(combine_vectors(l1tdes)), None, learning_rate = learning_rate)
		if save_all:
			all_l1ps.append(l1ps)
			all_l1pes.append(l1pes)
			all_l1tdes.append(l1tdes)
			all_l2_ps.append(l2_p)
			all_l2_pes.append(l2_pe)
			all_l2tdes.append(l2_tde)
	if save_all:
		return all_l1ps, all_l1pes, all_l2_ps, all_l2pes, all_l1tdes, all_l2tdes

	return l1ps, l1pes,  l1tdes, l2_p, l2_pe, l2_tde
def split_into_patches(imgs, num_patches):
	patchlist = []
	for k in xrange(num_patches):
		patchlist.append([])
	for i in xrange(len(imgs)//num_patches):
		for j in xrange(num_patches):
			patchlist[j].append(imgs[i+j])
	l = []
	for z in xrange(num_patches):
		l.append(np.array(patchlist[z]))
	return l
	# let's see if this works at all!

def split_into_patches_other(imgs, num_patches):
	patchlist = []
	for i in xrange(len(imgs)//num_patches):
		l = []
		for j in xrange(num_patches):
			l.append(flatten(imgs[i+j]))
		patchlist.append(np.array(l))
	return patchlist


image_shape = (15,15)
def rao_and_ballard_model(fname, epochs=1, learning_rate =0.00001, save_name=None):

	imgs = np.load(fname)
	img = normalize(imgs[0])
	image_shape = (50, 50) 
	num_patches = 3

	image_spacing = 3
	neurons_L1 = 124
	neurons_L2 = 32
	num_data = 50
	#num_images = 50
	# create the actual dataset!
	num_images = len(imgs)
	print type(imgs)
	print imgs.shape
	patchlist = split_into_patches_other(imgs, num_patches)
	print type(patchlist)
	print len(patchlist)
	print type(patchlist[0])
	print len(patchlist[0])
	#print patchlist[0].shape

	#
	#patchlist = []
	#for k in xrange(num_images):
	#	for i in xrange(num_data):
	#		patches = create_N_image_patches(imgs[k], num_patches, image_shape)
			# flatten the patches
	#		for j in xrange(len(patches)):
	#			patches[j] = flatten(patches[j])
	#		patchlist.append(patches)
	layers = []
	ps = []
	pes = []
	l2_ps = []
	l2_pes = []
	l1tdes = []
	l2tdes = []
	#level 1 initialization
	for i in xrange(num_patches):
		l = RB_Layer(neurons_L1, patchlist[i])
		layers.append(l)

	#level2 initialization0
	l2 = RB_Layer(neurons_L2,np.zeros((neurons_L1*num_patches)))
	print l2.get_bottom_up_input().shape
	# initialise original l2_p to nothing
	l2_p =[None for i in xrange(num_patches)]
	for i in xrange(epochs):
		print "Rb epoch: ", i
		l1ps, l1pes,l1tde, l2_prediction, l2_pe, l2tde = train_rao_ballard_epoch(patchlist, layers, l2,l2_p, learning_rate)
		assert len(l2_p) % num_patches == 0, 'L2 prediction has wrong dimensions to allow splitting'
		# then split
		s = len(l2_p) // num_patches
		for i in xrange(num_patches):
			l2_p[i] = l2_prediction[i*s: (i+1)*s]
		ps.append(l1ps)
		pes.append(l1pes)
		l2_ps.append(l2_p)
		l2_pes.append(l2_pe)
		l1tdes.append(l1tde)
		l2tdes.append(l2tde)

	if save_name is not None:
		assert type(save_name) is str,'Save name must be a string'
		layer_list = [layers, l2]
		try:
			save(layer_list, save_name)
		except OSError:
			raise ValueError('Cannot find file with that name')

	return ps, pes, l1tdes, l2_ps, l2_pes, l2tdes, patchlist


def reconstruct_rb_image(vect, num_patches, shape):
	# let's try reconstructing it the sane way and hope it helps
	assert len(vect) % num_patches == 0, 'Image cannot be reconstrcted as vector length is not a multiple of the image patches'
	images = []
	s = len(vect) // num_patches
	print s
	for i in xrange(num_patches-1):
		images.append(vect[i*s:(i+1)*s])
	images.append(vect[(num_patches-1)*s:len(vect)])
	for j in xrange(len(images)):
		print "image shape " ,images[j].shape
		img= unflatten(images[j], shape)
		print img.shape
		images[j] = img
	return images



#ps, pes, l1tdes, l2_ps, l2_pes, l2tdes, patchlist = rao_and_ballard_model(save_name='models/RB_model_1')
#results = [ps, pes, l1tdes, l2_ps, l2_pes, l2tdes, patchlist]
#save(results, 'results/RB_results_1')

def plot_RB_predictions_actuality(images, patches):
	num_patches = 3 # hardcode this atm
	fig = plt.figure()
	for i in xrange(num_patches):
		fig.add_subplot(3,i+1, 1)
		plt.imshow(images[i])
		plt.title('Prediciton')
		fig.add_subplot(3, i+1,2)
		plt.imshow(patches[i])
		plt.title('Image patch')
	fig.tight_layout()
	plt.show()
	return fig

def plot_predictions_through_time(ps, step_size):
	figs = []
	for i in xrange(len(ps)):
		if i % step_size == 0:
			fig = plt.figure()
			plt.title('Epoch ' + str(i))
			fig.add_subplot(131)
			plt.imshow(ps[i][0])
			fig.add_subplot(132)
			plt.imshow(ps[i][1])
			fig.add_subplot(133)
			plt.imshow(ps[i][2])
			fig.tight_layout()
			plt.show()
			figs.append(fig)

	return figs


# let's try to figure this out!
def Sequential_RB_model_test():
	ar =np.load('blurred_rb_data_test.npy')
	#plt.imshow(ar[0])
	#plt.show()
	#print np.max(ar)
	#print ar[0]
	#print np.mean(ar)
	patch = flatten_dataset(normalize(ar))
	print patch.shape
	print np.max(patch)
	print patch[0]
	
	print np.mean(patch)
	

	model = RB_Model(patch, N_per_layer=[124,36,10], N_layers = 3, learning_rate = 0.0001, epochs=100, batch_size=1)
	plogger = PredictionsLogger(save_name='logs/sequential_RB_prediction_logger2')
	pelogger = PredictionErrorLogger(save_name='logs/sequential_RB_pe_logger2')
	alogger = ActivationsLogger(save_name='logs/sequential_RB_activations_logger2')
	wlogger = WeightsLogger(save_name='logs/sequential_RB_weights_logger2')
	losslogger = LossLogger(save_name='logs/sequential_RB_loss_logger2')
	modelSaver = ModelSaver(save_name='models/SequentialRBModel')
	loggers = [plogger, pelogger, alogger, wlogger, losslogger, modelSaver]
	cb = [TerminateOnNaN(), modelSaver]
	#pdb.set_trace()
	weights, activations = model.train(loggers=loggers, callbacks=cb)
	
	p,pe = model.get_prediction_and_prediction_error_list()
	prediction = model.generate_average_prediction()
	#wlogger.write_to_file('logs/weight_logs_test.txt')
	
	comb = combine_logs([alogger.get_logs(), wlogger.get_logs()])
	write_logs_to_file(comb, 'logs/combined_logs_test')
	print patch.shape
	print prediction.shape
	plt.imshow(np.reshape(prediction, (50,50)))
	plt.show()

m = load_model('models/SequentialRBModel')
p = m.generate_average_prediction()
#plt.imshow(np.reshape(p, (50,50)))
#plt.show()
mu, var = m.top_layer_gaussian_stats()
l = m.top_layer_length()
# then sample a whole bunch of things
"""
for i in xrange(20):
	samp = np.random.normal(loc=mu, scale=0.1, size=l)
	print samp
	pred = m.propagate_top_down_prediction(samp)
	plt.imshow(np.reshape(pred, (50,50)))
	plt.show()

"""
inp = patch[0]
print inp.shape
l = m.top_layer_length()
print l
a = m.test()
print a
res_acts = m.forward_pass_to_final_activations(inp)
# then see if can be regenerated
print res_acts
pred = m.propagate_top_down_prediction(res_acts)
plt.imshow(np.reshape(pred, (50,50)))
plt.show()
plt.imshow(np.reshape(inp, (50,50)))
plt.show()
"""
#ps, pes, l1tdes, l2_ps, l2_pes, l2tdes, patchlist = load('results/RB_results_1')


print type(ps)
print len(ps)
#print ps.shape
# okay yeah it's one per epoch isn't it?
last = ps[-1]
print len(last)


print len(ps)
print len(ps[0])
print ps[0][0].shape
#images = reconstruct_rb_image(ps[0][0],3,(15,15))
shape = image_shape
image = np.reshape(unflatten(ps[-1][1], shape), shape)
print image.shape
print len(patchlist)
print len(patchlist[0])
print patchlist[0][0].shape
actual = unflatten(patchlist[0][1], shape)
print actual.shape


image2 = np.reshape(unflatten(ps[-1][0], shape), shape)
actual2 = unflatten(patchlist[0][0], shape)


fig = plt.figure()
fig.add_subplot(121)
plt.imshow(actual)
fig.add_subplot(122)
plt.imshow(image)
plt.show()

fig = plt.figure()
fig.add_subplot(121)
plt.imshow(actual2)
fig.add_subplot(122)
plt.imshow(image2)
plt.show()
"""
#plt.imshow(image)
#plt.show()
#for image in images:
#	print image.shape
#	plt.imshow(image)
#	plt.show()
