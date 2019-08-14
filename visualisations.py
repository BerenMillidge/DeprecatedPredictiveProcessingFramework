import numpy as np 
from utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def convert_dicts_into_lists(dicts):
	list_of_lists = []
	layer_lists = []

def convert_string_name_dict_into_list(dicts, name):
	master_list = []
	for i in xrange(len(dicts['epoch_0'])):
		#add the other lists
		master_list.append([])
	for i in xrange(len(dicts)):
		epoch_dict = dicts['epoch_'+str(i)]
		for j in xrange(len(epoch_dict)):
			layer_dict = epoch_dict['layer_'+str(j)]
			if name in layer_dict:
				master_list[j].append(layer_dict[name])
			else:
				raise AttributeError('Key : ' + str(name) + ' not in dict')
	return master_list

def convert_weight_dict_into_list(dicts):
	master_list = []
	for i in xrange(len(dicts['epoch_0'])):
		#add the other lists
		master_list.append([])
	for i in xrange(len(dicts)):
		epoch_dict = dicts['epoch_'+str(i)]
		for j in xrange(len(epoch_dict)):
			master_list[j].append(epoch_dict['layer_'+str(j)]['weights'])
	return master_list

#ml = convert_weight_dict_into_list(weightlogs)
#print type(ml)
#print len(ml)
#print type(ml[1])
#print len(ml[0])

def plot_weight_matrix(weights, cmap='gray',title='Weights'):
	print weights.shape
	if len(weights.shape) == 1:
		issquare = is_square_number(len(weights))
		if issquare == True:
			sqrt = np.sqrt(len(weights))
			weights = np.reshape(weights, (sqrt, sqrt))
		else:
			weights = np.reshape(weights, (len(weights), 1))
	if len(weights.shape) >2 or len(weights.shape) <=0:
		raise ValueError('Weight matrix should be two dimensional. Cannot represent higher dimensional weight matrices as an image')
	fig = plt.figure()
	plt.imshow(weights, cmap=cmap)
	if title is not None:
		plt.title(title)
	fig.tight_layout()
	plt.show()
	return fig

#plot_weight_matrix(ml[1][0])

def animate_weight_matrix(weightlist,save_name, weight_shape=None):

	if not isinstance(save_name, str):
		raise TypeError('Save name must be a string')
	if save_name.split('.')[-1] != 'mp4':
		save_name = save_name + '.mp4' # add the file encoding type on!
	fig = plt.figure()
	plt.xticks([])
	plt.yticks([])
	w0 = weightlist[0]
	if weight_shape is not None:
		w0 = np.reshape(w0, weight_shape)
	im = plt.imshow(w0, animated=True,cmap='gray')
	#plt.show(im)
	#print im.shape

	def updateFig(i):
		wi = weightlist[i]
		if weight_shape is not None:
			wi = np.reshape(wi, weight_shape)
		im.set_array(wi)
		title = plt.title('Epoch: ' + str(i))
		#plt.show(im)
		#print im.shape 

		return im,

	plt.subplots_adjust(wspace=0, hspace=0)
	anim = animation.FuncAnimation(fig, updateFig, interval=30, blit=True, save_count=99)
	anim.save(save_name,writer="ffmpeg", fps=30, extra_args=['-vcodec', 'libx264'])

"""
logs = load('logs/mnist_average_small')
print type(logs)
print len(logs)
l = logs[0]
print type(l)
print len(l)
a = l[0]
print type(a)
print len(a)
b = a[0]
print type(b)
print len(b)
print b.shape
weightlogs = load('logs/weight_logger_test_2')
print len(weightlogs)
print type(weightlogs)
print weightlogs.keys()
weights = convert_weight_dict_into_list(weightlogs)
print len(weights)
print type(weights)
print type(weights[0])
print len(weights[0])
print type(weights[0][0])
print len(weights[0][0])
#plot_weight_matrix(weights[1][50])
layer_1_weights = weights[0]
print np.max(layer_1_weights)
print len(layer_1_weights)
#print layer_1_weights
print layer_1_weights[99]
"""


#print alogs.keys()
#e1 = alogs['epoch_1']
#print e1.keys()
#l1 = e1['layer_1']
#print l1.keys()
#a = l1['activations']
#print a.shape

#for key in alogs.keys():
#	print key
def convert_attribute_dict_into_list(dicts, attr):
	master_list = []
	for i in xrange(len(dicts['epoch_0'])):
		#add the other lists
		master_list.append([])
	for i in xrange(len(dicts)):
		epoch_dict = dicts['epoch_'+str(i)]
		for j in xrange(len(epoch_dict)):
			master_list[j].append(epoch_dict['layer_'+str(j)][attr])
	return master_list


"""
alogs = load('logs/activations_logger_test_2_linear')
test = convert_attribute_dict_into_list(alogs, 'activations')
# don't have a clue why the layers are in the wrong order?
print len(test)
print len(test[0])
print test[0][0]
print test[1][0]
print test[2][0]
"""
"""
wlogs = load('logs/weights_logger_test_2_linear')
ps = convert_attribute_dict_into_list(wlogs, 'weights')
print np.max(ps[0][0])
print np.max(ps[1][0])
print np.max(ps[2][0])
print ps[0][1].shape
print ps[1][1].shape
print ps[2][1].shape
"""
"""
wlogs = load('logs/prediction_logger_test_2')
ps = convert_attribute_dict_into_list(wlogs, 'predictions')
print np.max(ps[0][99])
print np.max(ps[1][99])
print np.max(ps[2][99])
#print ps[0][99].shape
#print ps[1][99].shape
#print ps[2][99].shape
pelogs = load('logs/prediction_errors_logger_test_2')
ps = convert_attribute_dict_into_list(pelogs, 'prediction_errors')
print np.max(ps[0][99])
print np.max(ps[1][99])
print np.max(ps[2][99])
#print ps[0][99].shape
#print ps[1][99].shape
#print ps[2][99].shape
"""
"""
llogs = load('logs/loss_logger_test_2_linear')
print type(llogs)
#print llogs.keys()
loglist = convert_attribute_dict_into_list(llogs, 'loss')
print type(loglist)
print len(loglist)
print len(loglist[0])
print loglist[0][0]
"""

def plot_losses(logs):
	if isinstance(logs, str):
		# try to load
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		logs = convert_attribute_dict_into_list(logs, 'loss')
	if isinstance(logs, list):
		fig = plt.figure()
		plt.title('losses over time')
		x = np.linspace(0, len(logs[0]), num=len(logs[0]))
		for i in xrange(len(logs)):
			plt.plot(x, np.reshape(logs[i], (len(logs[i]))), label='Layer ' + str(i))
		plt.legend()
		fig.tight_layout()
		plt.show()
	else:
		raise TypeError('Input type not recognised. Needs to be a filename string, a log dictionary, or a log list. You inputted: ' + str(type(logs)))

#plot_losses(loglist)
def get_log_norm(logs, attr):
	if isinstance(logs, str):
		# try to load
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		logs = convert_attribute_dict_into_list(logs, attr)
	if isinstance(logs, list):
		master_list = []
		for l in xrange(len(logs)):
			master_list.append([])
			for e in xrange(len(logs[l])):
				master_list[l].append(np.mean(logs[l][e]))
	return master_list

def plot_weights_norm(logs):
	norm_list  = get_log_norm(logs, 'weights')
	fig = plt.figure()
	plt.title('losses over time')
	x = np.linspace(0, len(norm_list[0]), num=len(norm_list[0]))
	for i in xrange(len(norm_list)):
		plt.plot(np.reshape(norm_list[i], (len(norm_list[i]))), label='Layer ' + str(i))
	plt.legend()
	fig.tight_layout()
	plt.show()


def plot_activations_bar(logs, layer, epoch):

	if isinstance(logs, str):
		# try to load
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		logs = convert_attribute_dict_into_list(logs, 'activations')
	if isinstance(logs, list):
		activations = np.array(logs[layer][epoch])
		print activations
		activations = np.reshape(activations, (len(activations)))
		x = range(len(activations))
		fig = plt.figure()
		ax = plt.subplot(111)
		ax.bar(x, activations, width=1, color='b')
		plt.show()

def plot_norm(logs, attr=None, normalise = True):
	if isinstance(logs, str):
		# try to load
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		if attr is not None:
			logs = convert_attribute_dict_into_list(logs, 'loss')
		if attr is None:
			logs = convert_attribute_dict_into_list(logs, logs.keys()[0]) # just do the first key arbitrarily!
	if isinstance(logs, list):
		fig = plt.figure()
		plt.title('losses over time')
		x = np.linspace(0, len(logs[0]), num=len(logs[0]))
		for i in xrange(len(logs)):
			if normalise:
				logs[i] = logs[i] / np.sum(logs[i])
			plt.plot(x, np.reshape(logs[i], (len(logs[i]))), label='Layer ' + str(i))
		plt.legend()
		fig.tight_layout()
		plt.show()
	else:

		raise TypeError('Input type not recognised. Needs to be a filename string, a log dictionary, or a log list. You inputted: ' + str(type(logs)))

def get_norm_list(l):
	v = []
	for elem in l:
		v.append(np.mean(elem))
	return np.array(v)

def plot_layerwise_attribute(logs, attr, normalise=False):
	if isinstance(logs, str):
		# try to load
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		if attr is not None:
			logs = convert_attribute_dict_into_list(logs, attr)

	if isinstance(logs, list):

		x = np.linspace(0, len(logs[0]), num=len(logs[0]))
		for i in xrange(len(logs)):
	
			if not isinstance(logs[i], float) and not isinstance(logs[i], int):
				logs[i] = get_norm_list(logs[i])
			if normalise:
				logs[i] = logs[i] / np.sum(logs[i])
			fig = plt.figure()
			plt.title(str(attr) + ' for layer ' +str(i) + ' plotted across epochs')
			plt.plot(x, np.reshape(logs[i], (len(logs[i]))))
			plt.xlabel('Epochs')
			plt.ylabel(str(attr))
			plt.legend()
			fig.tight_layout()
			plt.show()
	else:

		raise TypeError('Input type not recognised. Needs to be a filename string, a log dictionary, or a log list. You inputted: ' + str(type(logs)))



def plot_attribute_func(logs,attr, func, func_name='', normalize = False):
	if isinstance(logs, str):
		logs = load(logs)
	if hasattr(logs, 'iteritems'):
		# i.e. a dictionary type
		if attr is not None:
			logs = convert_attribute_dict_into_list(logs, attr)
		if attr is None:
			logs = convert_attribute_dict_into_list(logs, logs[logs.keys()[0]].keys()[0]) # just do the first key arbitrarily!
	if isinstance(logs, list):
		x = np.linspace(0, len(logs[0]), num=len(logs[0]))
		for i in xrange(len(logs)):
			fig = plt.figure()
			plt.title(str(attr) + ' across epochs.')
			l = []
			for k in xrange(len(logs[i])):
				l.append(logs[i][k])
	
			for j in xrange(len(l)):
				print l[j].shape
				print np.var(l[j])
				l[j] = func(l[j])
			if normalize:
				l = np.array(l)
				print l.shape
				print l[0:10]
				print np.sum(l)
				l = l * 1. / np.sum(l)
			print l[0:10]
			plt.plot(x, np.reshape(l, (len(l))))
			plt.xlabel('Epoch')
			plt.ylabel(str(attr) +  ' ' + str(func_name))
			plt.legend()
			fig.tight_layout()
			plt.show()
	else:

		raise TypeError('Input type not recognised. Needs to be a filename string, a log dictionary, or a log list. You inputted: ' + str(type(logs)))

def plot_weight_mean(logs, normalize=False):
	return plot_attribute_func(logs, 'weights',np.mean, 'mean', normalize)

def plot_weight_std(logs, normalize=False):
	return plot_attribute_func(logs, 'weights', np.std, 'standard deviation', normalize)

def plot_weight_variance(logs, normalize = False):
	return plot_attribute_func(logs, 'weights', np.var, 'variance', normalize)

def plot_weight_range(logs, normalize=False):

	def r(m):
		return np.max(m) - np.min(m)

	return plot_attribute_func(logs, 'weights', r, 'range', normalize)

def plot_activations_mean(logs, normalize=False):
	return plot_attribute_func(logs, 'activations', np.mean, 'mean', normalize)

def plot_activations_std(logs, normalize=False):
	return plot_attribute_func(logs, 'activations', np.std, 'standard deviation', normalize)

def plot_predictions_mean(logs, normalize=False):
	return plot_attribute_func(logs, 'predictions', np.mean, 'mean', normalize)

def plot_predictions_std(logs, normalize=False):
	return plot_attribute_func(logs, 'predictions', np.std, 'standard deviation', normalize)

def plot_predictions_variance(logs, normalize = False):
	return plot_attribute_func(logs, 'predictions', np.var, 'variance', normalize)

def plot_predictions_range(logs, normalize=False):

	def r(m):
		return np.max(m) - np.min(m)

	return plot_attribute_func(logs, 'predictions', r, 'range', normalize)

def plot_prediction_errors_mean(logs, normalize=False):
	return plot_attribute_func(logs, 'prediction_errors', np.mean, 'mean', normalize)

def plot_prediction_errors_std(logs, normalize=False):
	return plot_attribute_func(logs, 'prediction_errors', np.std, 'standard deviation', normalize)

def plot_prediction_errors_variance(logs, normalize = False):
	return plot_attribute_func(logs, 'prediction_errros', np.var, 'variance', normalize)

def plot_prediction_errors_range(logs, normalize=False):

	def r(m):
		return np.max(m) - np.min(m)

	return plot_attribute_func(logs, 'prediction_errors', r, 'range', normalize)

def plot_loss_mean(logs, normalize=False):
	return plot_attribute_func(logs, 'loss',np.mean, 'mean', normalize)

def plot_loss_std(logs, normalize=False):
	return plot_attribute_func(logs, 'loss', np.std, 'standard deviation', normalize)

def plot_loss_variance(logs, normalize = False):
	return plot_attribute_func(logs, 'loss', np.var, 'variance', normalize)

def plot_loss_range(logs, normalize=False):

	def r(m):
		return np.max(m) - np.min(m)

	return plot_attribute_func(logs, 'loss', r, 'range', normalize)

def plot_attribute_histogram_epochs(wlist, num_bins=20):
	for i in xrange(len(wlist)):
		# assuming it's a list per layer
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		plt.title('')
		for i in xrange(len(wlist)):
			hist, bin_edges = np.histogram(wlist[i], num_bins)
			#print type(hist)
			#print type(bin_edges)
			#print len(hist)
			#print len(bin_edges)
			bs = bin_edges[0:len(bin_edges)-1]
			ax.bar(bs, hist, zs=i,zdir='x', width=0.4,alpha=0.8, align = 'center')
	
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.show()

wlogs = load('logs/sequential_RB_weights_logger')
plot_weight_mean(wlogs)
plot_weight_std(wlogs)
alogs = load('logs/sequential_RB_activations_logger')
plot_activations_mean(alogs)
plot_activations_std(alogs)
pelogs = load('logs/sequential_RB_pe_logger')
plot_prediction_errors_mean(pelogs)
plot_prediction_errors_std(pelogs)
llogs = load('logs/sequential_RB_loss_logger')
plot_loss_mean(llogs)
plot_loss_std(llogs)

