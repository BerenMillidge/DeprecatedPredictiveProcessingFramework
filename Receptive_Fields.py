from __future__ import division
import numpy as np
from utils import *

def tile_RFs(image,tile_radius):
	#first check image type
	if type(image) is not 'numpy.ndarray' and len(image.shape) != 2:
		raise TypeError('Image must be 2x2 numpy array')
	h,w = image.shape
	RF_centers = []
	hstep = h//tile_radius
	wstep = w//tile_radius
	for i in xrange(h):
		for j in xrange(w):
			if i % hstep == 0 and w % wstep == 0:
				RF_centers.append([i,j])
	return RF_centers

def random_generate_RFs(image, N,disallowed_radius=1):
	if type(image) is not 'numpy.ndarray' and len(image.shape) != 2:
		raise TypeError('Image must be 2x2 numpy array')
	h,w = image.shape
	RF_centers = []
	num_rfs = 0
	while num_rfs < = N:
		rh = int(h*np.random.uniform(low=0, high=1))
		rw = int(w*np.random.uniform(low=0, high=1))
		valid = True
		for center in RF_centers:
			if euclidean_distance(center, [rh,rw]) <= disallowed_radius:
				valid = False

		if valid is True:
			RF_centers.append([rh, rw])
			num_rfs +=1
	return RF_centers

def get_RF_data(image, RF_center, RF_radius, twoD = False):
	if type(image) is not 'numpy.ndarray' and len(image.shape) != 2:
		raise TypeError('Image must be 2x2 numpy array')
	h,w = image.shape
	ch, cw = RF_center

	maxh = ch + RF_radius
	if maxh > h:
		maxh = h
	minh = ch - RF_radius
	if minh < 0:
		minh = 0
	maxw = cw + RF_radius
	if maxw > w:
		maxw = w
	if maxw < 0:
		maxw = 0

	img_bound = image[minh : maxh, minw : maxw]
	p,q = img_bound.shape
	
	if twoD is True:
		for i in xrange(p):
			for j in xrange(q):
				if euclidean_distance([i,j], RF_center) > RF_radius:
					img_bound[i][j] = 0
		return img_bound


	# now append the point
	vals = []
	for i in xrange(p):
		for j in xrange(q):
			if euclidean_distance([i,j], RF_center) <= RF_radius:
				vals.append(img_bound[i][j])

	return np.array(vals)

def get_RFs_data(image, RF_centers, RF_radius, twoD = False):
	results = []
	for center in RF_centers:
		results.append(get_RF_data(image, center, RF_radius, twoD))
	return results
 
def RF_data_for_dataset(dataset, RF_centers, RF_radius,twoD = False):
	data = []
	for data_item in dataset:
		data.append(get_RFs_data(data_item, RF_centers, RF_radius, twoD = twoD))
	return data

