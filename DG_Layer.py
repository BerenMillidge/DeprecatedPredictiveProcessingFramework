import scipy 
import numpy

class DG_Layer():

	def __init__(self, data, sigma_1 = 1, sigma_2 = 3):
		self.data = data
		self.sigma_1 = sigma_1
		self.sigma_2 = sigma_2
		self.image = data
		self.filtered_image = None
		#set transformer and trainable attributes
		self.__transformer = True
		self.__trainable = False

	def run(self):
		if len(data.shape) == 2:
			# so a sigle image
			self.filtered_image = self.DOG_filter()

		if len(data.shape)==3:
			# so a list of images
			imgs = []
			for i in xrange(len(data)):
				self.image = data[i]
				imgs.append(self.DOG_filter())
			self.filtered_image = np.array(imgs)
		return self.filtered_image

	def DOG_filter(self):
		return scipy.ndimage.filters.gaussian_filter(self.image, self.sigma_1) - scipy.ndimage.filters.gaussian_filter(self.image, self.sigma_2)


	def get_sigma_1(self):
		return self.sigma_1

	def get_sigma_2(self):
		return self.sigma_2

	def get_data(self):
		return self.data

	def get_image(self):
		return self.image

	def get_filtered_image(self):
		return self.filtered_image

	def get_layer_info(self):
		info = {}
		info['data'] = self.data
		info['sigma_1'] = self.sigma_1
		info['sigma_2'] = self.sigma_2
		info['image'] = self.image
		info['filtered_image'] = self.filtered_image
		return info




