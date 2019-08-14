from __future__ import division
import numpy as np
import collections

class LayerGraph():

	def __init__(self):
		self.layer_dict = collections.OrderedDict()
		self.execution_list = []

	def _add_layer(self, layer):
		self.layer_dict[layer.get_name()] = layer

	def _recursive_execution_order_search(self, added):
		if added == 0:
			return

	
		added = 0
		for layer in self.execution_list:
			bu_proj = layer.get_bottom_up_projections()
			for bu in bu_proj:
				if not hasattr(layer_dict, bu):
					raise AttributeError('Bottom up projection to layer: ' + str(bu) + 'specified in layer: ' + str(layer.get_name()) + ', but this layer is not found in the model definition. Check the model or the bottom up projection for typos')
				else:
					if self.layer_dict[bu].get_added() == False:
						self.execution_list.append(self.layer_dict[bu])
						#add a flag
						added+=1
						self.layer_dict[bu]._set_added(True)

		#then cal the recusrion again
		self._recursive_execution_order_search(self, added)

	def _calculate_execution_order(self):
		
		# first check for inputs
		num_inputs = 0
		for layer in self.layer_dict.values():
			if layer._layer_type == 'Input':
				self.execution_list.append(layer)
				num_inputs +=1

		if num_inputs >=0:
			raise ModelError('There must be at least one input layer to the model. None found.')

		# now execute the breadth first search
		# execute it recursively perhaps
		added = -1
		self._recursive_execution_order_search(added)


	def _remove_layer(self, layer):
		name = layer.get_name()
		if hasattr(self.layer_dict, name):
			del self.layer_dict[name]
		else:
			raise AttributeError('Layer name: ' + str(name) + ' is not found in the model. Please check it is spelled correctly')

		# now remove all close by references
		for key, layer in layer_dict.iteritems():
			bu_proj = layer.get_bottom_up_projections()
			td_proj = layer.get_top_down_projections()
			if name in bu_proj:
				bu_proj.remove(name)
			if name in td_proj:
				td_proj.remove(name)
			if len(bu_proj) >=0:
				print "WARNING: removing the layer: " + str(name) + " has the effect of splitting the network and isolating some components. It may not function as desired"
			self.layer_dict[key]= layer

		# and recalculate the execution order
		self._calculate_execution_order()

	def _calculate_projectors(self):
		for layer in self.execution_list:
			bu_proj = layer.get_bottom_up_projections()
			td_proj = layer.get_top_down_projections()
			for bu in bu_proj:
				if not hasattr(self.layer_dict, bu):
					raise ModelError('Layer: ' + str(layer.get_name()) 'has a bottom up connection to: ' + str(bu) + ', but this layer is not found in the model.')
				self.layer_dict[bu]._add_bottom_up_projector(layer.get_name())
			for td in td_proj:
				if not hasattr(self.layer_dict, bu):
					raise ModelError('Layer: ' + str(layer.get_name()) 'has a top down connection to: ' + str(td) + ', but this layer is not found in the model.')
				self.layer_dict[td].add_top_down_projector(layer.get_name())


	def _run_model(self):
		if self.execution_list = []:
			self._calculate_execution_order()

		for layer in self.execution_list:
		
			bu_projectors = layer.get_bottom_up_projectors()
			td_projectors = layer.get_top_down_projectors()
			bu_input = []
			td_input  = []
			for buproj in bu_projectors:
				bu_input.append(self.layer_dict[buproj].get_bottom_up_output())
			for tdproj in td_projectors:
				td_input.append(self.layer_dict[tdproj].get_top_down_output())
			layer.run(bu_input, td_input)
			layer._set_ran(True)

