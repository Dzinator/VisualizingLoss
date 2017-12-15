import keras
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import regularizers
from pylab import *
import gc

#data required for plotting the graphs
small_batch_weights_by_graph_number ={1: [], 2: [], 3: [], 4:[]}
big_batch_weights_by_graph_number = {1: [], 2: [], 3: [], 4:[]}

#hyper parameters
number_bins = 300
small_batch_size = 128
big_batch_size = 512

#iterate over all configurations
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:

		#list of all weights for a current model
		weights_histogram_small = []
		weights_histogram_big = []

		#load the small and big batch models
		small_batch_model = load_model('model_batch_size_' + str(small_batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')
		big_batch_model = load_model('model_batch_size_' + str(big_batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

		#calculate and store the names of all layers in the VGG network
		layer_names = []
		for layer in small_batch_model.layers:
			layer_names.append(layer.name)
		
		#iterate over all the layers of the VGG network
		for layer_name in layer_names:

			#get the weights for the given layer of both the small and big batch model
			weight_matrix_array_small = small_batch_model.get_layer(layer_name).get_weights()
			weight_matrix_array_big = big_batch_model.get_layer(layer_name).get_weights()

			#check to skip the max pool layer as it does not have any weights
			if len(weight_matrix_array_small) == 2:

				#iterate over all weight matrices stored within a layer
				for weight_matrix_small in weight_matrix_array_small:
					flattened_weights = weight_matrix_small.flatten()
					weights_histogram_small.append(flattened_weights)
		
			#check to skip the max pool layer as it does not have any weights
			if len(weight_matrix_array_big) == 2:

				#iterate over all weight matrices stored within a layer
				for weight_matrix_big in weight_matrix_array_big:
					flattened_weights = weight_matrix_big.flatten()
					weights_histogram_big.append(flattened_weights)

		#concatenate all individual numpy arrays into a single array
		weights_histogram_small = np.concatenate(weights_histogram_small, axis=0 )
		weights_histogram_big = np.concatenate(weights_histogram_big, axis=0 )

		#debug purposes
		# print('sizes ' + str(len(weights_histogram_small)))
		# print('mins is ' + str(min(weights_histogram_small))) # -.059
		# print('maxs is ' + str(max(weights_histogram_small))) # 2.71

		# print('sizeb ' + str(len(weights_histogram_big)))
		# print('minb is ' + str(min(weights_histogram_big))) # -.059
		# print('maxb is ' + str(max(weights_histogram_big))) # 2.71
		
		#update the global variable structure
		small_batch_weights_by_graph_number[graph_counter] = weights_histogram_small
		big_batch_weights_by_graph_number[graph_counter] = weights_histogram_big

		#cdestroy models and call garbage collection to avoid out of memory issues
		K.clear_session()
		del small_batch_model
		del big_batch_model
		gc.collect()
		gc.collect()
		gc.collect()
		gc.collect()
		gc.collect()

		graph_counter += 1

		break
	break

#save the data 
with open('figure3_data.data', 'wb') as f:
	#create a dictionary containign everything
	everything = {'small': small_batch_weights_by_graph_number, 'large': big_batch_weights_by_graph_number}
	pickle.dump(everything, f)

#Plot the histogram of the weights. This function uses the parameters to achieve this
def plotHistogram(graph_number, weight_range, optimizer, weight_decay):
	weights_histogram_small = small_batch_weights_by_graph_number[graph_number]
	weights_histogram_big = big_batch_weights_by_graph_number[graph_number]

	#plot a histogram of the weights
	fig, plt = subplots()
	plt.hist(weights_histogram_small, bins=number_bins, range=weight_range, facecolor='blue', alpha=0.5, label=str(small_batch_size))
	plt.hist(weights_histogram_big, bins=number_bins, range=weight_range, facecolor='orange', alpha=0.5, label=str(big_batch_size))

	#set axis labels and titles
	plt.set_xlabel('Weight Bins')
	plt.set_ylabel('Number of Weights')
	plt.set_title('Histogram of Weights for Configuration' + str(optimizer) + ', WD = ' + str(weight_decay))

	#obtain min and max values of tuple
	mi, mx = weight_range

	#set axis limits
	plt.set_xlim([mi, mx])
	plt.set_ylim([0,50000])
	plt.legend(loc='upper right')
	
#plot all 4 graphs
clf()
plotHistogram(1, (-0.2, 0.2), 'SGD', 0)
plotHistogram(2, (-0.2, 0.2), 'Adam', 0)
plotHistogram(3, (-0.01, 0.01), 'SGD', 0.0005)
plotHistogram(4, (-0.04, 0.04), 'Adam', 0.0005)

#present all graphs
show()