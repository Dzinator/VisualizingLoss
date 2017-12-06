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
K.set_image_dim_ordering('th')

#seed for reproducibility
np.random.seed(123)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#setup the y instances using 1 hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#data required for plotting the graphs
#x axis values
alphas = np.arange(-0.50, 1.50, 0.025) 

#y axis values
train_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[]} 
test_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[]}
train_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[]}
test_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[]}

graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:

		#dictionary to store the vectors (big batch - small batch) for each layer of the network
		vectors_by_layer_name = dict()

		#load the small and big batch models
		small_batch_model = load_model('model_batch_size_128_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')
		big_batch_model = load_model('model_batch_size_512_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')


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
			if len(weight_matrix_array_small) != 0:

				#iterate over all weight matrices stored within a layer
				vectors = []
				for weight_matrix_small, weight_matrix_big in zip(weight_matrix_array_small, weight_matrix_array_big):
					
					#calculate difference between the big and small batch members to establish a vector for the current weight matrix for a given layer
					diff = weight_matrix_big - weight_matrix_small
					vectors.append(diff)

				#store the vector representing the (big_batch - small_batch) direction for the current layer
				vectors_by_layer_name[layer_name] = vectors
		
		
		#we will use the big batch model's architecture as a basis to calculate the results
		#copy for readability	
		new_model = big_batch_model
		
		#iterate over all alpha values
		for alpha in alphas:

			#adjust weights on all layers of the VGG given the alpha value
			for layer_name in layer_names:

				#get the original weights of the small batch model
				weight_matrix_array_small = small_batch_model.get_layer(layer_name).get_weights()

				#check to skip the max pool layer as it does not have any weights
				if len(weight_matrix_array_small) != 0: 
					
					#get the vector representing the (big_batch - small_batch) direction for the current layer
					vector = vectors_by_layer_name[layer_name]
					
					#calculate new weight matrix for member of the layer's weights array
					a_vector = []
					for component, weight_matrix_small in zip(vector, weight_matrix_array_small):
						a_component = weight_matrix_small + alpha * component
						a_vector.append(a_component)

					#set the new weights
					new_model.get_layer(layer_name).set_weights(a_vector)

			#evalute accuracy and loss on testing set
			scores = new_model.evaluate(X_test, Y_test, batch_size=256, verbose=1)			

			#extract loss and accuracy
			loss = scores[0]
			accuracy = scores[1] * 100

			#store testing metrics
			test_loss_by_graph_number[graph_counter].append(loss)
			test_accuracy_by_graph_number[graph_counter].append(accuracy)

			#evalute accuracy on training set X_train.shape[0]
			scores = new_model.evaluate(X_train, Y_train, batch_size=256, verbose=1)	

			#extract loss and accuracy
			loss = scores[0]
			accuracy = scores[1] * 100	

			#store training metrics
			train_loss_by_graph_number[graph_counter].append(loss)
			train_accuracy_by_graph_number[graph_counter].append(accuracy)

		#cdestroy models and call garbage collection to avoid out of memory issues
		K.clear_session()
		del small_batch_model
		del big_batch_model
		gc.collect()
		gc.collect()
		gc.collect()
		gc.collect()
		gc.collect()

		#update the graph number for the figure to be generated
		graph_counter += 1


############################ plotting section!
# store data
with open('figure2_data.data', 'wb') as f:
	#create a dictionary containign everything
	everything = {'training_accuracy': train_accuracy_by_graph_number, 'testing_accuracy': test_accuracy_by_graph_number, 'training_loss': train_loss_by_graph_number, 'testing_loss': test_loss_by_graph_number, 'alphas': alphas}
	pickle.dump(everything, f)

clf()

#create the 4 graphs
for graph_number in [1,2,3,4]:

	#get array of losses for training and testing 
	train_loss = train_loss_by_graph_number[graph_number] 
	test_loss = test_loss_by_graph_number[graph_number]

	#get array of accuracies for training and testing
	train_accuracy = train_accuracy_by_graph_number[graph_number]
	test_accuracy = test_accuracy_by_graph_number[graph_number]

	#create the plots
	fig, ax1 = subplots()

	#create a double y-axis in the plot
	ax2 = ax1.twinx()

	#plot losses
	ax1.plot(alphas, train_loss, 'b-')
	ax1.plot(alphas, test_loss, 'b--')

	#plot accuracies
	ax2.plot(alphas, train_accuracy, 'r-')
	ax2.plot(alphas, test_accuracy, 'r--')

	#set label names
	ax1.set_xlabel('Alpha')
	ax1.set_ylabel('Loss', color='b')
	ax2.set_ylabel('Accuracy', color='r')

	#set the axis limits on the graph
	ax1.set_xlim([-0.50, 1.50])
	ax1.set_ylim([0,20])
	ax2.set_ylim([0,100])

	title('Loss and Accuracy as a Function of Alpha ' + str(graph_number))

#show the plots
show()

#are in Keras use them!
# Xception
# VGG16
# VGG19
# ResNet50
# Inception v3
# Inception-ResNet v2
# MobileNet v1

#m.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
#scores = m.evaluate(preds, y_regression, batch_size=32, verbose=0)
#print '\nevaluate result: mse={}, mae={}, mape={}'.format(*scores)