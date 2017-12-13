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
from numpy import linalg as LA
import gc
K.set_image_dim_ordering('th')

#Problem
#Weight vectors for batch normalization probably don't have to have this filter nomalization...

#seed for reproducibility
np.random.seed(123)

#Data to test the model on
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#setup the y instances using 1 hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# =========================== 
#x axis values
alphas = np.arange(-1.0, 1.0, 0.025) 

#y axis values
gaussian_vectors_by_graph_number = dict()

#iterate over all models, 8 in total (2*2*2)
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:
		for batch_size in [128,512]:

			#load the model
			model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#calculate and store the names of all layers in the VGG network
			layer_names = []
			for layer in model.layers:
				layer_names.append(layer.name)

			gaussian_vectors_by_layer_name = dict()

			#iterate over all the layers of the VGG network
			for layer_name in layer_names:

				#get the weights for the given layer of the model
				weight_matrix_array_model = model.get_layer(layer_name).get_weights()
				
				#check to skip the max pool layer as it does not have any weights
				if len(weight_matrix_array_model) != 0:

					#iterate over all weight matrices stored within a layer
					gaussian_vectors = []
					for weight_matrix_model in weight_matrix_array_model:
						
						#create a new gaussian vector using the same dimensions as the current weight vector
						gaussian_weights = np.random.normal(size=weight_matrix_model.shape)

						#print('original weights' +  str(weight_matrix_model.shape))
						#print('guassian weights' + str(gaussian_weights.shape))

						#calculate the frobenius norm of the weight filter
						frobenius_norm_filter = LA.norm(weight_matrix_model)
						frobenius_norm_vector = LA.norm(gaussian_weights)

						#normalize the randomized gaussian weight vector (i.e di/||df||*||theta_i||)
						gaussian_weights = (frobenius_norm_filter / frobenius_norm_vector) * gaussian_weights
					
						#add weight matrix to current layer's list of weight matrices
						gaussian_vectors.append(gaussian_weights)

					#store the gaussian vector direction for the current layer
					gaussian_vectors_by_layer_name[layer_name] = gaussian_vectors

					#print('test for sizes of weight arrays')
					#print('gaussian sizes ' + str(len(gaussian_vectors)))
					#print('normal sizes ' + str(len(weight_matrix_array_model)))

			#store the gaussian vectors for the current model
			gaussian_vectors_by_graph_number[graph_counter] = gaussian_vectors_by_layer_name

			#update model number for next iteration
			graph_counter += 1

			#clear the model from memory
			K.clear_session()
			del model
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()

			#Delete these break tokens afterwards
			break
		break
	break

#pickle the data gathered so far for the gaussian vectors generate so far
with open('figure4_gaussian_vectors.data', 'wb') as f:
	#create a dictionary containign everything
	pickle.dump(gaussian_vectors_by_graph_number, f)


#y axis values
train_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 
test_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 
train_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 
test_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 

#iterate over all models, 8 in total (2*2*2) to generate data for the 8 graphs
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:
		for batch_size in [128,512]:
			#load the original, non-altered model
			model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#use the same architecture for the new model using the gaussian vectors
			alpha_model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#get the corresponding gaussian vector associated to the model
			gaussian_vectors_by_layer_name = gaussian_vectors_by_graph_number[graph_counter]

			#calculate and store the names of all layers in the VGG network
			layer_names = []
			for layer in model.layers:
				layer_names.append(layer.name)

			#iterate over all alpha values
			for alpha in alphas:

				#iterate over all the layers of the VGG network
				for layer_name in layer_names:

					#get the weights for the given layer of the model
					weight_matrix_array_model = model.get_layer(layer_name).get_weights()
					
					#check to skip the max pool layer as it does not have any weights
					if len(weight_matrix_array_model) != 0:
						
						#get the vector for the current layer
						gaussian_vectors = gaussian_vectors_by_layer_name[layer_name]

						#print(len(gaussian_vectors))
						#print(len(weight_matrix_array_model))

						#iterate over the different weight matrices in a given layer
						new_weights_array = []
						for weight_matrix, gaussian_vector in zip(weight_matrix_array_model, gaussian_vectors):
							#print(weight_matrix.shape)
							#print(gaussian_vector.shape)

							#calculate the new weights for the model and store it in the list
							new_weights = weight_matrix + alpha * gaussian_vector
							new_weights_array.append(new_weights)
					
						#set new weight matrix to alpha model
						alpha_model.get_layer(layer_name).set_weights(new_weights_array)

				#evalute accuracy and loss on testing set
				scores = alpha_model.evaluate(X_test, Y_test, batch_size=256, verbose=1)			

				#extract loss and accuracy
				loss = scores[0]
				accuracy = scores[1] * 100

				#store testing metrics
				test_loss_by_graph_number[graph_counter].append(loss)
				test_accuracy_by_graph_number[graph_counter].append(accuracy)

				#evalute accuracy on training set X_train.shape[0]
				scores = alpha_model.evaluate(X_train, Y_train, batch_size=256, verbose=1)	

				#extract loss and accuracy
				loss = scores[0]
				accuracy = scores[1] * 100	

				#store training metrics
				train_loss_by_graph_number[graph_counter].append(loss)
				train_accuracy_by_graph_number[graph_counter].append(accuracy)

			#cdestroy models and call garbage collection to avoid out of memory issues
			K.clear_session()
			del model
			del alpha_model
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()

			#update the graph number for the figure to be generated
			graph_counter += 1

# store data
with open('figure4_data.data', 'wb') as f:
	#create a dictionary containign everything
	everything = {'training_accuracy': train_accuracy_by_graph_number, 'testing_accuracy': test_accuracy_by_graph_number, 'training_loss': train_loss_by_graph_number, 'testing_loss': test_loss_by_graph_number, 'alphas': alphas}
	pickle.dump(everything, f)

#plot the data
clf()
#create the 4 graphs
for graph_number in [1]: #[1,2,3,4,5,6,7,8]:

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
	ax1.set_xlim([-1.0, 1.0])
	ax1.set_ylim([0,20])
	ax2.set_ylim([0,100])

	title('Loss and Accuracy as a Function of Alpha for Gaussian Vector' + str(graph_number))

#show the plots
show()