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
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

#Problem
#Weight vectors for batch normalization probably don't have to have this filter nomalization...
#No specification for the loss required for the contour plot (i.e training or testing loss?)
#As of now, I use testing loss as it is faster to compute

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
#x and y axis values
alphas = np.arange(-1.0, 1.0, 0.1) 

gaussian_vectors_by_graph_number_1 = dict()
gaussian_vectors_by_graph_number_2 = dict()

#iterate over all models, 8 in total (2*2*2)
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['adam', 'sgd']:
		for batch_size in [512,128]:

			#load the model
			model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#calculate and store the names of all layers in the VGG network
			layer_names = []
			for layer in model.layers:
				layer_names.append(layer.name)

			gaussian_vectors_by_layer_name_1 = dict()
			gaussian_vectors_by_layer_name_2 = dict()

			#iterate over all the layers of the VGG network
			for layer_name in layer_names:

				#get the weights for the given layer of the model
				weight_matrix_array_model = model.get_layer(layer_name).get_weights()
				
				#check to skip the max pool layer as it does not have any weights
				if len(weight_matrix_array_model) != 0:

					#iterate over all weight matrices stored within a layer
					gaussian_vectors_1 = []
					gaussian_vectors_2 = []

					for weight_matrix_model in weight_matrix_array_model:
						
						#create a new gaussian vector using the same dimensions as the current weight vector
						gaussian_weights_1 = np.random.normal(size=weight_matrix_model.shape)
						gaussian_weights_2 = np.random.normal(size=weight_matrix_model.shape)

						#print('original weights' +  str(weight_matrix_model.shape))
						#print('guassian weights' + str(gaussian_weights_1))

						#calculate the frobenius norm of the weight filter
						frobenius_norm_filter = LA.norm(weight_matrix_model)
						frobenius_norm_vector_1 = LA.norm(gaussian_weights_1)
						frobenius_norm_vector_2 = LA.norm(gaussian_weights_2)

						#normalize the randomized gaussian weight vector (i.e di/||df||*||theta_i||)
						gaussian_weights_1 = (frobenius_norm_filter / frobenius_norm_vector_1) * gaussian_weights_1
						gaussian_weights_2 = (frobenius_norm_filter / frobenius_norm_vector_2) * gaussian_weights_2

						#print('after fulter ' + str(gaussian_weights_1))

						#add weight matrix to current layer's list of weight matrices
						gaussian_vectors_1.append(gaussian_weights_1)
						gaussian_vectors_2.append(gaussian_weights_2)

					#store the gaussian vector direction for the current layer
					gaussian_vectors_by_layer_name_1[layer_name] = gaussian_vectors_1
					gaussian_vectors_by_layer_name_2[layer_name] = gaussian_vectors_2

					#print('test for sizes of weight arrays')
					#print('gaussian sizes ' + str(len(gaussian_vectors)))
					#print('normal sizes ' + str(len(weight_matrix_array_model)))

			#store the gaussian vectors for the current model
			gaussian_vectors_by_graph_number_1[graph_counter] = gaussian_vectors_by_layer_name_1
			gaussian_vectors_by_graph_number_2[graph_counter] = gaussian_vectors_by_layer_name_2

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

			break
		break
	break


#pickle the data gathered so far for the gaussian vectors generate so far
with open('figure5_gaussian_vectors.data', 'wb') as f:
	#create a dictionary containign everything
	everything = {1: gaussian_vectors_by_graph_number_1, 2: gaussian_vectors_by_graph_number_2}
	pickle.dump(everything, f)

#z axis values
test_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 

#iterate over all models, 8 in total (2*2*2) to generate data for the 8 graphs
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['adam', 'adam']:
		for batch_size in [512,128]:
			#load the original, non-altered model
			model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#use the same architecture for the new model using the gaussian vectors
			alpha_model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#get the corresponding gaussian vector associated to the model
			gaussian_vectors_by_layer_name_1 = gaussian_vectors_by_graph_number_1[graph_counter]
			gaussian_vectors_by_layer_name_2 = gaussian_vectors_by_graph_number_2[graph_counter]

			#calculate and store the names of all layers in the VGG network
			layer_names = []
			for layer in model.layers:
				layer_names.append(layer.name)

			#iterate over all alpha values
			for alpha_y in alphas:
				#losses along the x axis, keeping y fixed
				losses_y = []

				for alpha_x in alphas:

					#iterate over all the layers of the VGG network
					for layer_name in layer_names:

						#get the weights for the given layer of the model
						weight_matrix_array_model = model.get_layer(layer_name).get_weights()
						
						#check to skip the max pool layer as it does not have any weights
						if len(weight_matrix_array_model) != 0:
							
							#get the vector for the current layer
							gaussian_vectors_1 = gaussian_vectors_by_layer_name_1[layer_name]
							gaussian_vectors_2 = gaussian_vectors_by_layer_name_2[layer_name]

							#print(len(gaussian_vectors))
							#print(len(weight_matrix_array_model))

							#iterate over the different weight matrices in a given layer
							new_weights_array = []
							for weight_matrix, gaussian_vector_1, gaussian_vector_2 in zip(weight_matrix_array_model, gaussian_vectors_1, gaussian_vectors_2):
								#print(weight_matrix.shape)
								#print(gaussian_vector.shape)

								#calculate the new weights for the model and store it in the list
								new_weights = weight_matrix + alpha_x * gaussian_vector_1 + alpha_y * gaussian_vector_2
								new_weights_array.append(new_weights)
						
							#set new weight matrix to alpha model
							alpha_model.get_layer(layer_name).set_weights(new_weights_array)

					#evalute accuracy and loss on testing set
					scores = alpha_model.evaluate(X_train[:10000], Y_test[:10000], batch_size=256, verbose=1)			

					#extract loss
					loss = scores[0]

					#append loss
					losses_y.append(loss)

				#store testing metrics
				test_loss_by_graph_number[graph_counter].append(losses_y)

				# store data
				with open('figure5_data.data', 'wb') as f:
					#create a dictionary containign everything
					everything = {'test_loss_by_graph_number': test_loss_by_graph_number, 'alphas': alphas}
					pickle.dump(everything, f)

			#destroy models and call garbage collection to avoid out of memory issues
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

			break
		break
	break

#plot the data
clf()
#create the 4 graphs
for graph_number in [1]: #[1,2,3,4,5,6,7,8]:

	#get array of losses for training and testing 
	test_loss = test_loss_by_graph_number[graph_number]

	alpha_x = alphas
	alpha_y = alphas

	X,Y = np.meshgrid(alpha_x, alpha_y)
	Z = test_loss

	plt.figure()
	cp = plt.contour(X, Y, Z)
	plt.clabel(cp, inline=True, 
          fontsize=10)
	plt.title('Testing Loss as a function of alpha 1 and alpha 2')
	plt.xlabel('alpha for gaussian vector 1')
	plt.ylabel('alpha for gaussian vector 2')
	plt.show()