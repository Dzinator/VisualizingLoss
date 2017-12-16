import keras
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10
#from matplotlib import pyplot as plt
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import regularizers
from pylab import *
from numpy import linalg as LA
import gc
#import matplotlib.pyplot as plt
from keras.callbacks import History 
from sklearn import decomposition
from sklearn import datasets
import scipy.optimize as optimize
K.set_image_dim_ordering('th')

#seed for reproducibility
np.random.seed(123)


#Problem, need to get the % variance axis for PCA...

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
alphas = np.arange(-20, 20, 1) 

##### ==============================================
pca_vector_1_by_layer_name_by_graph_number = {1: {}, 2: {}, 3: {}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}} 
pca_vector_2_by_layer_name_by_graph_number = {1: {}, 2: {}, 3: {}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}} 
test_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]} 

graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:
		for batch_size in [128,512]:

			#load the original, non-altered model
			model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#use the same architecture for the new model using the gaussian vectors
			alpha_model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			pca_vector_1_by_layer_name = dict()
			pca_vector_2_by_layer_name = dict()

			#read the vectors generated from training
			with open('figure_8_model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '_training.data', 'rb') as f:
				weight_matrix_list_by_layer_name = pickle.load(f)

			#calculate and store the names of all layers in the VGG network
			layer_names =  weight_matrix_list_by_layer_name.keys()
			for layer_name, pca_weights_list in weight_matrix_list_by_layer_name.items():
				pca_vector_1 = []
				pca_vector_2 = []

				for pca_weights, model_weights in zip(pca_weights_list, model.get_layer(layer_name).get_weights()):
					#calculate the pca of the vector
					pca = decomposition.PCA(n_components=2)
					pca.fit(pca_weights)
					pca_weights = pca.transform(pca_weights).transpose()
					
					#resize the array weights
					pca_matrix_1 = np.reshape(pca_weights[0], model_weights.shape)
					pca_matrix_2 = np.reshape(pca_weights[1], model_weights.shape)

					#calculate frobenius norm of model weights
					frobenius_norm_filter = LA.norm(model_weights)
					frobenius_norm_vector_1 = LA.norm(pca_matrix_1)
					frobenius_norm_vector_2 = LA.norm(pca_matrix_2)

					#normalize the pca weight vector (i.e di/||df||*||theta_i||)
					pca_matrix_1 = (frobenius_norm_filter / frobenius_norm_vector_1) * pca_matrix_1
					pca_matrix_2 = (frobenius_norm_filter / frobenius_norm_vector_2) * pca_matrix_2

					#add the normalized pca vectors to the list
					pca_vector_1.append(pca_matrix_1)
					pca_vector_2.append(pca_matrix_2)

				#set the list of weights for the pca vectors 1 and 2 for the given layer
				pca_vector_1_by_layer_name[layer_name] = pca_vector_1
				pca_vector_2_by_layer_name[layer_name] = pca_vector_2

			#set the list of weights for the pca vectors for the current graph
			pca_vector_1_by_layer_name_by_graph_number[graph_counter] = pca_vector_1_by_layer_name
			pca_vector_2_by_layer_name_by_graph_number[graph_counter] = pca_vector_2_by_layer_name

			#release memory from weight matrices
			#weight_matrix_list_by_layer_name = None

			#clear the model from memory
			K.clear_session()
			del model
			del alpha_model
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()

			break
		break
	break

###
with open('figure8_training_losses.data', 'rb') as f:
	#create a dictionary containign everything
	train_loss_by_graph_number = pickle.load(f)
	train_losses = train_loss_by_graph_number[1]

#get the vectors to get back each training instance
#weight_matrix_list_by_layer_name
training_loss_x = []
training_loss_y = []
training_loss_z = train_losses

#load the original, non-altered model
model = load_model('model_batch_size_128_optimizer_sgd_weight_decay_0.h5')

#Obtain the original vectors
layer_names =  weight_matrix_list_by_layer_name.keys()
for i in range(0,len(train_losses)):
	if i % 2 == 1:
		continue

	original_weight_list_by_layer_name = dict()

	for layer_name, pca_weights_list in weight_matrix_list_by_layer_name.items():

		final_weights = model.get_layer(layer_name).get_weights()
		
		original_weights = []
		for vector_weight_matrix, final_weight_array in zip(pca_weights_list[:, int(i/2)].transpose(), final_weights):
			
			#flatten the final weights
			final_weight_array = final_weight_array.flatten()

			#original weight
			original_weight = vector_weight_matrix + final_weight_matrix
			original_weights.append(original_weight)

	
		original_weight_list_by_layer_name[layer_name] = original_weights


	def weight_difference(params):
		alpha, beta = params
		total = 0
		for layer_name in layer_names:
			for orignal, pca_vector_1, pca_vector_2 in zip(original_weight_list_by_layer_name[layer_name], pca_vector_1_by_layer_name_by_graph_number[1][layer_name], pca_vector_2_by_layer_name_by_graph_number[1][layer_name]):
				difference = original - (alpha * pca_vector_1) - (beta * pca_vector_2)
				difference = np.absolute(difference)
				total += sum(difference)
		return total

	initial_guess = [0.0, 0.0]
	result = optimize.minimize(weight_difference, initial_guess)

	if result.success:
	    alpha, beta = result.x
	    training_loss_x.append(alpha)
	    training_loss_y.append(beta)
	    print(fitted_params)
	else:
	    raise ValueError(result.message)


# import scipy.optimize as optimize

# def f(params):
#     # print(params)  # <-- you'll see that params is a NumPy array
#     a, b, c = params # <-- for readability you may wish to assign names to the component variables
#     return a**2 + b**2 + c**2

# initial_guess = [1, 1, 1]
# result = optimize.minimize(f, initial_guess)
# if result.success:
#     fitted_params = result.x
#     print(fitted_params)
# else:
#     raise ValueError(result.message)
