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
from keras.callbacks import History 
from sklearn import decomposition
from sklearn import datasets
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
alphas = np.arange(-1.0, 1.0, 0.025) 



#intialize model
def createModel(batch_size, optimizer, weight_decay):
	#define model architecture
	model = Sequential()

	#padding = "same" (=padding), padding = "valid" (= no padding)
	#layer 1,2
	model.add(Convolution2D(64, (3, 3), input_shape=(3,32,32), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c1'))
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(64, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c2'))
	model.add(BatchNormalization(axis=1))
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

	#layer 3,4
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(128, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c3'))
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(128, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c4'))
	model.add(BatchNormalization(axis=1))
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

	#layer 4,5
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(256, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c5'))
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(256, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c6'))
	model.add(BatchNormalization(axis=1))
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

	#layer 6
	model.add(BatchNormalization(axis=1))
	model.add(Convolution2D(512, (3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='c7'))
	model.add(BatchNormalization(axis=1))
	model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

	#layer 7,8,9
	model.add(BatchNormalization(axis=1))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='f1'))
	model.add(BatchNormalization())
	model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='f2'))
	model.add(BatchNormalization())
	model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay), name='f3'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax', name='o'))

	#create required optimizer
	if optimizer == 'sgd':
		opt = SGD(lr=0.1, decay=0, nesterov=True)
	elif optimizer == 'adam':
		opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0)

	#compile model
	model.compile(loss='categorical_crossentropy',
				  optimizer=opt,
	              metrics=['acc'])
	return model


####################33


#weight_matrix_list_by_layer_name_by_graph_number = {}
#training curve in the contour plot
train_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[], 5:[], 6:[], 7:[], 8:[]}

#iterate over all models, 8 in total (2*2*2)
graph_counter = 1
for weight_decay in [0, 0.0005]:
	for optimizer in ['sgd', 'adam']:
		for batch_size in [128,512]:

			#initialize matrix on which to compute PCA 
			#weight_matrix_list_by_layer_name_by_graph_number[graph_counter] = dict()
			weight_matrix_list_by_layer_name = dict()

			#load the model
			oracle_model = load_model('model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

			#calculate and store the names of all layers in the VGG network
			layer_names = []
			for layer in oracle_model.layers:
				layer_weight_list = layer.get_weights()
				if len(layer_weight_list) != 0:
					layer_names.append(layer.name)
					for weights in layer_weight_list:
						flatten_weights_length = 1
						for size in weights.shape:
							flatten_weights_length *= size
						weight_matrix_list_by_layer_name[layer.name] = []
						if optimizer == 'sgd':
							weight_matrix_list_by_layer_name[layer.name].append(np.zeros((int(flatten_weights_length), 52)))
						elif optimizer == 'adam':
							weight_matrix_list_by_layer_name[layer.name].append(np.zeros((int(flatten_weights_length), 143)))

			#create a training model
			training_model = createModel(batch_size, optimizer, weight_decay)

			#Due to lack of time, sgd takes fewer epochs before converging, so we shorten its training time
			learning_rates = [0.1, 0.01, 0.001, 0.0001]
			sgd_epochs = [28, 14, 7, 3]
			adam_epochs = [76, 38, 19, 10]
			
			if optimizer == 'sgd':
				epochs = sgd_epochs
			elif optimizer == 'adam':
				epochs = adam_epochs

			current_epoch = 0

			#fit the model
			for num_epochs, learning_rate in zip(epochs, learning_rates):
				#set the learning rate
				training_model.optimizer.lr.assign(learning_rate)	

				#iterate over each individual epoch
				for epoch in range(0,num_epochs):
					#fit the model for 1 epoch
					hist = training_model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

					#get the training loss of the current iteration
					training_loss = hist.history['loss'][0]

					#append it to the losses for the current training
					train_loss_by_graph_number[graph_counter].append(training_loss)

					#iterate over all layers with weights
					for layer_name in layer_names:

						#get weights for both the final and current epoch models
						oracle_weights_list = oracle_model.get_layer(layer_name).get_weights()
						training_weights_list = training_model.get_layer(layer_name).get_weights()

						#iterate over both weight matrices
						for oracle_weights, training_weights, pca_weight in zip(oracle_weights_list, training_weights_list, weight_matrix_list_by_layer_name[layer.name]):
							vector = training_weights - oracle_weights
							pca_weight[:, current_epoch] = vector.flatten() 

					current_epoch += 1

			#update graph counter
			graph_counter += 1

			#clear the model from memory
			K.clear_session()
			del training_model
			del oracle_model
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()
			gc.collect()

			#store the results 
			with open('figure_8_model_batch_size_' + str(batch_size) + '_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '_training.data', 'wb') as f:
				#create a dictionary containign everything
				pickle.dump(weight_matrix_list_by_layer_name, f)

			with open('figure8_training_losses.data', 'wb') as f:
				#create a dictionary containign everything
				pickle.dump(train_loss_by_graph_number, f)




