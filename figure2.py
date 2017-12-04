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
K.set_image_dim_ordering('th')

#8 models to be generated (2*2*2)

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
alphas = np.arange(-0.50, 1.50, 0.05) #x axis
train_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[]}
test_accuracy_by_graph_number = {1: [], 2: [], 3: [], 4:[]}
train_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[]}
test_loss_by_graph_number = {1: [], 2: [], 3: [], 4:[]}

graph_counter = 1
for weight_decay in [0]: #[0, 0.0005]:
	for optimizer in ['sgd']: #['sgd', 'adam']:
		vectors_by_layer_name_1 = dict()
		vectors_by_layer_name_2 = dict()

		# small_batch_model = load_model('model_batch_size_128_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')
		# big_batch_model = load_model('model_batch_size_8192_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

		small_batch_model = load_model('model_batch_size_128_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')
		big_batch_model = load_model('model_batch_size_512_optimizer_' + str(optimizer) + '_weight_decay_' + str(weight_decay) + '.h5')

		#print(len(big_batch_model.get_layer('c1').get_weights()))
		#print(big_batch_model.get_layer('c1').get_weights()[0].shape)

		#calculate vectors of all configuartions
		for layer_name in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'f1', 'f2', 'f3']:
			vectors_by_layer_name_1[layer_name] = big_batch_model.get_layer(layer_name).get_weights()[0] - small_batch_model.get_layer(layer_name).get_weights()[0]
			vectors_by_layer_name_2[layer_name] = big_batch_model.get_layer(layer_name).get_weights()[1] - small_batch_model.get_layer(layer_name).get_weights()[1]
		
		#copy for readability	
		new_model = big_batch_model

		#we will use the big batch model to calculate the results
		for alpha in alphas:
			for layer_name in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'f1', 'f2', 'f3']:
				new_weights_1 = small_batch_model.get_layer(layer_name).get_weights()[0] + alpha * vectors_by_layer_name_1[layer_name]
				new_weights_2 = small_batch_model.get_layer(layer_name).get_weights()[1] + alpha * vectors_by_layer_name_2[layer_name]
				new_model.get_layer(layer_name).set_weights([new_weights_1, new_weights_2])


			#m.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
			#scores = m.evaluate(preds, y_regression, batch_size=32, verbose=0)
			#print '\nevaluate result: mse={}, mae={}, mape={}'.format(*scores)

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

		graph_counter += 1


############################ plotting section!
# store data
with open('figure2_data.data', 'wb') as f:
	#create a dictionary containign everything
	everything = {'training_accuracy': train_accuracy_by_graph_number, 'testing_accuracy': test_accuracy_by_graph_number, 'training_loss': train_loss_by_graph_number, 'testing_loss': test_loss_by_graph_number, 'alphas': alphas}
	pickle.dump(everything, f)

clf()

#create the 4 graphs
for graph_number in [1]: #[1,2,3,4]:

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
	ax1.plot(alphas, train_loss, 'k-')
	ax1.plot(alphas, test_loss, 'k--')

	#plot accuracies
	ax2.plot(alphas, train_accuracy, 'r-')
	ax2.plot(alphas, test_accuracy, 'r--')

	#set label names
	ax1.set_xlabel('Alpha')
	ax1.set_ylabel('Loss', color='k')
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