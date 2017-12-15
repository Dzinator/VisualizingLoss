from pylab import *
import pickle
import sys  

# store data
with open('figure4_data.data', 'rb') as f:
	#create a dictionary containign everything
	everything = pickle.load(f)

	#{'training_accuracy': train_accuracy_by_graph_number, 'testing_accuracy': test_accuracy_by_graph_number, 'training_loss': train_loss_by_graph_number, 'testing_loss': test_loss_by_graph_number, 'alphas': alphas}
	train_loss = everything['training_loss']	
	test_loss = everything['testing_loss']
	alphas = everything['alphas']
	train_accuracy = everything['training_accuracy']
	test_accuracy = everything['testing_accuracy']

#plot the data
clf()
#create the 4 graphs
for graph_number in [1,2,3,4,5,6,7,8]:

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