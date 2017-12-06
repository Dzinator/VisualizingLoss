from pylab import *

import pickle

import sys  


with open('figure2_data.data', 'rb') as f:
	#create a dictionary containign everything
	everything = pickle.load(f)

	train_accuracy_by_graph_number = everything['training_accuracy']
	test_accuracy_by_graph_number = everything['testing_accuracy']

	train_loss_by_graph_number = everything['training_loss']
	test_loss_by_graph_number = everything['testing_loss']

	alphas = everything['alphas']	

	print(train_accuracy_by_graph_number[1])
	print(test_accuracy_by_graph_number[1])

	print(train_loss_by_graph_number[1])
	print(test_loss_by_graph_number[1])

	print(alphas)

clf()

#create the 4 graphs
for graph_number in [1,2,3,4]:

	#get array of losses for training and testing 
	train_loss = train_loss_by_graph_number[graph_number]
	test_loss = test_loss_by_graph_number[graph_number]

	#get array of accuracies for training and testing
	train_accuracy = train_accuracy_by_graph_number[graph_number]

	for i in range(0,len(train_accuracy)):
		train_accuracy[i] *= 100

	test_accuracy = test_accuracy_by_graph_number[graph_number]


	print(train_loss)
	print(test_loss)

	print(train_accuracy)
	print(test_accuracy)

	print(alphas)

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

# x = [1, 2, 4, 5, 6.7, 7, 8, 10 ] #alpha

# y1_1 = [40, 30, 10, 20, 53, 20, 10, 5] #loss of training
# y1_2 = [42, 32, 12, 22, 55, 22, 12, 7] # loss of testing

# y2_1 = [30, 28, 8, 19, 50, 22, 12, 6] # accuracy of training
# y2_2 = [28, 26, 6, 17, 48, 20, 10, 4] # accuracy of testing


# clf()

# fig, ax1 = subplots()

# ax2 = ax1.twinx()
# ax1.plot(x, y1_1, 'k-')
# ax1.plot(x, y1_2, 'k--')

# ax2.plot(x, y2_1, 'r-')
# ax2.plot(x, y2_2, 'r--')

# ax1.set_xlabel('Alpha')
# ax1.set_ylabel('Loss', color='k')
# ax2.set_ylabel('Accuracy', color='r')

# ax1.set_xlim([-0.50, 1.50])
# ax1.set_ylim([0,5])
# ax2.set_ylim([0,100])

# title('Loss and Accuracy as a Function of Alpha')

# fig, ax1 = subplots()

# ax2 = ax1.twinx()
# ax1.plot(x, y1_1, 'k-')
# ax1.plot(x, y1_2, 'k--')

# ax2.plot(x, y2_1, 'r-')
# ax2.plot(x, y2_2, 'r--')

# ax1.set_xlabel('Alpha')
# ax1.set_ylabel('Loss', color='k')
# ax2.set_ylabel('Accuracy', color='r')

# ax1.set_xlim([-0.50, 1.50])
# ax1.set_ylim([0,5])
# ax2.set_ylim([0,100])

# title('Loss and Accuracy as a Function of Alpha')

# show()

# figure('Figure 2 of Paper')
# plot(x,y, 'b', label='Mori')
# xlabel('alpha')
# ylabel('y')
# title('y as a function of x')

# plot(x,y2, 'r',label="Gallup",color='r',linewidth=2,markerfacecolor='g',
# markeredgecolor='y',marker='d')






#subplot(2,2,4)



#legend()
#show()	