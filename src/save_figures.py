#!/usr/bin/env python
# coding: utf-8

# Imports
# get_ipython().run_line_magic('matplotlib', 'inline')

# Import matplotlib
import matplotlib

# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import globals
import itertools
import numpy as np

# Call function saveTrainingLoss
def saveTrainingLoss(model1,
					 model2,
					 model3,
					 model4,
					 model5,
					 model6,
					 descriptor_name,
					 dataset_name):
	# Plot Training Loss

	# Turn interactive plotting off
	plt.ioff()

	# Create a new figure
	plt.figure(figsize = (16, 8))
	plt.plot(model1.loss_curve_, label = 'SGD + M = 0.0 + constant')
	plt.plot(model2.loss_curve_, label = 'SGD + M = 0.3 + constant')
	plt.plot(model3.loss_curve_, label = 'SGD + M = 0.6 + constant')
	plt.plot(model4.loss_curve_, label = 'SGD + M = 0.9 + constant')
	plt.plot(model5.loss_curve_, label = 'SGD + M = 0.9 + adaptive')
	plt.plot(model6.loss_curve_, label = 'Adam + M = 0.9 + constant')
	plt.title('Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.grid()
	plt.legend()

	# Save it
	plt.savefig(fname = 'Figures/Datasets/%s/%s-MLP-training-loss.png' % (dataset_name, descriptor_name),
	            bbox_inches = 'tight',
	            transparent = True,
	            dpi = 300)

	# Close it
	plt.close()

# Call function confusionMatrix
def confusionMatrix(predict,
                    descriptor_name,
					dataset_name,
                    model_name):
	# MLP Model

	# Define Labels and Rotation of plt.xticks
	if dataset_name == 'CIFAR-10':
		# Labels
		labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

		# Rotation
		rotation = 75
		
	elif (dataset_name == 'Extended-CK+') or (dataset_name == 'FER-2013') or (dataset_name == 'JAFFE'):
		# Labels
		labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

		# Rotation
		rotation = 75

	elif dataset_name == 'FEI':
		# Labels
		labels = ['happy', 'neutral']

		# Rotation
		rotation = 75

	elif dataset_name == 'MNIST':
		# Labels
		labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

		# Rotation
		rotation = 0

	# Turn interactive plotting off
	plt.ioff()

	# Create a new figure
	plt.figure()

	# Get the Confusion Matrix using Sklearn
	cm = confusion_matrix(y_true = globals.test_feature_vec[1],
						  y_pred = predict,
						  labels = labels)

	# Print the Confusion Matrix as text
	# print(cm)

	# Plot the Confusion Matrix as an image
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j,
				i,
				cm[i, j],
				horizontalalignment = 'center',
				color = 'white' if cm[i, j] > thresh else 'black')

	# Make various adjustments to the plot
	plt.imshow(cm, interpolation='nearest', cmap='Greys')
	plt.colorbar()
	tick_marks = np.arange(globals.num_classes)
	plt.xticks(tick_marks, labels, rotation = rotation)
	plt.yticks(tick_marks, labels)
	plt.title('Confusion Matrix')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')

	# Save it
	plt.savefig('Figures/Datasets/%s/%s-%s-confusion-matrix.png' % (dataset_name, descriptor_name, model_name),
				bbox_inches = 'tight',
				transparent = True,
				dpi = 300)

	# Close it
	plt.close()

# Call function weights
def weights(model,
            descriptor_name,
			dataset_name,
            model_name):
	# MLP Model

	# Define reshape
	if (dataset_name == 'CIFAR-10') or (dataset_name == 'MNIST'):
		reshape = 7, 7

	elif (dataset_name == 'Extended-CK+') or (dataset_name == 'FER-2013') or (dataset_name == 'JAFFE'):
		reshape = 2, 17

	elif dataset_name == 'FEI':
		reshape = 3, 3

	# Plot Weights of Multilayer Percentron

	# Turn interactive plotting off
	plt.ioff()

	# Create a new figure
	fig, axes = plt.subplots(4, 4)
	vmin, vmax = model.coefs_[0].min(), model.coefs_[0].max()
	for coef, ax in zip(model.coefs_[0].T,
						axes.ravel()):
		ax.matshow(coef.reshape(reshape),
				cmap = plt.cm.gray,
				vmin = .5 * vmin,
				vmax = .5 * vmax)
		ax.set_xticks(())
		ax.set_yticks(())

	# Save it
	plt.savefig('Figures/Datasets/%s/%s-%s-weights.png' % (dataset_name, descriptor_name, model_name),
				bbox_inches = 'tight',
				transparent = True,
				dpi = 300)

	# Close it
	plt.close()