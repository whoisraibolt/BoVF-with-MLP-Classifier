#!/usr/bin/env python
# coding: utf-8

# Evaluating a Bag of Visual Features (BoVF) approach extracting
# Features for Handwritten Digit Recognition and
# Classification Task through MLP Classifier

# Imports
from timer import Timer
import argparse
import cv2 as cv
import datasets
import features
import globals
import multilayer
import os
import outputs
import parameters
import save_figures
import save_models
import sys
import time

# Initializing global variables
globals.initialize()

# Message from usage
message = '''main.py [-h]

             --detector     {SIFT, SURF, KAZE, ORB, BRISK, AKAZE}
             --descriptor   {SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK}
             --dataset      {MNIST, JAFFE, Extended-CK+, FEI, CIFAR-10, FER-2013}'''

# Create the parser
parser = argparse.ArgumentParser(description = 'Evaluating a Bag of Visual Features (BoVF) approach extracting ' +
                                               'Features for Recognition and Classification Task through MLP Classifier.',
                                usage = message)

# Argument --detector
parser.add_argument('--detector',
                    action = 'store',
                    choices = ['SIFT', 'SURF', 'KAZE', 'ORB', 'BRISK', 'AKAZE'],
                    required = True,
                    metavar = '',
                    dest = 'detector',
                    help = 'select the detector to be used in this experiment')

# Argument --descriptor
parser.add_argument('--descriptor',
                    action = 'store',
                    choices = ['SIFT', 'SURF', 'KAZE', 'BRIEF', 'ORB', 'BRISK', 'AKAZE', 'FREAK'],
                    required = True,
                    metavar = '',
                    dest = 'descriptor',
                    help = 'select the descriptor to be used in this experiment')

# Argument --dataset
parser.add_argument('--dataset',
                    action = 'store',
                    choices = ['MNIST', 'JAFFE', 'Extended-CK+', 'FEI', 'CIFAR-10', 'FER-2013'],
                    required = True,
                    metavar = '',
                    dest = 'dataset',
                    help = 'select the visual dataset to be used in this experiment')

# Execute the parse_args() method
arguments = parser.parse_args()

# File name
filename = 'Outputs/Datasets/FEI/%s-outputs.txt' % arguments.descriptor

# Open File to write
outputs.openFile(filename = filename)

# Initiate Detector and Descriptor

# Initiate detector selected
if arguments.detector == 'SIFT':
    globals.detector = features.SIFT()

elif arguments.detector == 'SURF':
    globals.detector = features.SURF()

elif arguments.detector == 'KAZE':
    globals.detector = features.SIFT()

elif arguments.detector == 'ORB':
    globals.detector = features.ORB()

elif arguments.detector == 'BRISK':
    globals.detector = features.BRISK()

elif arguments.detector == 'AKAZE':
    globals.detector = features.AKAZE()

# Print detector
features.printDetector()

# Initiate descriptor selected
if arguments.descriptor == 'SIFT':
    globals.descriptor = features.SIFT()

elif arguments.descriptor == 'SURF':
    globals.descriptor = features.SURF()

elif arguments.descriptor == 'KAZE':
    globals.descriptor = features.SIFT()

elif arguments.descriptor == 'BRIEF':
    globals.descriptor = features.BRIEF()

elif arguments.descriptor == 'ORB':
    globals.descriptor = features.ORB()

elif arguments.descriptor == 'BRISK':
    globals.descriptor = features.BRISK()

elif arguments.descriptor == 'AKAZE':
    globals.descriptor = features.AKAZE()

elif arguments.descriptor == 'FREAK':
    globals.descriptor = features.FREAK()

# Print descriptor
features.printDescriptor()

# Path of Dataset
if arguments.dataset == 'MNIST':
    globals.data_train_images, globals.data_test_images = datasets.MNIST()

elif arguments.dataset == 'JAFFE':
    globals.data_train_images, globals.data_test_images = datasets.JAFFE()

elif arguments.dataset == 'Extended-CK+':
    globals.data_train_images, globals.data_test_images = datasets.extendedCK()

elif arguments.dataset == 'FEI':
    globals.data_train_images, globals.data_test_images = datasets.FEI()

elif arguments.dataset == 'CIFAR-10':
    globals.data_train_images, globals.data_test_images = datasets.CIFAR10()

elif arguments.dataset == 'FER-2013':
    globals.data_train_images, globals.data_test_images = datasets.FER2013()

# Print training-set path
datasets.printTrainingPath()

# Print test-set path
datasets.printTestPath()

# Print number of classes
datasets.printNumberOfClasses()

# K-Means classes
globals.K = globals.num_classes * 5

# Print
print('\nExtract Features\n')

# Extract Features
all = {}

# Print
print('Extract Features\n', file = globals.file)

object = sorted(os.listdir(globals.data_train_images))    

#wall_start, cpu_start = time.time(), time.clock()

i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']', file = globals.file)

        # Read in cropped data
        crop_names = os.listdir(os.path.join(globals.data_train_images, subject))
        crop_names = list(map(lambda x: os.path.join(globals.data_train_images, subject, x), crop_names)) 
        crops = [cv.imread(x , cv.IMREAD_GRAYSCALE) for x in crop_names]

        # Get Features
        desc = features.extractFeatures(crops, features.features)
        all[subject] = desc
        print('Extracted', arguments.descriptor, '\n', file = globals.file)

print('Time:', timer, '\n', file = globals.file)

# Print
print('Done!\n')

# Print
print('Create Bag of Visual Features\n')

# Features
matrix = features.groupAllFeatures(all)

kmeans = None

# Print
print('Create Bag of Visual Features\n', file = globals.file)

# Train K-Means
print('Training', arguments.descriptor, 'K-Means\n', file = globals.file)

#wall_start, cpu_start = time.time(), time.clock()

with Timer() as timer:
    kmeans = features.trainKMeans(matrix)
    
print('Time:', timer, '\n', file = globals.file)

# Print
print('Done!\n')

# Print
print('Prepare Training Data\n')

# Training Data
globals.train_feature_vec = [[], []]

print(arguments.descriptor, 'Training Data\n', file = globals.file)
object = sorted(os.listdir(globals.data_train_images))

i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']\n', file = globals.file)

        # Get Features
        histograms = features.generateHistograms(all[subject], kmeans)
        globals.train_feature_vec[0].extend(histograms)
        globals.train_feature_vec[1].extend([subject] * len(histograms))
    
print('Time:', timer, '\n', file = globals.file)

# Print
print('Done!\n')

# Print
print('Prepare Testing Data\n')

# Testing Data
globals.test_feature_vec = [[], []]

print(arguments.descriptor, 'Testing Data\n', file = globals.file)
object = sorted(os.listdir(globals.data_test_images))

i, total = 0, len(object)

with Timer() as timer:
    for subject in object:
        i += 1
        print('Processing the subdirectory named:', subject, '\t[', i , '/', total, ']\n', file = globals.file)
      
      
        crop_names = os.listdir(os.path.join(globals.data_test_images, subject))
        crop_names = list(map(lambda x: os.path.join(globals.data_test_images, subject, x), crop_names)) 
        crops = [cv.imread(x , cv.IMREAD_GRAYSCALE) for x in crop_names ]
      
        # Get Features
        desc = features.extractFeatures(crops, features.features)
      
        # Get Histograms
        histograms = features.generateHistograms(desc, kmeans)
        globals.test_feature_vec[0].extend(histograms)
        globals.test_feature_vec[1].extend([subject] * len(histograms))
    
print('Time:', timer, '\n', file = globals.file)

# Print
print('Done!\n')

# Print
print('Training Multilayer Perceptron Model\n')

# Train MLP Model
print('Training %s MLP Models\n' % arguments.descriptor, file = globals.file)

# MLP Model 1
MLP1 = multilayer.train(solver = 'sgd',
                        momentum = 0,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP1')

# MLP Model 2
MLP2 = multilayer.train(solver = 'sgd',
                        momentum = 0.3,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP2')

# MLP Model 3
MLP3 = multilayer.train(solver = 'sgd',
                        momentum = 0.6,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP3')

# MLP Model 4
MLP4 = multilayer.train(solver = 'sgd',
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP4')

# MLP Model 5
MLP5 = multilayer.train(solver = 'sgd',
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'adaptive',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP5')

# MLP Model 6
MLP6 = multilayer.train(solver = 'adam',
                        momentum = 0.9,
                        hidden_layer_sizes = parameters.hidden_layer,
                        alpha = parameters.alpha,
                        learning_rate = 'constant',
                        learning_rate_init = parameters.learning_rate,
                        random_state = parameters.seed,
                        max_iter = parameters.max_iterations,
                        descriptor_name = arguments.descriptor,
                        model_name = 'MLP6')

# Print
print('Done!\n')

# Print
print('Saving Training Loss\n')

# Save Training Loss Figure
save_figures.saveTrainingLoss(model1 = MLP1,
                              model2 = MLP2,
                              model3 = MLP3,
                              model4 = MLP4,
                              model5 = MLP5,
                              model6 = MLP6,
                              descriptor_name = arguments.descriptor,
                              dataset_name = arguments.dataset)

# Print
print('Done!\n')

# Print
print('Testing Multilayer Perceptron Model\n')

# Test MLP Model
print('Testing %s MLP Model\n' % arguments.descriptor, file = globals.file)

# MLP Model 1
MLP1_predict = multilayer.test(model = MLP1,
                               solver = 'sgd',
                               momentum = 0,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP1')

# MLP Model 2
MLP2_predict = multilayer.test(model = MLP2,
                               solver = 'sgd',
                               momentum = 0.3,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP2')

# MLP Model 3
MLP3_predict = multilayer.test(model = MLP3,
                               solver = 'sgd',
                               momentum = 0.6,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP3')

# MLP Model 4
MLP4_predict = multilayer.test(model = MLP4,
                               solver = 'sgd',
                               momentum = 0.9,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP4')

# MLP Model 5
MLP5_predict = multilayer.test(model = MLP5,
                               solver = 'sgd',
                               momentum = 0.9,
                               learning_rate = 'adaptative',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP5')

# MLP Model 6
MLP6_predict = multilayer.test(model = MLP6,
                               solver = 'adam',
                               momentum = 0.9,
                               learning_rate = 'constant',
                               descriptor_name = arguments.descriptor,
                               model_name = 'MLP6')

# Print
print('Done!\n')

# Print
print('Classification Report\n')

# Print
print('Classification Report\n', file = globals.file)

# MLP Model 1
multilayer.classificationReport(model = MLP1,
                                predict = MLP1_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP1')

# MLP Model 2
multilayer.classificationReport(model = MLP2,
                                predict = MLP2_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP2')

# MLP Model 3
multilayer.classificationReport(model = MLP3,
                                predict = MLP3_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP3')

# MLP Model 4
multilayer.classificationReport(model = MLP4,
                                predict = MLP4_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP4')

# MLP Model 5
multilayer.classificationReport(model = MLP5,
                                predict = MLP5_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP5')

# MLP Model 6
multilayer.classificationReport(model = MLP6,
                                predict = MLP6_predict,
                                descriptor_name = arguments.descriptor,
                                model_name = 'MLP6')

# Print
print('Done!\n')

# Print
print('Saving Confusion Matrix\n')

# Save Confusion Matrix figure MLP Model 1
save_figures.confusionMatrix(predict = MLP1_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP1')

# Save Confusion Matrix figure MLP Model 2
save_figures.confusionMatrix(predict = MLP2_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP2')

# Save Confusion Matrix figure MLP Model 3
save_figures.confusionMatrix(predict = MLP3_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP3')

# Save Confusion Matrix figure MLP Model 4
save_figures.confusionMatrix(predict = MLP4_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP4')

# Save Confusion Matrix figure MLP Model 5
save_figures.confusionMatrix(predict = MLP5_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP5')

# Save Confusion Matrix figure MLP Model 6
save_figures.confusionMatrix(predict = MLP6_predict,
                             descriptor_name = arguments.descriptor,
                             dataset_name = arguments.dataset,
                             model_name = 'MLP6')

# Print
print('Done!\n')

# Print
print('Saving Weights\n')

# Save Weights figure MLP Model 1
save_figures.weights(model = MLP1,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP1')

# Save Weights figure MLP Model 2
save_figures.weights(model = MLP2,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP2')

# Save Weights figure MLP Model 3
save_figures.weights(model = MLP3,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP3')

# Save Weights figure MLP Model 4
save_figures.weights(model = MLP4,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP4')

# Save Weights figure MLP Model 5
save_figures.weights(model = MLP5,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP5')

# Save Weights figure MLP Model 6
save_figures.weights(model = MLP6,
                     descriptor_name = arguments.descriptor,
                     dataset_name = arguments.dataset,
                     model_name = 'MLP6')

# Print
print('Done!\n')

# Print
print('Saving Bag of Visual Features\n')

# Save Bag of Visual Features 
save_models.saveBoVF(kmeans = kmeans,
                     descriptor_name = arguments.descriptor,
			         dataset_name = arguments.dataset)

# Print
print('Done!\n')

# Print
print('Saving Descriptors\n')

# Save Training Descriptors
save_models.saveDescriptors(feature_vector = globals.train_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'train')

# Save Test Descriptors
save_models.saveDescriptors(feature_vector = globals.test_feature_vec,
                            descriptor_name = arguments.descriptor,
			                dataset_name = arguments.dataset,
                            flag = 'test')

# Print
print('Done!\n')

# Print
print('Saving MLP Models\n')

# MLP Model 1
save_models.saveMLP(model = MLP1,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP1')

# MLP Model 2
save_models.saveMLP(model = MLP2,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP2')

# MLP Model 3
save_models.saveMLP(model = MLP3,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP3')

# MLP Model 4
save_models.saveMLP(model = MLP4,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP4')

# MLP Model 5
save_models.saveMLP(model = MLP5,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP5')

# MLP Model 6
save_models.saveMLP(model = MLP6,
                    descriptor_name = arguments.descriptor,
			        dataset_name = arguments.dataset,
                    model_name = 'MLP6')

# Print
print('Done!\n')

# Close File
outputs.closeFile()

# Print
print('BoVF with MLP Classifier executed with success!')