#!/usr/bin/env python
# coding: utf-8

# Imports
import globals
import os

# Call function MNIST
def MNIST():
    # Path data_train_images
    data_train_images = 'Datasets/MNIST/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/MNIST/Dataset/Test/'

    return data_train_images, data_test_images

# Call function JAFFE
def JAFFE():
    # Path data_train_images
    data_train_images = 'Datasets/JAFFE/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/JAFFE/Dataset/Test/'

    return data_train_images, data_test_images

# Call function extendedCK
def extendedCK():
    # Path data_train_images
    data_train_images = 'Datasets/Extended-CK+/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/Extended-CK+/Dataset/Test/'

    return data_train_images, data_test_images

# Call function FEI
def FEI():
    # Path data_train_images
    data_train_images = 'Datasets/FEI/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/FEI/Dataset/Test/'

    return data_train_images, data_test_images

# Call function CIFAR10
def CIFAR10():
    # Path data_train_images
    data_train_images = 'Datasets/CIFAR-10/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/CIFAR-10/Dataset/Test/'

    return data_train_images, data_test_images

# Call function FER2013
def FER2013():
    # Path data_train_images
    data_train_images = 'Datasets/FER-2013/Dataset/Train/'

    # Path data_test_images
    data_test_images = 'Datasets/FER-2013/Dataset/Test/'

    return data_train_images, data_test_images

# Call function printTrainingPath
def printTrainingPath():
    print('Training-set path: %s\n' % globals.data_train_images, file = globals.file)

# Call function printTestPath
def printTestPath():
    print('Test-set path: %s\n' % globals.data_test_images, file = globals.file)

# Call function printNumberOfClasses
def printNumberOfClasses():
    globals.num_classes = len(os.listdir(globals.data_train_images))
    print('Number of Classes: %i\n' % globals.num_classes, file = globals.file)