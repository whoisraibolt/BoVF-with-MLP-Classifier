#!/usr/bin/env python
# coding: utf-8

# Imports
import joblib as joblib

# Call function saveBoVF
def saveBoVF(kmeans,
             descriptor_name,
			 dataset_name):

    joblib.dump(value = kmeans,
                filename = 'Saves/Datasets/%s/BoVF-Features/%s-BoVF-K-Means.pkl' % (dataset_name, descriptor_name))

# Call function saveDescriptors
def saveDescriptors(feature_vector,
                    descriptor_name,
			        dataset_name,
                    flag):

    joblib.dump(value = feature_vector,
                filename = 'Saves/Datasets/%s/Descriptors/%s-%s-desc.pkl' % (dataset_name, descriptor_name, flag))

# Call function saveMLP
def saveMLP(model,
            descriptor_name,
			dataset_name,
            model_name):

    joblib.dump(value = model, 
                filename = 'Saves/Datasets/%s/MLP-Models/%s-%s.pkl' % (dataset_name, descriptor_name, model_name))