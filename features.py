#!/usr/bin/env python
# coding: utf-8

# Imports
from sklearn.cluster import MiniBatchKMeans
import cv2 as cv
import globals
import numpy as np

# Call function SIFT
def SIFT():
    # Initiate SIFT detector
    SIFT = cv.xfeatures2d.SIFT_create(nfeatures = 0,
                                      nOctaveLayers = 3,
                                      contrastThreshold = 0.04,
                                      edgeThreshold = 10,
                                      sigma = 1.6)

    return SIFT

# Call function SURF
def SURF():
    # Initiate SURF descriptor
    SURF = cv.xfeatures2d.SURF_create(hessianThreshold = 500,
                                      nOctaves = 4,
                                      nOctaveLayers = 3,
                                      extended = True,
                                      upright = False)

    return SURF

# Call function KAZE
def KAZE():
    # Initiate KAZE descriptor
    KAZE = cv.KAZE_create(extended = False,
                          upright = False,
                          threshold = 0.001,
                          nOctaves = 4,
                          nOctaveLayers = 4,
                          diffusivity = cv.KAZE_DIFF_PM_G2)

    return KAZE

# Call function BRIEF
def BRIEF():
    # Initiate BRIEF descriptor
    BRIEF = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes = 16,
                                                           use_orientation = False)

    return BRIEF

# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv.ORB_create(nfeatures = 300,
                        scaleFactor = 1.2,
                        nlevels = 2,
                        edgeThreshold = 2,
                        firstLevel = 0,
                        WTA_K = 2,
                        scoreType = 0,
                        patchSize = 2,
                        fastThreshold = 20)

    return ORB

# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv.BRISK_create(thresh = 30,
                            octaves = 3,
                            patternScale = 1.0)

    return BRISK

# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv.AKAZE_create(descriptor_type = cv.AKAZE_DESCRIPTOR_MLDB,
                            descriptor_size = 0,
                            descriptor_channels = 3,
                            threshold = 0.001,
                            nOctaves = 4,
                            nOctaveLayers = 4,
                            diffusivity = cv.KAZE_DIFF_PM_G2)

    return AKAZE

# Call function FREAK
def FREAK():
    # Initiate FREAK descriptor
    FREAK = cv.xfeatures2d.FREAK_create(orientationNormalized = True,
                                        scaleNormalized = True,
                                        patternScale = 22.0,
                                        nOctaves = 4)

    return FREAK

# Call function printDetector
def printDetector():
    print('Detector selected: %s\n' % globals.detector, file = globals.file)

# Call function printDescriptors
def printDescriptor():
    print('Descriptor selected: %s\n' % globals.descriptor, file = globals.file)

# Call function features
def features(image):
    # Find the keypoints
    keypoints = globals.detector.detect(image, None)

    # Compute the descriptors
    keypoints, descriptors = globals.descriptor.compute(image, keypoints)
    
    return keypoints, descriptors

# Call function extractFeatures
def extractFeatures(crop_list, featureFunc):
    # dictionary:
    # keypoints == kp
    # descriptors == desc

    descriptors = {}
    
    for i in range(0, len(crop_list)):
        kp, desc = featureFunc(crop_list[i])
        if type(desc) != type(None):
            descriptors[i] = desc
        
    return descriptors

# Call function groupAllFeatures
def groupAllFeatures(descriptors_hash):
    all_descriptors = []
    
    for subject, crops in descriptors_hash.items():
        for idx, desc in crops.items():
            all_descriptors.extend(desc)
            
    return all_descriptors

# Call function trainKMeans
def trainKMeans(descriptors):
    initial_size = 1 * globals.K
    batch_size = int(len(descriptors) / 3) 
    kmeans = MiniBatchKMeans(n_clusters = globals.K,
                             batch_size = batch_size,
                             init_size = initial_size,
                             verbose = 0).fit(descriptors)
    
    return kmeans

# Call function generateHistograms
def generateHistograms(descriptors, kmeans_centers):
    histograms = []
    
    for i, desc in descriptors.items(): 
        preds = kmeans_centers.predict(desc)
        hist, bin_edges = np.histogram(a = preds,
                                       bins = range(0, globals.K))
        histograms.append(hist)
    
    return histograms