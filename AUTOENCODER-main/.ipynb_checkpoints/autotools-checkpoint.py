#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : autotools
# Description : Convenience functions to train, evaluate, save and visualize
#               autoencoders.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 02-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from autogenerator import AutoGenerator
from automodel import AutoModel
from utils import build_model_basename, get_filenames,montage
import numpy as np

###############################################################################
# BUILDS, TRAINS, EVALUATES AND SAVES ONE AUTOENCODER
# Input  : pathTrain  - Path to the training images folder.
#          pathTest   - Path to the test images folder.
#          pathModels - Path to the folder to save the models.
#          extImage   - File extension for the images in the datasets.
#          inputShape - Input image shape.
#          theFilters - Number of convolutions in the encoder and decoder
#                       layers.
#          numEpochs  - Number of training epochs.
#          splitVal   - Ratio of the training dataset to be used as validation.
###############################################################################

def autobuild(pathTrain, pathTest, pathModels, extImage='png',
              inputShape=(64,64,3), theFilters=[128,128,64],
              numEpochs=100,splitVal=0.2):

    ###########################################################################
    # Check if the model already exists.
    ###########################################################################

    # Create the autoencoder object
    theModel=AutoModel()

    # Build the file name corresponding to that model
    baseName=build_model_basename(pathModels, 'AUTOENCODER', theFilters, numEpochs)
    print('PROCESSING MODEL %s'%baseName)

    # If the model is already saved, do nothing
    if theModel.is_saved(baseName):
        print('[ ABORTING ] The model seems to be already trained and saved.')
        return baseName

    ###########################################################################
    # Prepare training data
    ###########################################################################

    # Get the training and validation file names
    trainFileNames,valFileNames=get_filenames(pathTrain,extImage,splitVal)

    # Create the training and validation data generators
    trainGenerator=AutoGenerator(trainFileNames,imgSize=inputShape[:2],doRandomize=True)
    valGenerator=AutoGenerator(valFileNames,imgSize=inputShape[:2],doRandomize=False)

    ###########################################################################
    # Create and train the model
    ###########################################################################

    # Create the model and compile it
    theModel.create(inputShape=inputShape,theFilters=theFilters)

    # Train the model
    print('TRAINING THE MODEL')
    theModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=numEpochs)

    ###########################################################################
    # Prepare test data
    ###########################################################################

    # Get the test file names
    testFileNames=get_filenames(pathTest,extImage)

    # Build the test data generator
    testGenerator=AutoGenerator(testFileNames,imgSize=inputShape[:2],doRandomize=False)

    ###########################################################################
    # Evaluate the model
    ###########################################################################

    print('EVALUATING THE MODEL')
    theModel.evaluate(testGenerator)

    ###########################################################################
    # Save
    ###########################################################################

    theModel.save(baseName)

    return baseName

###############################################################################
# PRINTS AND PLOTS ARCHITECTURAL, TRAINING AND EVALUATION INFORMATIO ABOUT
# A SAVED MODEL.
# Input : baseName  - Model file base name
#         testBatch - Batch of example images to encode/decode. Must be of
#                     shape (n,64,64,3) where n is the number of images
# Note  : Executing this function is quite verbose and not really practical.
#         Think of it as an example to see how to access and print stats.
###############################################################################

def autoshow(baseName,testBatch=None):

    ###########################################################################
    # Check if the model already exists and load it if possible
    ###########################################################################

    # Create the autoencoder object
    theModel=AutoModel()

    # If the model is already saved, do nothing
    if not (theModel.is_saved(baseName)):
        print('[ ABORTING ] Model file %s not found. Please execute autobuild before.'%baseName)
        return
    print('SHOWING MODEL %s'%baseName)

    # Load the model
    theModel.load(baseName)

    ###########################################################################
    # Show modelwrapper built-in stats
    ###########################################################################

    theModel.summary()                          # Print the summary
    theModel.plot()                             # Plot the structure
    theModel.plot_training_history(baseName)    # Training stats
    theModel.print_evaluation()                 # Evaluation stats

    ###########################################################################
    # Process one batch of test images and plot the results if requested
    ###########################################################################

    if not (testBatch is None):
        # Encode the batch
        theFeatures=theModel.encode(testBatch)

        # Decode the batch
        thePredictions=theModel.decode(theFeatures)

        # Plot both data and predictions together
        jointData=np.vstack((testBatch,thePredictions))
        montage(jointData)