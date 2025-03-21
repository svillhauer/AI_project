#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : AutoGenerator
# Description : Simple data generator (Sequence) to feed a Convolutional Auto
#               Encoder.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 26-March-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

from keras.utils import Sequence
import numpy as np
from utils import montage
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_float

class AutoGenerator(Sequence):
###############################################################################
# CONSTRUCTOR
# Input: fileNames   - List of the names of the image files in the dataset.
#        imgSize     - Desired image size
#        batchSize   - The size of the batches.
#        doRandomize - Randomize file order at the end of each epoch and
#                      apply some basic data augmentation (True/False)
###############################################################################
    def __init__(self,fileNames,imgSize=(64,64),batchSize=10,doRandomize=True):
        # Store input parameters
        self.fileNames=fileNames
        self.imgSize=imgSize
        self.batchSize=batchSize
        self.doRandomize=doRandomize

        # Get the number of images
        self.numImages=len(fileNames)

        # Additional initializations
        self.on_epoch_end()
###############################################################################
# CALLED AFTER EACH TRAINING EPOCH
###############################################################################
    def on_epoch_end(self):
        if self.doRandomize:
            np.random.shuffle(self.fileNames)

###############################################################################
# GET THE NUMBER OF BATCHES
###############################################################################
    def __len__(self):
        return int(np.ceil(self.numImages/float(self.batchSize)))

###############################################################################
# OUTPUT ONE DATA-LABEL PAIR. SINCE THIS IS AIMED AT AN AUTOENCODER,
# DATA AND LABEL ARE THE SAME.
###############################################################################
    def __getitem__(self,theIndex):
        X=[]

        # Compute start and end image indexes in the requested batch
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.numImages)

        # If randomization is requested, simple data augmentation will also
        # be performed.
        if self.doRandomize:
            theRandoms=np.random.random(bEnd-bStart)

        # For each image in the batch
        for i in range(bStart,bEnd):
            # Read the image
            curImage=img_as_float(imread(self.fileNames[i]))

            # Resize only if necessary
            if curImage.shape[:2]!=self.imgSize:
                curImage=resize(curImage,self.imgSize)

            # Apply random flip if randomization requested
            if self.doRandomize:
                curRandom=theRandoms[i-bStart]
                if curRandom<.25:
                    curImage=curImage[::-1,:]
                elif curRandom<.50:
                    curImage=curImage[:,::-1]
                elif curRandom<.75:
                    curImage=curImage[::-1,::-1]

            # Append the resulting image
            X.append(curImage)

        # Return the same data as X and y
        return np.array(X),np.array(X)

###############################################################################
# PLOT THE SPECIFIED BATCH
###############################################################################
    def plot_batch(self,batchNum):
        montage(self.__getitem__(batchNum)[0])

###############################################################################
# USAGE EXAMPLE: UNCOMMENT THE FOLLOWING CODE AND EXECUTE
###############################################################################

# # Get the file names
# import os
# basePath='../../DATA/AUTIMGTR/'
# fileNames=[os.path.join(basePath,f) for f in os.listdir(basePath) if os.path.isfile(os.path.join(basePath,f))]

# # Build the generator
# theGenerator=AutoGenerator(fileNames)

# # Plot a batch
# theGenerator.plot_batch(2)
