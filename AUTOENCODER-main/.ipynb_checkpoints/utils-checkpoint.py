#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : utils
# Description : Several utilities with no specific order.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 21-March-2021 - Creation
#               01-April-2021 - Added set_gpu, contrastive_loss and
#                               build_model_basename.
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import img_as_float
import tensorflow.keras.backend as K
import tensorflow as tf
import os

###############################################################################
# MONTAGE
# Creates an image containing all the images in the input. All the images
# in the input must be the same size.
# Input : theBatch : NumPy array of size (n,r,c,chan)
#                    The images to show. The first dimension (n) is the number
#                    of images. The second and third dimension (r,c) are the
#                    image sizes (rows and columns). The last dimension is the
#                    number of channels. In grayscale images it can either be
#                    1 or do not exist.
#         doPlot   : Boolean, optional. Plot the resulting image. The default
#                    is True.
#         savePath : String, optional. File name of the image to save or None
#                    to not save. The default is None.
# Output :outImage : NumPy matrix. The resulting image.
###############################################################################
def montage(theBatch,doPlot=True,savePath=None):
    # Determine the input dimensions
    if len(theBatch.shape)==4:
        (nItems,nRowsImage,nColsImage,nChan)=theBatch.shape
    elif len(theBatch.shape)==3:
        (nItems,nRowsImage,nColsImage)=theBatch.shape
        nChan=1
    else:
        sys.exit('Input shape must be (n,r,c,ch) or (n,r,c) where n is the number of images, r and c are the number of rows and columns and ch is the number of channels. This last dimension can be ommited for grayscale images.')

    # Determine the output dimensions
    nCols=int(np.ceil(np.sqrt(nItems)))
    nRows=int(np.ceil(nItems/nCols))

    # Create the output matrix
    outImage=np.zeros((nRows*nRowsImage,nCols*nColsImage,nChan)).astype('float')

    # Convert the batch to float images
    theBatch=img_as_float(theBatch)

    # Build the output matrix
    batchIndex=0
    for r in range(nRows):
        if batchIndex>=nItems:
            break
        for c in range(nCols):
            if batchIndex>=nItems:
                break
            outImage[r*nRowsImage:(r+1)*nRowsImage,c*nColsImage:(c+1)*nColsImage]=theBatch[batchIndex]
            batchIndex=batchIndex+1

    # If plot requested, do it
    if doPlot:
        plt.figure()
        plt.imshow(outImage)
        plt.axis('off')
        plt.show()

    # If save requested, do it
    if not (savePath is None):
        imsave(savePath,outImage)

    return outImage

###############################################################################
# SIMPLE TEXT PROGRESS BAR
###############################################################################
def progress_bar(curValue,maxValue):
    thePercentage=curValue/maxValue
    curSize=int(50*thePercentage)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*curSize, int(thePercentage*100)))
    sys.stdout.flush()

###############################################################################
# SET GPU ON/OFF
# MUST BE THE FIRST CALL. CANNOT BE CHANGED WHEN SET. TO CHANGE, RESTART KERNEL
###############################################################################
def set_gpu(useGPU=True):
    # Please note that this is a mix of cabala and magic (probably black magic)
    # from:
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    # This works on MY computer (Ubuntu 20 and CUDA Toolkit 10.1)
    if not useGPU:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import tensorflow as tf
        physicalDevices=tf.config.experimental.list_physical_devices('GPU')
        assert len(physicalDevices) > 0, "Not enough GPU hardware devices available"
        config=tf.config.experimental.set_memory_growth(physicalDevices[0], True)

###############################################################################
# CONTRASTIVE LOSS
###############################################################################
def contrastive_loss(y, preds, margin=1):
    # Explicitly cast the true class label data type to the predicted one.
    y = tf.cast(y, preds.dtype)
    # Calculate both losses (for loops and non loops)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    # Compute the contrastive loss
    loss = K.mean((1-y) * squaredPreds + y * squaredMargin)
    return loss

###############################################################################
# CONVENIENCE FUNCTION TO BUILD BASE MODEL FILE NAMES
###############################################################################
def build_model_basename(thePath,thePrefix,theList,theEpochs):
    # Build the model file name base
    baseName=os.path.join(thePath,thePrefix)
    for curItem in theList:
        baseName+='_%d'%curItem
    baseName+='_EPOCHS%d'%theEpochs
    return baseName

###############################################################################
# GET FILE NAMES IN PATH AND SPLITS THEM IF REQUESTED
# Gets all the name of the files in thePath with the extension theExtension and
# randomly shuffles the list.
# If smallSplitRatio==None, returns that list.
# If smallSplitRatio!=None, returns two list. The first contains (1-ratio) of
# the original list items and the second one contains ratio of the original
# list items.
###############################################################################
def get_filenames(thePath,theExtension,smallSplitRatio=None):
    # Get all the filenames and shuffle them
    fileNames=[os.path.join(thePath,f) for f in os.listdir(thePath) if os.path.isfile(os.path.join(thePath,f)) and f.endswith(theExtension)]
    np.random.shuffle(fileNames)

    # If no split requested, return the list
    if smallSplitRatio is None:
        return fileNames

    # Divide the file list in two sets
    numFiles=len(fileNames)
    cutIndex=int(smallSplitRatio*numFiles)

    # Return the two sets
    return fileNames[cutIndex:],fileNames[:cutIndex]