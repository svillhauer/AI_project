#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : ModelWrapper
# Description : Simple wrapper to ease the access to our models. Must be
#               used as a base class for each specific model.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 31-March-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from pickle import dump, load
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import plot_model
from time import time
from skimage.io import imread


class ModelWrapper:
    ###############################################################################
    # CONSTRUCTOR
    ###############################################################################
    def __init__(self):
        self.theModel = None
        self.trainHistory = None
        self.trainTime = None
        self.evaluationResults = None
        self.metricsNames = None

    ###############################################################################
    # CREATES AND COMPILES THE MODEL
    ###############################################################################
    def create(self):
        pass

    ###############################################################################
    # TRAIN
    ###############################################################################
    def fit(self, *args, **kwargs):
        tStart = time()
        self.trainHistory = self.theModel.fit(*args, **kwargs).history
        self.trainTime = time() - tStart
        self.metricsNames = self.theModel.metrics_names

        # Debug to check the metrics names
        print("Metrics Names after fit:", self.metricsNames)

    ###############################################################################
    # EVALUATE
    ###############################################################################
    def evaluate(self, *args, **kwargs):
        self.evaluationResults = self.theModel.evaluate(*args, **kwargs)

        # Debug to check evaluation results
        print("Evaluation Results:", self.evaluationResults)
        return self.evaluationResults

    ###############################################################################
    # PREDICT
    ###############################################################################
    def predict(self, *args, **kwargs):
        return self.theModel.predict(*args, **kwargs)

    ###############################################################################
    # PRINT MODEL SUMMARY
    ###############################################################################
    def summary(self):
        self.theModel.summary()

    ###############################################################################
    # CHECK IF A MODEL WITH THE SPECIFIED BASE NAME IS ALREADY SAVED
    ###############################################################################
    def is_saved(self, baseName):
        return os.path.exists(baseName + '.h5')

    ###############################################################################
    # PRINTS THE MODEL EVALUATION RESULTS
    ###############################################################################
    def print_evaluation(self):
        if not self.metricsNames or not self.evaluationResults:
            print("[ERROR] Metrics names or evaluation results are missing.")
            return
        if len(self.metricsNames) != len(self.evaluationResults):
            print(f"[ERROR] Length mismatch: metricsNames={len(self.metricsNames)}, "
                  f"evaluationResults={len(self.evaluationResults)}")
            return
        for i in range(len(self.evaluationResults)):
            print('%s : %.3f' % (self.metricsNames[i], self.evaluationResults[i]))

    ###############################################################################
    # SAVES MODEL AND HISTORY TO DISK IF POSSIBLE
    ###############################################################################
    def save(self, baseName, forceOverwrite=False):
        if forceOverwrite or (not self.is_saved(baseName)):
            self.theModel.save(baseName + '.h5')
            with open(baseName + '_HISTORY.pkl', 'wb') as outFile:
                dump([self.trainHistory, self.trainTime, self.evaluationResults, self.metricsNames], outFile)
            return True
        else:
            print('[SAVING ABORTED] Model file already exists. Use forceOverwrite=True to overwrite it.')
            return False

    ###############################################################################
    # LOADS MODEL AND HISTORY FROM DISK IF POSSIBLE
    ###############################################################################
    def load(self, baseName):
        if self.is_saved(baseName):
            self.theModel = load_model(baseName + '.h5')
            with open(baseName + '_HISTORY.pkl', 'rb') as inFile:
                [self.trainHistory, self.trainTime, self.evaluationResults, self.metricsNames] = load(inFile)
        else:
            print('[LOADING ABORTED] Model file not found.')

    ###############################################################################
    # PLOTS THE MODEL ARCHITECTURE
    ###############################################################################
    def plot(self, fileName='model.png'):
        plot_model(self.theModel, to_file=fileName, show_shapes=True,
                   show_layer_names=True, expand_nested=True)
        theImage = imread(fileName)
        plt.figure()
        plt.imshow(theImage)
        plt.show()
        return theImage

    ###############################################################################
    # PLOTS THE TRAINING HISTORY
    ###############################################################################
    def plot_training_history(self, plotTitle='TRAINING EVOLUTION'):
        # Get the keys in trainHistory to be plotted together
        nonVal = [theKey for theKey in self.trainHistory if not theKey.startswith('val_')]
        theVal = [theKey[4:] for theKey in self.trainHistory if theKey.startswith('val_')]
        pairsList = []
        labelsList = []
        for curKey in nonVal:
            if curKey in theVal:
                theVal.remove(curKey)
                pairsList.append([curKey, 'val_' + curKey])
            else:
                pairsList.append([curKey])
            labelsList.append(curKey)

        for curKey in theVal:
            pairsList.append(['val_' + curKey])
            labelsList.append(curKey)

        # Plot everything
        for i in range(len(pairsList)):
            plt.figure()
            for curItem in pairsList[i]:
                plt.plot(self.trainHistory[curItem])

            plt.title(plotTitle)
            plt.xlabel('epoch')
            plt.ylabel(labelsList[i])
            plt.legend(pairsList[i], loc='upper left')
            plt.show()




























# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# ###############################################################################
# # Name        : ModelWrapper
# # Description : Simple wrapper to ease the acces to our models. Must be
# #               used as base class for each specific models.
# # Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# # History     : 31-March-2021 - Creation
# # Citation    : Please, refer to the README file to know how to properly cite
# #               us if you use this software.
# ###############################################################################

# ###############################################################################
# # IMPORTS
# ###############################################################################

# from pickle import dump,load
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import os
# #from keras.utils.vis_utils import plot_model
# from tensorflow.keras.utils import plot_model

# from time import time
# from skimage.io import imread

# class ModelWrapper:

# ###############################################################################
# # CONSTRUCTOR
# ###############################################################################
#     def __init__(self):
#         self.theModel=None
#         self.trainHistory=None
#         self.trainTime=None
#         self.evaluationResults=None
#         self.metricsNames=None

# ###############################################################################
# # CREATES AND COMPILES THE MODEL
# ###############################################################################
#     def create(self):
#         pass

# ###############################################################################
# # TRAIN
# ###############################################################################
#     # def fit(self,*args,**kwargs):
#     #     tStart=time()
#     #     self.trainHistory=self.theModel.fit(*args,**kwargs).history
#     #     self.trainTime=time()-tStart
#     #     self.metricsNames=self.theModel.metrics_names


#     def fit(self, *args, **kwargs): 
#         tStart = time() 
#         self.trainHistory = self.theModel.fit(*args, **kwargs).history 
#         self.trainTime = time() - tStart 
#         # Check for full metrics names 
#     self.metricsNames = self.theModel.metrics_names 
#     if len(self.metricsNames) == 2 and self.metricsNames[1] == 'compile_metrics': 
#         # Replace 'compile_metrics' with individual metrics 
#         self.metricsNames = ['loss', 'accuracy', 'mse', 'mae']

# ###############################################################################
# # EVALUATE
# ###############################################################################
#     def evaluate(self,*args,**kwargs):
#         self.evaluationResults=self.theModel.evaluate(*args,**kwargs)
#         return self.evaluationResults

# ###############################################################################
# # PREDICT
# ###############################################################################
#     def predict(self,*args,**kwargs):
#         return self.theModel.predict(*args,**kwargs)

# ###############################################################################
# # PRINT MODEL SUMMARY
# ###############################################################################
#     def summary(self):
#         self.theModel.summary()

# ###############################################################################
# # CHECK IF A MODEL WITH THE SPECIFIED BASE NAME IS ALREADY SAVED
# ###############################################################################
#     def is_saved(self,baseName):
#         return os.path.exists(baseName+'.h5')

# ###############################################################################
# # PRINTS THE MODEL EVALUATION RESULTS
# ###############################################################################
#     def print_evaluation(self):
#         if self.evaluationResults is None:
#             print('[ERROR] Cannot print evaluation since the model has not been evaluated.')
#         else:
#             for i in range(len(self.evaluationResults)):
#                 print('%s : %.3f'%(self.metricsNames[i],self.evaluationResults[i]))


#     # def print_evaluation(self):
#     # if not self.metricsNames or not self.evaluationResults:
#     #     print("Evaluation results or metric names are empty.")
#     #     return
#     # if len(self.metricsNames) != len(self.evaluationResults):
#     #     print(f"Length mismatch: metricsNames={len(self.metricsNames)}, evaluationResults={len(self.evaluationResults)}")
#     #     return
#     # for i in range(len(self.evaluationResults)):
#     #     print('%s : %.3f' % (self.metricsNames[i], self.evaluationResults[i]))


# ###############################################################################
# # SAVES MODEL AND HISTORY TO DISK IF POSSIBLE
# ###############################################################################
#     def save(self,baseName,forceOverwrite=False):
#         if forceOverwrite or (not self.is_saved(baseName)):
#             self.theModel.save(baseName+'.h5')
#             with open(baseName+'_HISTORY.pkl','wb') as outFile:
#                 dump([self.trainHistory,self.trainTime,self.evaluationResults,self.metricsNames],outFile)
#             return True
#         else:
#             print('[SAVING ABORTED] Model file already exists. Use forceOverwrite=True to overwrite it.')
#             return False

# ###############################################################################
# # LOADS MODEL AND HISTORY FROM DISK IF POSSIBLE
# ###############################################################################
#     def load(self,baseName):
#         if self.is_saved(baseName):
#             self.theModel=load_model(baseName+'.h5')
#             with open(baseName+'_HISTORY.pkl','rb') as inFile:
#                 [self.trainHistory,self.trainTime,self.evaluationResults,self.metricsNames]=load(inFile)
#         else:
#             print('[LOADING ABORTED] Model file not found.')

# ###############################################################################
# # PLOTS THE MODEL ARCHITECTURE
# ###############################################################################
#     def plot(self,fileName='model.png'):
#         plot_model(self.theModel,to_file=fileName,show_shapes=True,
#                    show_layer_names=True,expand_nested=True)
#         theImage=imread(fileName)
#         plt.figure()
#         plt.imshow(theImage)
#         plt.show()
#         return theImage

# ###############################################################################
# # PLOTS THE TRAINING HISTORY
# ###############################################################################
#     def plot_training_history(self,plotTitle='TRAINING EVOLUTION'):
#         # Get the keys in trainHistory to be plotted together
#         nonVal=[theKey for theKey in self.trainHistory if not theKey.startswith('val_')]
#         theVal=[theKey[4:] for theKey in self.trainHistory if theKey.startswith('val_')]
#         pairsList=[]
#         labelsList=[]
#         for curKey in nonVal:
#             if curKey in theVal:
#                 theVal.remove(curKey)
#                 pairsList.append([curKey,'val_'+curKey])
#             else:
#                 pairsList.append([curKey])
#             labelsList.append(curKey)

#         for curKey in theVal:
#             pairsList.append(['val_'+curKey])
#             labelsList.append(curKey)

#         # Plot everything
#         for i in range(len(pairsList)):
#             plt.figure()
#             for curItem in pairsList[i]:
#                 plt.plot(self.trainHistory[curItem])

#             plt.title(plotTitle)
#             plt.xlabel('epoch')
#             plt.ylabel(labelsList[i])
#             plt.legend(pairsList[i],loc='upper left')
#             plt.show()