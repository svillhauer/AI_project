#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : LoopModelTest
# Description : Usage example of LoopModel
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# SET GPU
###############################################################################

# These must be the first lines to execute. Restart the kernel before.
# If you don't have GPU/CUDA or this does not work, just set it to False or
# (preferrably) remove these two lines.
from utils import set_gpu
set_gpu(False)

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from automodel import AutoModel
from loopreader import LoopReader
from loopmodel import LoopModel
from loopgenerator import LoopGenerator
from utils import build_reader_basename, montage
import matplotlib.pyplot as plt
# For generation of testing and validation data 
import os 
import glob
import random
import shutil

###############################################################################
# PARAMETERS
###############################################################################

#------------------------------------------------------------------------------
# LOOPREADER PARAMETERS
#------------------------------------------------------------------------------

image_folder = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
PATH_TRAIN_DATASET= "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
PATH_TEST_DATASET= "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
DATA_CONTINUOUS=False;
DATA_GROUPS=[[0,0.5],[0.5,2]]
DATA_SEPARATION=3
DATA_BALANCE=True
DATA_INVERT=False
DATA_MIRROR=True
DATA_CATEGORICAL=False
DATA_NORMALIZE_NO_CATEGORICAL=False

#------------------------------------------------------------------------------
# LOOPGENERATOR PARAMETERS
#------------------------------------------------------------------------------

IMG_SIZE=(64,64)
BATCH_SIZE=10

#------------------------------------------------------------------------------
# LOOPMODEL PARAMETERS
#------------------------------------------------------------------------------

DENSE_LAYERS=[128,16]
CATEGORICAL_CLASSES=0
LOSS_FUNCTION='binary_crossentropy'
LOOP_METRICS=['accuracy']
DO_REGRESSION=False
RETRAIN_FEATURE_EXTRACTOR=True
LOOP_EPOCHS=10
VAL_SPLIT=0.2
AUTOENCODER_BASENAME= "/home/svillhauer/Desktop/AI_project/AUTOENCODER-main/REALDATASETMODELS/AUTOENCODER_128_128_16_EPOCHS100"

###############################################################################
# CREATE THE TRAIN AND VALIDATION LOOP GENERATORS
###############################################################################

# Create and save or load the train loop reader
theReader=LoopReader()
readerParams={'basePath':PATH_TRAIN_DATASET,
              'outContinuous':DATA_CONTINUOUS,
              'theGroups':DATA_GROUPS,
              'stepSeparation':DATA_SEPARATION,
              'doBalance':DATA_BALANCE,
              'doInvert':DATA_INVERT,
              'doMirror':DATA_MIRROR,
              'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
              'toCategorical':DATA_CATEGORICAL}
readerBaseName=build_reader_basename(**readerParams)
if theReader.is_saved(readerBaseName):
    theReader.load(readerBaseName)
else:
    theReader.create(**readerParams)
    theReader.save(readerBaseName)

# Split the reader loop specs into train and validation
numSpecs=theReader.loopSpecs.shape[1]
cutIndex=int(VAL_SPLIT*numSpecs)
trainSpecs=theReader.loopSpecs[:,cutIndex:]
valSpecs=theReader.loopSpecs[:,:cutIndex]

# Build train and validation data generators
trainGenerator=LoopGenerator(trainSpecs,
                             theReader.get_image,
                             imgSize=IMG_SIZE,
                             batchSize=BATCH_SIZE,
                             doRandomize=True)
valGenerator=LoopGenerator(valSpecs,
                           theReader.get_image,
                           imgSize=IMG_SIZE,
                           batchSize=BATCH_SIZE,
                           doRandomize=False)

###############################################################################
# CREATE THE LOOP DETECTOR MODEL
###############################################################################

autoModel=AutoModel()
autoModel.load(AUTOENCODER_BASENAME)
loopModel=LoopModel()
loopModel.create(featureExtractorModel=autoModel.encoderModel,
                 denseLayers=DENSE_LAYERS,
                 categoricalClasses=CATEGORICAL_CLASSES,
                 lossFunction=LOSS_FUNCTION,
                 theMetrics=LOOP_METRICS,
                 doRegression=DO_REGRESSION,
                 trainFeatureExtractor=RETRAIN_FEATURE_EXTRACTOR)

###############################################################################
# TRAIN THE MODEL
###############################################################################

print('[TRAINING]')
history = loopModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=LOOP_EPOCHS)

import pandas as pd

# Save training metrics to CSV
metrics_df = pd.DataFrame({
    'epoch': list(range(1, LOOP_EPOCHS + 1)),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
    'train_acc': history.history.get('accuracy', history.history.get('acc')),
    'val_acc': history.history.get('val_accuracy', history.history.get('val_acc')),
})
metrics_df.to_csv("loopmodel_training_metrics.csv", index=False)

print('[TRAINING READY]')

###############################################################################
# PLOT AND SAVE LOSS AND VAL_LOSS
###############################################################################

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
loss_plot_path = 'loss_plot.png'
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss plot saved as: {loss_plot_path}")

###############################################################################
# CREATE THE TEST LOOP GENERATOR
###############################################################################

theReader=LoopReader()
readerParams={'basePath':PATH_TEST_DATASET,
              'outContinuous':DATA_CONTINUOUS,
              'theGroups':DATA_GROUPS,
              'stepSeparation':DATA_SEPARATION,
              'doBalance':DATA_BALANCE,
              'doInvert':DATA_INVERT,
              'doMirror':DATA_MIRROR,
              'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
              'toCategorical':DATA_CATEGORICAL}
readerBaseName=build_reader_basename(**readerParams)
print(f"Reader basename path being used: {readerBaseName}.pkl")

if theReader.is_saved(readerBaseName):
    theReader.load(readerBaseName)
else:
    theReader.create(**readerParams)
    #theReader.save(readerBaseName)
    theReader.save(readerBaseName, forceOverwrite=True)


testGenerator=LoopGenerator(theReader.loopSpecs,
                            theReader.get_image,
                            imgSize=IMG_SIZE,
                            batchSize=BATCH_SIZE,
                            doRandomize=False)

###############################################################################
# EVALUATE THE MODEL
###############################################################################

print('[EVALUATING]')
loopModel.evaluate(testGenerator)
print('[EVALUATED]')

# Print some model weights, for debugging 
print("\nModel summary:")
loopModel.theModel.summary()

print("\nSample model weights (first layer):")
weights = loopModel.theModel.get_weights()
for i, w in enumerate(weights):
    print(f"Layer {i}: shape {w.shape}")
    print(w.flatten()[:5])  # just print first 5 values for brevity
    if i >= 1:
        break


# Save the trained model
MODEL_SAVE_PATH = '/home/svillhauer/Desktop/AI_Project/SNNLOOP-main/AI_PROJECT2/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16'
loopModel.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}.h5")

# Make sue model is saved properly, debugging step 
print("Files saved:")
for f in os.listdir(os.path.dirname(MODEL_SAVE_PATH)):
    if os.path.basename(MODEL_SAVE_PATH) in f:
        print(f)




###############################################################################
# PRINT AND SAVE PREDICTIONS AS MONTAGE
###############################################################################

[X,y]=testGenerator.__getitem__(0)

raw_predictions = loopModel.predict(X)
interpreted_predictions = np.round(raw_predictions)

print("First 15 raw predictions:", raw_predictions.flatten()[:15]) # Print raw predictions, ensure loops are detected
print("First 15 rounded predictions:", interpreted_predictions.flatten()[:15])
print("First 15 ground truths:", y[:15])




interpreted_predictions = np.round(raw_predictions)

# Modify red channel based on predictions
for i in range(X[0].shape[0]):
    X[0][i,:,:,0]=interpreted_predictions[i]
    X[1][i,:,:,0]=interpreted_predictions[i]

# Save montage for predictions
montage_image1 = montage(X[0])
montage_image2 = montage(X[1])

plt.figure(figsize=(12, 6))
plt.title("Montage of Predictions - Input 1")
plt.imshow(montage_image1, cmap='gray')
plt.axis('off')
plt.tight_layout()
montage1_path = 'montage_input1.png'
plt.savefig(montage1_path)
plt.close()

plt.figure(figsize=(12, 6))
plt.title("Montage of Predictions - Input 2")
plt.imshow(montage_image2, cmap='gray')
plt.axis('off')
plt.tight_layout()
montage2_path = 'montage_input2.png'
plt.savefig(montage2_path)
plt.close()

print(f"Montage for Input 1 saved as: {montage1_path}")
print(f"Montage for Input 2 saved as: {montage2_path}")

###############################################################################
# FINAL SUMMARY
###############################################################################
print('[TRAINING, EVALUATION, AND VISUALIZATION COMPLETE]')











# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# ###############################################################################
# # Name        : LoopModelTest
# # Description : Usage example of LoopModel
# # Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# # History     : 4-April-2021 - Creation
# # Citation    : Please, refer to the README file to know how to properly cite
# #               us if you use this software.
# ###############################################################################

# ###############################################################################
# # SET GPU
# ###############################################################################

# # These must be the first lines to execute. Restart the kernel before.
# # If you don't have GPU/CUDA or this does not work, just set it to False or
# # (preferrably) remove these two lines.
# from utils import set_gpu
# set_gpu(False)

# ###############################################################################
# # IMPORTS
# ###############################################################################

# import numpy as np
# from automodel import AutoModel
# from loopreader import LoopReader
# from loopmodel import LoopModel
# from loopgenerator import LoopGenerator
# from utils import build_reader_basename,montage
# # For generation of testing and validation data 
# import os 
# import glob
# import random
# import shutil

# ###############################################################################
# # PARAMETERS
# ###############################################################################

# #------------------------------------------------------------------------------
# # LOOPREADER PARAMETERS
# #------------------------------------------------------------------------------

# # SOURCE_DIR = "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/SAMPLE_RANDOM/IMAGES" 
# # PATH_TRAIN_DATASET = "/home/svillhauer/Desktop/AI Project/SNNLOOP-main/IMAGES/LOOPTRAIN"        # The desired training directory
# # PATH_TEST_DATASET = "/home/svillhauer/Desktop/AI Project/SNNLOOP-main/IMAGES/LOOPTEST"  

# # # Create the train and test directories if they don't exist
# # os.makedirs(PATH_TRAIN_DATASET, exist_ok=True)
# # os.makedirs(PATH_TEST_DATASET, exist_ok=True)


# # for f in glob.glob(os.path.join(PATH_TRAIN_DATASET, "*")):
# #     print(f"Removing: {f}")
# #     os.remove(f)

# # for f in glob.glob(os.path.join(PATH_TEST_DATASET, "*")):
# #     print(f"Removing: {f}")
# #     os.remove(f)



# # # # Specify the name of the CSV file
# # # csv_file_name = "OVERLAP.csv"  # Replace with the actual CSV file name

# # # # Construct full source and destination file paths
# # # source_file_path = os.path.join("/home/svillhauer/Desktop/AI Project/UCAMGEN-main/SAMPLE_RANDOM/" , csv_file_name)
# # # destination_file_path = os.path.join(PATH_TRAIN_DATASET, csv_file_name)

# # # # Check if the file exists in the source directory
# # # if os.path.exists(source_file_path):
# # #     # Move the file
# # #     shutil.move(source_file_path, destination_file_path)
# # #     print(f"Moved {csv_file_name} to {PATH_TRAIN_DATASET}")
# # # else:
# # #     print(f"{csv_file_name} does not exist in {SOURCE_DIR}")

# # # Gather all image files (adjust extensions as needed)
# # # You can also include multiple extensions if necessary, for example:
# # # image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# # image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.png')]

# # # Shuffle the images to ensure a random split
# # random.shuffle(image_files)

# # # Calculate how many should go to test (10% for example)
# # total_images = len(image_files)
# # test_count = int(total_images * 0.10)  # 10% test
# # train_count = total_images - test_count

# # # Split the list
# # train_files = image_files[:train_count]
# # test_files = image_files[train_count:]

# # # Copy files to train directory
# # for f in train_files:
# #     src_path = os.path.join(SOURCE_DIR, f)
# #     dst_path = os.path.join(PATH_TRAIN_DATASET, f)
# #     shutil.copy2(src_path, dst_path)

# # # Copy files to test directory
# # for f in test_files:
# #     src_path = os.path.join(SOURCE_DIR, f)
# #     dst_path = os.path.join(PATH_TEST_DATASET, f)
# #     shutil.copy2(src_path, dst_path)

# # print(f"Done! {len(train_files)} images copied to {PATH_TRAIN_DATASET} and {len(test_files)} images copied to {PATH_TEST_DATASET}.")


# image_folder = "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/REALDATASET/IMAGES"
# # "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/SAMPLE_RANDOM/IMAGES"
# # # Iterate through all files in the folder
# # for filename in os.listdir(image_folder):
# #     # Check if the filename contains "(1)" and ends with a valid image extension
# #     if "(1)" in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
# #         # Build the full path to the file
# #         file_path = os.path.join(image_folder, filename)
# #         print(f"Deleting duplicate: {file_path}")  # Optional: For debugging
# #         os.remove(file_path)  # Delete the file

# # print("Duplicate images deleted.")

# # A loop reader is, basically, in charge of pre-processing an UCAMGEN dataset
# # to be used under a particular configuration. Please check loopreadertest.py
# # for more information about loopreaders and the UCAMGEN repository for more
# # information about the UCAMGEN format.
# PATH_TRAIN_DATASET= "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/REALDATASET/IMAGES"
# # "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/SAMPLE_RANDOM/IMAGES" #'../../DATA/LOOPDSTRSMALL'
# PATH_TEST_DATASET= "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/REALDATASET/IMAGES"
# # "/home/svillhauer/Desktop/AI Project/UCAMGEN-main/SAMPLE_RANDOM/IMAGES" #'../../DATA/LOOPDSTRSMALL'
# DATA_CONTINUOUS=False;
# DATA_GROUPS=[[0,0.5],[0.5,2]]
# DATA_SEPARATION=3
# DATA_BALANCE=True
# DATA_INVERT=False
# DATA_MIRROR=True
# DATA_CATEGORICAL=False
# DATA_NORMALIZE_NO_CATEGORICAL=False

# #------------------------------------------------------------------------------
# # LOOPGENERATOR PARAMETERS
# #------------------------------------------------------------------------------

# # A loop generator inherits from Keras Sequences and can be used to train/
# # test/... a Keras model. Please check loopgeneratortest.py for more
# # information.
# IMG_SIZE=(64,64)
# BATCH_SIZE=10

# #------------------------------------------------------------------------------
# # LOOPMODEL PARAMETERS
# #------------------------------------------------------------------------------

# # Dense layers to use after the siamese branches to perform classification
# # or regression. For example, [128,16] means two dense layers, the first
# # one with 128 units and the second one with 2 units. These layers are
# # followed by the output layer.
# DENSE_LAYERS=[128,16]

# # Used to decide the output layer size. Since in this example the output is
# # not categorical (DATA_CATEGORICAL=False) we set CATEGORICAL_CLASSES to 0.
# # This ensures a single unit in the output layer.
# CATEGORICAL_CLASSES=0

# # Loss function to use. Since we have two classes in this example (two groups
# # defined in the loop reader), let's use binary crossentropy
# LOSS_FUNCTION='binary_crossentropy'

# # Metrics to measure during training, evaluation, testing, ... Must be a list.
# # Since we have 2 classes in this example, let's measure the accuracy.
# LOOP_METRICS=['accuracy']

# # The loop model is not aimed at regression (since the reader output is
# # not set to continuous)
# DO_REGRESSION=False

# # Let's re-train the feature extractor (encoder). This improves the results
# # and, being already pre-trained, it begins with quite good weights and
# # is relatively fast.
# RETRAIN_FEATURE_EXTRACTOR=True

# # Number of training epochs.
# LOOP_EPOCHS=10

# #------------------------------------------------------------------------------
# # OTHER PARAMETERS
# #------------------------------------------------------------------------------

# # In this example, the train set is split into train and validation data.
# # Thus, 2 readers will be created (train and test) but 3 generators used
# # (train, validation and test). To accomplish this, the data from the train
# # reader is split into train and validation using the following proportion.
# VAL_SPLIT=0.2

# # The feature extraction is performed by the encoder part of an autoencoder.
# # It is extremely advisable for the feature extractor to be pre-trained
# # even if it has to be re-trained again as part of the loop detector. In this
# # example we will use a pre-trained feature extractor (encoder of an pre-
# # trained autoencoder), and the following is the base name to load it.
# # Please check the repository AUTOENCODER to learn about this aspect.
# #AUTOENCODER_BASENAME='../../DATA/MODELS/AUTOENCODER_128_128_32_EPOCHS100'

# AUTOENCODER_BASENAME= "/home/svillhauer/Desktop/AI Project/AUTOENCODER-main/REALDATASETMODELS/AUTOENCODER_128_128_16_EPOCHS100"
# #"/home/svillhauer/Desktop/AI Project/AUTOENCODER-main/MODELSFINAL/AUTOENCODER_128_128_16_EPOCHS100"



# ###############################################################################
# # CREATE THE TRAIN AND VALIDATION LOOP GENERATORS
# ###############################################################################

# # Create and save or load the train loop reader
# theReader=LoopReader()
# readerParams={'basePath':PATH_TRAIN_DATASET,
#               'outContinuous':DATA_CONTINUOUS,
#               'theGroups':DATA_GROUPS,
#               'stepSeparation':DATA_SEPARATION,
#               'doBalance':DATA_BALANCE,
#               'doInvert':DATA_INVERT,
#               'doMirror':DATA_MIRROR,
#               'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
#               'toCategorical':DATA_CATEGORICAL}
# readerBaseName=build_reader_basename(**readerParams)
# if theReader.is_saved(readerBaseName):
#     theReader.load(readerBaseName)
# else:
#     theReader.create(**readerParams)
#     theReader.save(readerBaseName)

# # Split the reader loop specs into train and validation
# numSpecs=theReader.loopSpecs.shape[1]
# cutIndex=int(VAL_SPLIT*numSpecs)
# trainSpecs=theReader.loopSpecs[:,cutIndex:]
# valSpecs=theReader.loopSpecs[:,:cutIndex]

# # Build train and validation data generators. Note that the train generator
# # is randomized whilst the validation generator is not. Randomizing the train
# # generator helps in reducing overfitting but randomizing the validation
# # usually only leads to larger computation times. That is why validation
# # generator is not randomized.
# trainGenerator=LoopGenerator(trainSpecs,
#                              theReader.get_image,
#                              imgSize=IMG_SIZE,
#                              batchSize=BATCH_SIZE,
#                              doRandomize=True)
# valGenerator=LoopGenerator(valSpecs,
#                            theReader.get_image,
#                            imgSize=IMG_SIZE,
#                            batchSize=BATCH_SIZE,
#                            doRandomize=False)

# ###############################################################################
# # CREATE THE LOOP DETECTOR MODEL
# ###############################################################################

# # Load the feature extractor. Actually, the whole autoencoder is loaded
# # but only the encoder will be used as feature extractor. Please check the
# # AUTOENCODER repository for more information.
# autoModel=AutoModel()
# autoModel.load(AUTOENCODER_BASENAME)

# # Create the model
# loopModel=LoopModel()
# loopModel.create(featureExtractorModel=autoModel.encoderModel,
#                  denseLayers=DENSE_LAYERS,
#                  categoricalClasses=CATEGORICAL_CLASSES,
#                  lossFunction=LOSS_FUNCTION,
#                  theMetrics=LOOP_METRICS,
#                  doRegression=DO_REGRESSION,
#                  trainFeatureExtractor=RETRAIN_FEATURE_EXTRACTOR)


# ###############################################################################
# # TRAIN THE MODEL
# ###############################################################################

# print('[TRAINING]')
# loopModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=LOOP_EPOCHS)
# print('[TRAINING READY]')

# ###############################################################################
# # CREATE THE TEST LOOP GENERATOR
# ###############################################################################

# # Create and save or load the test loop reader. Please not that since we are
# # overwriting the train/validation reader, the train and validation loop
# # generator will no longer be valid.
# theReader=LoopReader()
# readerParams={'basePath':PATH_TEST_DATASET,
#               'outContinuous':DATA_CONTINUOUS,
#               'theGroups':DATA_GROUPS,
#               'stepSeparation':DATA_SEPARATION,
#               'doBalance':DATA_BALANCE,
#               'doInvert':DATA_INVERT,
#               'doMirror':DATA_MIRROR,
#               'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
#               'toCategorical':DATA_CATEGORICAL}
# readerBaseName=build_reader_basename(**readerParams)
# if theReader.is_saved(readerBaseName):
#     theReader.load(readerBaseName)
# else:
#     theReader.create(**readerParams)
#     theReader.save(readerBaseName)

# # Build test data generator. Note that, contrarily to the train reader,
# # this one is not split. Also note that, similary to the validation generator,
# # the test one is not randomized.
# testGenerator=LoopGenerator(theReader.loopSpecs,
#                             theReader.get_image,
#                             imgSize=IMG_SIZE,
#                             batchSize=BATCH_SIZE,
#                             doRandomize=False)


# ###############################################################################
# # EVALUATE AND SAVE THE MODEL
# ###############################################################################

# print('[EVALUATING]')
# loopModel.evaluate(testGenerator)
# print('[EVALUATED]')


# # # Save the trained model
# # MODEL_SAVE_PATH = '/home/svillhauer/Desktop/AI Project/SNNLOOP-main/MODELSREALDATA/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16'
# # loopModel.save(MODEL_SAVE_PATH)
# # print(f"Model saved to {MODEL_SAVE_PATH}.h5")


# ###############################################################################
# # PRINT AND PLOT SOME RESULTS
# ###############################################################################

# # Print the evaluation results. Note that they are saved. So, they can be also
# # accessed when the model is loaded.

# print('[EVALUATION RESULTS]')
# loopModel.print_evaluation()

# # Get the first batch
# # The data in X contains two nparrays. Each array is a batch of images. Each
# # images in one array relates to the corresponding image in the other array
# # depending on the values of y, which state the class (in this case, class=0
# # means low or no overlap and class 1 means large overlap)
# [X,y]=testGenerator.__getitem__(0)

# # Predict on the batch
# raw_predictions = loopModel.predict(X)


# # Print raw predictions
# print("Raw predictions (before rounding or thresholding):")
# print(raw_predictions)

# # Interpret predictions (round or threshold)
# interpreted_predictions = np.round(raw_predictions)

# # Print interpreted predictions
# print("Interpreted predictions (after rounding):")
# print(interpreted_predictions)

# # Print ground truth labels for comparison
# print("Ground truth labels:")
# print(y)


# # Let's predict the first batch. The output is rounded since we have 2 classes.
# # That is, we round the prediction to be 0 or 1.
# thePredictions=np.round(loopModel.predict(X))

# # To provide a "clear" representation, let's change the red channel of the
# # images to their predicted class (so, class 1 will have be sort of red and
# # the other ones sort of... non-red). Note that LoopGenerator can also work
# # with grayscale images. So, this "approach" to visualize loop information
# # would not work with grayscale images. Obviously.
# for i in range(X[0].shape[0]):
#     X[0][i,:,:,0]=thePredictions[i]
#     X[1][i,:,:,0]=thePredictions[i]

# # Now plot the modified images using montage
# montage(X[0])
# montage(X[1])