# Convolutional Autoencoder using Keras and Tensorflow

The repository contains some convenience objects and examples to build, train and evaluate a convolutional autoencoder using Keras. The used Keras and Tensorflow.

If you use this software, please cite the following paper:

A. Burguera. Lightweight Underwater Visual Loop Detection and Classification using a Siamese Convolutional Neural Network. In 13th IFAC Conference on Control Applications in Marine Systems, Robotics, and Vehicles (CAMS), Oldenburg (Germany) - On-line, 2021.

## Credits

* Antoni Burguera (antoni dot burguera at uib dot es)

## Understanding the system

The main classes are:

* **AutoGenerator** : Inherits from Keras Sequence. Designed to feed the AutoEncoder during training and testing.
* **AutoModel** : Wrapper to ease the autoencoder creation, training, evaluation, saving and loading as well as printing and plotting some stats.

There are some auxiliary classes/modules/files:

* **main** : Main code to train and evaluate several autoencoder configurations.
* **autotools** : Helper functions used in main.
* **example** : Step by step usage example.
* **ModelWrapper** : Utility class we use to ease the constructions of our models, including AutoModel. It is used here as the base class of AutoModel.
* **utils** : Contains some general-purpose functions. Some of them are not used in this project.

## Using the system

Creating, training and evaluating an autoencoder is straightforward and only involves the following steps:

* Prepare the train, evaluation and test **AutoGenerators**.
* **Instantiate** an AutoModel: theModel=AutoModel()
* **Create** the AutoModel with the desired parameters: theModel.create(...)
* **Train** the model: theModel.fit(x=trainGenerator, validation_data=valGenerator, ...)
* **Evaluate** the model: theModel.evaluate(testGenerator, ...)
* **Save** the model: theModel.save(fileName)

Then, you can print some stats:

* Model **summary**: theModel.summary()
* Plot **architecture**: theModel.plot()
* Plot **training history** (loss, metrics, ...): theModel.plot_training_history()
* Print **evaluation results**: theModel.print_evaluation()

In order to use the trained autoencoder there are a few convenience methods:

* **Autoencode** a batch of images: thePredictions=theModel.predict(theBatch)
* **Encode** a batch of images: theFeatures=theModel.encode(theBatch)
* **Decode** a batch of features: thePredictions=theModel.decode(theFeatures)

All these steps are clearly explained in **example.py**. You can run this example directly just by making the paths (PATH_TRAIN and PATH_TEST) to point to your datasets and configuring (if desired) the shape of the images (SHAPE_IMG) and the autoencoder structure (FILTERS).

You can also configure the **main.py** easily to train and evaluate several autoencoder architectures.

Please note that if you do not have GPU support you should change set_gpu(True) to set_gpu(False) at the beginning of example.py and main.py. **Actually, if you do not have GPU suppport it is advisable to remove the call to set_gpu**. Even if you have GPU, this may not work. Read the **Troubleshooting** section.

## Requirements

To execute this software, you will need:

* Python 3
* Keras
* Tensorflow
* NumPy
* Matplotlib
* Pickle
* SciKit-Image

## Troubleshooting

The GPU activation/deactivation function (set_gpu) works on my computer (Ubuntu 20 and CUDA Toolkit 10.1) but may not work on other computers, even if they use Ubuntu 20 and CUDA 10.1. Installing CUDA on Ubuntu 20 and making it useable by Keras+Tensorflow is (at least in April 2020) a true nightmare. Really. I don't even know why it works today and didn't work yesterday. So, if it doesn't work do not blame me. **Just remove the set_gpu call and use your own or execute the code on CPU.**

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it or, eventually, create a time paradox, the result of which could cause a chain reaction that would unravel the very fabric of the space-time continuum and destroy the entire universe. Some users have reported inter-dimensional portals being created and letting demon-like creatures enter our reality after executing this code because it is not dead which can eternal lie, and with strange aeons even death may die. Just be careful and try to understand everything before using it. If you have questions, please carefully read the code and the paper. If this doesn't help, contact us.