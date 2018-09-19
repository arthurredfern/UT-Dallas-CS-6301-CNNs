################################################################################
#
# File
#
#    Code03MNIST.py
#
# Purpose
#
#    MNIST digit classification with a 2 level fully connected network using
#    TensorFlow with the Keras interface
#
################################################################################


################################################################################
#
# History
#
#     A. Singh      2018-09-17    Created as part of Homework assignment 2
#     A. Redfern    2018-09-18    Minor formatting changes to align with
#                                 existing code examples
#
################################################################################


################################################################################
#
# Import
#
################################################################################

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


################################################################################
#
# Constants
#
################################################################################

# number of classes
NUM_CLASSES = 10

# image side
IMAGE_ROWS = 28
IMAGE_COLS = 28

# image normalization
IMAGE_NORM = 255.0


################################################################################
#
# User Parameters
#
################################################################################

# input layer neurons
M0 = 100

# epochs (5 gives ~ 93 %, 20 gives ~ 96 %)
numEpochs = 5

# visualization
visualizeRows = 5
visualizeCols = 4


################################################################################
#
# Data
#
################################################################################

# display
print("\nData")

# download MNIST
mnistDataset = keras.datasets.mnist

# load training and testing images
(trainImages, trainLabels), (testImages, testLabels) = mnistDataset.load_data()

# split training images to training and validation images
trainImages, validationImages, trainLabels, validationLabels = train_test_split(trainImages, trainLabels, shuffle=True, test_size=0.1)

# pre processing (normalization to [0, 1])
trainImages      = trainImages/IMAGE_NORM
testImages       = testImages/IMAGE_NORM
validationImages = validationImages/IMAGE_NORM

# number of test labels
numTestLabels = testLabels.size


################################################################################
#
# Network
#
################################################################################

# display
print("Network")

# network (vectorization, fully connected layer, fully connected layer)
layerPreprocess = keras.layers.Flatten(input_shape=(IMAGE_ROWS, IMAGE_COLS))
layer0          = keras.layers.Dense(M0, use_bias=True, activation='relu')
layer1          = keras.layers.Dense(NUM_CLASSES, use_bias=True, activation='softmax')
model           = keras.models.Sequential([layerPreprocess, layer0, layer1])

# compile (sparse_categorical_crossentropy loss, SGD optimizer)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')


################################################################################
#
# Training
#
################################################################################

# display
print("Training\n")

# training
model.fit(trainImages, trainLabels, epochs=numEpochs, validation_data=(validationImages, validationLabels))


################################################################################
#
# Testing
#
################################################################################

# testing
testLoss, testAccuracy = model.evaluate(testImages, testLabels)

# display
print("\nTesting accuracy: {}".format(testAccuracy))


################################################################################
#
# Visualization
#
################################################################################

# display
print("Visualization\n")

# predictions
preds = model.predict(testImages)
pos   = 0

# create a plot of incorrect predictions
plt.figure(figsize=(5, 3))

# cycle through the images
for i in range(numTestLabels):

    # add to plot if incorrect prediction
    predLabel = np.argmax(preds[i])
    if testLabels[i] != predLabel:
        pos = pos + 1
        plt.subplot(visualizeRows, visualizeCols, pos)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(testImages[i])
        plt.xlabel("True: {}, Predicted: {}".format(testLabels[i], predLabel))

    # stop plotting if a sufficient number of incorrectly labeled examples are found
    if pos == (visualizeRows*visualizeCols):
        break

# show plot
plt.show()

