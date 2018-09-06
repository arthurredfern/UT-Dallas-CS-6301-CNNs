################################################################################
#
# File
#
#    Code02xNNTraining.py
#
# Purpose
#
#    A toy example demonstrating training for a MNIST classifier including
#
#    Data preparation
#    Network specification
#    Initialization
#    Initial validation
#    Training iteration
#        Forward propagation
#        Error calculation
#        Backward propagation
#        Weight update
#        Training validation
#    Testing
#        Forward propagation
#        Accuracy calculation
#
# Notes
#
#    1.  This is a quickly written self contained toy network to do MNIST digit
#        classification with a number of qualifiers
#
#        Coding style is meant for clarity, but it's not fully beautiful or optimized
#        Better networks exist, little (~ 0) time was spent on the specific design
#        Better training methods exist, the choice was based on clarity
#        There are potentially bugs, if you find 1 send me a note
#
#    2.  Data is a set of 1 x 28 x 28 x numImages vectorized to 784 x numImages,
#        to duplicate this locally download MNIST and duplicate the formatting
#        as specified in the data preparation section
#
################################################################################


################################################################################
#
# History
#
#     A. Redfern    2018-09-03    Created
#
################################################################################


################################################################################
#
# Import
#
################################################################################

import numpy as np
import cPickle
import gzip


################################################################################
#
# Constants
#
################################################################################

# number of classes
NUM_CLASSES = 10


################################################################################
#
# User Parameters
#
################################################################################

# learning rate and amount to scale the learning rate by after every epoch
learningRate  = 0.1
learningScale = 0.5

# epochs
numEpochs = 5

# number of inputs to average the error over for tracking during training
errorFrequency = 10000


################################################################################
#
# Derived Parameters
#
################################################################################


################################################################################
#
# Data Preparation
#
################################################################################

# load data
dataTrain, dataValidate, dataTest = cPickle.load(gzip.open('mnist.pkl.gz', 'rb'))

# training data (x = data 784 x 50000, y = labels 50000 x 1)
xTrain = dataTrain[0].T
yTrain = dataTrain[1]

# validation data (x = data 784 x 10000, y = labels 10000 x 1)
xValidate = dataValidate[0].T
yValidate = dataValidate[1]

# testing data (x = data 784 x 10000, y = labels 10000 x 1)
xTest = dataTest[0].T
yTest = dataTest[1]

# data dimensions
(Ktrain,    Ntrain)    = xTrain.shape
(Kvalidate, Nvalidate) = xValidate.shape
(Ktest,     Ntest)     = xTest.shape


################################################################################
#
# Pre Processing
#
################################################################################

# 1 vector per input with values [0, 1] and a high level of sparsity
# will skip normalizing to 0 mean as that would destroy the sparsity


################################################################################
#
# Network Specification
#
################################################################################

# 0 matrix vector multiplication
K0 = Ktrain # 28*28 = 784
M0 = 100    # 10 features per class input to final linear layer

# 1 bias addition
K1 = M0
M1 = K1

# 2 ReLU
K2 = M1
M2 = K2

# 3 matrix vector multiplication
K3 = M2
M3 = NUM_CLASSES

# 4 bias addition
K4 = M3
M4 = K4


################################################################################
#
# Initialization
#
################################################################################

# weights Gaussian 0 mean 1 variance
H0 = np.random.randn(M0, K0)
h1 = np.random.randn(K1)
H3 = np.random.randn(M3, K3)
h4 = np.random.randn(K4)

# feature maps
x0 = np.zeros(K0)
x1 = np.zeros(K1)
x2 = np.zeros(K2)
x3 = np.zeros(K3)
x4 = np.zeros(K4)
x5 = np.zeros(M4)

# error calculations
px = np.zeros(M4)

# gradients
dedx1 = np.zeros(K1)
dedx2 = np.zeros(K2)
dedx3 = np.zeros(K3)
dedx4 = np.zeros(K4)
dedx5 = np.zeros(M4)


################################################################################
#
# Initial Validation
#
################################################################################

# initialize the error
eNumVal = 0

# cycle through the inputs
for n in range(Nvalidate):

    # forward propagation
    x0 = np.copy(xValidate[:, n])
    x1 = np.dot(H0, x0)
    x2 = np.add(x1, h1)
    x3 = np.maximum(0, x2)
    x4 = np.dot(H3, x3)
    x5 = np.add(x4, h4)

    # track errors
    khat = np.argmax(x5)
    if yValidate[n] != khat:
        eNumVal = eNumVal + 1

# normalize the error percentage
eNumVal = (100.0*eNumVal)/Nvalidate

# display results
print
print("Validation initial accuracy:        {0:6.2f}".format(100.0 - eNumVal))
print


################################################################################
#
# Training Iteration
#
################################################################################

# error reset
eIndex = 0
eNum   = 0
eSum   = 0.0

# cycle through the epochs
for epoch in range(numEpochs):
    
    #
    # training
    #

    # cycle through the inputs
    for n in range(Ntrain):

        # forward propagation
        x0 = np.copy(xTrain[:, n])
        x1 = np.dot(H0, x0)
        x2 = np.add(x1, h1)
        x3 = np.maximum(0, x2)
        x4 = np.dot(H3, x3)
        x5 = np.add(x4, h4)

        # error calculation
        kstar        = yTrain[n]
        px           = np.subtract(x5, np.max(x5))    # softmax
        px           = np.exp(px)                     # softmax
        px           = np.true_divide(px, np.sum(px)) # softmax
        e            = -np.log(px[kstar])             # cross entropy
        dedx5        = np.copy(px)                    # initial error gradient
        dedx5[kstar] = dedx5[kstar] - 1.0             # initial error gradient

        # error percentage (local training data)
        khat = np.argmax(x5)
        if kstar != khat:
            eNum = eNum + 1
            
        # error sum (local training data)
        eSum = eSum + e
        
        # error reporting
        eIndex = eIndex + 1
        if eIndex == errorFrequency:
            eNum = (100.0*eNum)/errorFrequency
            eSum = (1.0*eSum)/errorFrequency
            print("Training local accuracy and error:  {0:6.2f}   {1:6.2f}".format(100.0 - eNum, eSum))
            eIndex = 0
            eNum   = 0
            eSum   = 0.0
        
        # backward propagation
        dedx4 = np.copy(dedx5)
        dedx3 = np.dot(H3.T, dedx4)
        dedx2 = np.multiply(np.minimum(1.0, np.ceil(np.maximum(0.0, x2))), dedx3)
        dedx1 = np.copy(dedx2)
        
        # weight gradient
        dedH0 = np.outer(dedx1, x0)
        dedh1 = np.copy(dedx2)
        dedH3 = np.outer(dedx4, x3)
        dedh4 = np.copy(dedx5)

        # weight update
        H0 = np.subtract(H0, np.multiply(learningRate, dedH0))
        h1 = np.subtract(h1, np.multiply(learningRate, dedh1))
        H3 = np.subtract(H3, np.multiply(learningRate, dedH3))
        h4 = np.subtract(h4, np.multiply(learningRate, dedh4))

    #
    # validation
    #

    # initialize the error
    eNumVal = 0

    # cycle through the inputs
    for n in range(Nvalidate):

        # forward propagation
        x0 = np.copy(xValidate[:, n])
        x1 = np.dot(H0, x0)
        x2 = np.add(x1, h1)
        x3 = np.maximum(0, x2)
        x4 = np.dot(H3, x3)
        x5 = np.add(x4, h4)

        # track errors
        khat = np.argmax(x5)
        if yValidate[n] != khat:
            eNumVal = eNumVal + 1

    # normalize the error percentage
    eNumVal = (100.0*eNumVal)/Nvalidate

    # display results
    print
    print("Validation current accuracy:        {0:6.2f}".format(100.0 - eNumVal))
    print

    #
    # learning rate update
    #

    # update the learning rate
    learningRate = learningScale*learningRate


################################################################################
#
# Testing
#
################################################################################

# initialize the error
eNumTest = 0

# cycle through the inputs
for n in range(Ntest):

    # forward propagation
    x0 = np.copy(xTest[:, n])
    x1 = np.dot(H0, x0)
    x2 = np.add(x1, h1)
    x3 = np.maximum(0, x2)
    x4 = np.dot(H3, x3)
    x5 = np.add(x4, h4)

    # track errors
    khat = np.argmax(x5)
    if yTest[n] != khat:
        eNumTest = eNumTest + 1

# normalize the error percentage
eNumTest = (100.0*eNumTest)/Ntest

# display results
print("Testing accuracy:                   {0:6.2f}".format(100.0 - eNumTest))
print

