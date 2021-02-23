################################################################################
#
# LOGISTICS
#
#    <TO DO: first and last name as in eLearning>
#    <TO DO: UTD ID>
#    <TO DO: this comment block is only included in the file nn.py>
#
# FILE
#
#    nn.py
#
# DESCRIPTION
#
#    Grade = nn.py grade (max 80) + sw.py grade (max 20) + cnn.py grade (max 20)
#
#    This file is required; see above for grade calculation
#
#    This is the start of an exceedingly simple / lite / reduced functionality
#    PyTorch style xNN library written in Python and it's example use for MNIST
#    image classification
#
#    This code does not use PyTorch, TensorFlow or any other xNN library
#
#    NN specification:
#
#       ----------------------------   -------
#       Data loader                    Output
#       ----------------------------   -------
#       Data                           1x28x28
#       Division by 255.0              1x28x28
#
#       ----------------------------   -------
#       Network                        Output
#       ----------------------------   -------
#       Vectorization                  1x784
#       Vector matrix multiplication   1x100
#       Vector vector addition         1x100
#       ReLU                           1x100
#       Vector matrix multiplication   1x100
#       Vector vector addition         1x100
#       ReLU                           1x100
#       Vector matrix multiplication   1x10
#       Vector vector addition         1x10
#
#       ----------------------------   -------
#       Error                          Output
#       ----------------------------   -------
#       Softmax                        1x10
#       Cross entropy                  1
#
# INSTRUCTIONS
#
#    1. Complete all <TO DO: ...> code portions of this file
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics
#
#       <TO DO: cut and paste per epoch statistics here>
#
#    3. Submit nn.py via eLearning (no zip files, no Jupyter / iPython
#       notebooks, ...) with this comment block at the top and all code from
#       the IMPORT comment block to the end
#
################################################################################

################################################################################
#
# LOGISTICS
#
#    <TO DO: first and last name as in eLearning>
#    <TO DO: UTD ID>
#    <TO DO: this comment block is only included in the file sw.py>
#
# FILE
#
#    sw.py
#
# DESCRIPTION
#
#    Grade = nn.py grade (max 80) + sw.py grade (max 20) + cnn.py grade (max 20)
#
#    This file is optional; see above for grade calculation
#
#    This builds on nn.py and adds various software enhancements that are
#    commonly provided in xNN libraries
#
#    This code does not use PyTorch, TensorFlow or any other xNN library
#
#    Software enhancements (not all are required):
#
#    A. Enable arbitrary data pre processing operations via passing in the name
#       of a data pre processing function to the DataLoader.get() function or
#       via DataLoader initialization and implement some such as mean and
#       variance normalization, batching, shuffling, left right flip (but don't
#       use this 1 for MNIST), color / intensity modification, ...
#
#    B. Automate the creation of the backwards graph from the forwards graph in
#       the Network.backwards() function for:
#
#       - Simple sequential network structures
#
#       - More complex network structures with branching and or conditionals
#         which will require tracking 1 or more layer inputs and outputs and
#         their associated connections; as an example of branching, think of
#         supporting network structures such as y = x + f(x) where f can be
#         multiple layers itself
#
#    C. Automate the updating of parameters from the network variables in the
#       Network.update() function via automatically cycling through all layers
#       in the network and taking advantage of a common trainable parameter
#       naming (e.g., all trainable parameters in all layers are called h)
#
#    D. Enable arbitrary parameter update functions (beyond simple sgd) in the
#       Network.update() function via passing in the name of the function used
#       for the update function or via setting it in the network initialization
#
#    E. Enable separate learning rates for each layer by including a learning
#       rate instance variable with each layer and using that for the update;
#       set this parameter for all layers after each call to the learning rate
#       schedule function
#
#    F. Add train and evaluation modes to the layers that reduce storage in the
#       case of evaluation mode; appropriately set the modes during training
#       and testing to improve performance
#
#    G. Add support for different types of parameter initialization for each
#       layer; ideally, enable a function to be specified for each layer during
#       the layer initialization
#
#    H. Add support for batch processing and demonstrate it's use in training;
#       note how accuracy and performance change for a batch size of 1, 32 and
#       128 (note that the learning rate may need to be modified for maximizing
#       the accuracy for different batch sizes); include results from different
#       batch sizes if this software enhancement is implemented
#
#    I. Add support for momentum and L2 regularization in the weight update and
#       demonstrate their use in training
#
#    J. Impress me
#
# INSTRUCTIONS
#
#    1. Starting from nn.py add some or all of the above software enhancements;
#       indicate the software enhancements that were implemented via letter:
#
#       <TO DO: letters of implemented software enhancements>
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics
#
#       <TO DO: cut and paste per epoch statistics here>
#
#    3. Submit sw.py via eLearning (no zip files, no Jupyter / iPython
#       notebooks, ...) with this comment block at the top and all code from
#       the IMPORT comment block to the end
#
################################################################################

################################################################################
#
# LOGISTICS
#
#    <TO DO: first and last name as in eLearning>
#    <TO DO: UTD ID>
#    <TO DO: this comment block is only included in the file cnn.py>
#
# FILE
#
#    cnn.py
#
# DESCRIPTION
#
#    Grade = nn.py grade (max 80) + sw.py grade (max 20) + cnn.py grade (max 20)
#
#    This file is optional; see above for grade calculation
#
#    This builds on nn.py and adds various layers that are commonly provided in
#    xNN libraries for the design and training of CNNs
#
#    This code does not use PyTorch, TensorFlow or any other xNN library
#
#    CNN specification:
#
#       ----------------------------   -------
#       Data loader                    Output
#       ----------------------------   -------
#       Data                           1x28x28
#       Division by 255.0              1x28x28
#
#       ----------------------------   -------
#       Network                        Output
#       ----------------------------   -------
#       Zero pad                       1x30x30
#       3x3/1 CNN style 2D conv        8x28x28
#       3D tensor 3D tensor addition   8x28x28
#       ReLU                           8x28x28
#       2x2/2 avg pool                 8x14x14
#       Zero pad                       8x16x16
#       3x3/1 CNN style 2D conv        16x14x14
#       3D tensor 3D tensor addition   16x14x14
#       ReLU                           16x14x14
#       2x2/2 avg pool                 16x7x7
#       Zero pad                       16x9x9
#       3x3/1 CNN style 2D conv        32x7x7
#       3D tensor 3D tensor addition   32x7x7
#       ReLU                           32x7x7
#       Vectorization                  1x1568
#       Vector matrix multiplication   1x100
#       Vector vector addition         1x100
#       ReLU                           1x100
#       Vector matrix multiplication   1x10
#       Vector vector addition         1x10
#
#       ----------------------------   -------
#       Error                          Output
#       ----------------------------   -------
#       Softmax                        1x10
#       Cross entropy                  1
#
# INSTRUCTIONS
#
#    1. Starting from nn.py add the following layers
#
#       - Zero pad
#       - CNN style 2D convolution
#       - 3D tensor 3D tensor addition (where all elements in a feature map in
#         the 2nd 3D tensor have the same value as would be expected in a full
#         CNN style 2D convolution layer)
#       - Avg pool
#
#       and implement the above specified CNN
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics
#
#       <TO DO: cut and paste per epoch statistics here>
#
#    3. Submit cnn.py via eLearning (no zip files, no Jupyter / iPython
#       notebooks, ...) with this comment block at the top and all code from
#       the IMPORT to the end
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

import os.path
import urllib.request
import gzip
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_NORM              = np.float32(255.0)
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# model
MODEL_N0 = DATA_ROWS*DATA_COLS
MODEL_N1 = 100
MODEL_N2 = 100
MODEL_N3 = DATA_CLASSES

# training
TRAIN_LR_MAX          = 0.01
TRAIN_LR_INIT_SCALE   = 0.01
TRAIN_LR_FINAL_SCALE  = 0.001
TRAIN_LR_INIT_EPOCHS  = 3
TRAIN_LR_FINAL_EPOCHS = 10
TRAIN_NUM_EPOCHS      = TRAIN_LR_INIT_EPOCHS + TRAIN_LR_FINAL_EPOCHS
TRAIN_LR_INIT         = TRAIN_LR_MAX*TRAIN_LR_INIT_SCALE
TRAIN_LR_FINAL        = TRAIN_LR_MAX*TRAIN_LR_FINAL_SCALE

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

################################################################################
#
# DATA LOADER
#
################################################################################

# data loader class
class DataLoader:

    # save images x, labels y and data normalization factor x_norm
    def __init__(self, x, y, x_norm):
        # <TO DO: your code goes here>

    # return normalized image t and label t
    def get(self, t):
        # <TO DO: your code goes here>

    # return the total number of images
    def num(self):
        # <TO DO: your code goes here>

# data loaders
data_loader_train = DataLoader(train_data, train_labels, DATA_NORM)
data_loader_test  = DataLoader(test_data,  test_labels,  DATA_NORM)

################################################################################
#
# LAYERS
#
################################################################################

# vector matrix multiplication layer
class VectorMatrixMultiplication:

    # initialize input x, parameters h and parameter gradient de/dh
    def __init__(self, x_channels, y_channels):
        # <TO DO: your code goes here>

    # save the input x and return y = f(x, h)
    def forward(self, x):
        # <TO DO: your code goes here>

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        # <TO DO: your code goes here>

# vector vector addition layer
class VectorVectorAddition:

    # initialize parameters h and parameter gradient de/dh
    def __init__(self, x_channels):
        # <TO DO: your code goes here>
    
    # return y = f(x, h)
    def forward(self, x):
        # <TO DO: your code goes here>

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        # <TO DO: your code goes here>

# ReLU layer
class ReLU:

    # initialize input x
    def __init__(self, x_channels):
        # <TO DO: your code goes here>
    
    # save the input x and return y = f(x, h)
    def forward(self, x):
        # <TO DO: your code goes here>

    # return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        # <TO DO: your code goes here>

# soft max cross entropy layer
class SoftMaxCrossEntropy:

    # initialize probability p and label
    def __init__(self, y_channels):
        # <TO DO: your code goes here>
    
    # save the label, compute and save the probability p and return e = f(label, y)
    def forward(self, label, y):
        # <TO DO: your code goes here>

    # compute and return the input gradient de/dx from the saved probability and label; e is not used
    def backward(self, e):
        # <TO DO: your code goes here>

################################################################################
#
# NETWORK
#
################################################################################

# network
class Network:

    # save the network description parameters and create all layers
    def __init__(self, rows, cols, n0, n1, n2, n3):
        # <TO DO: your code goes here>

    # connect layers forward functions together to map the input image to the network output
    # return the network output
    def forward(self, img):
        # <TO DO: your code goes here>

    # connect layers backward functions together to map de/dy at the end of the network to de/dx at the beginning
    # note that inside the backward functions de/dh is computed for all parameters
    # optionally return de/dx (unused)
    def backward(self, dedy):
        # <TO DO: your code goes here>

    # update all layers with trainable parameters via h = h - lr * de/dh
    def update(self, lr):
        # <TO DO: your code goes here>

# network
network = Network(DATA_ROWS, DATA_COLS, MODEL_N0, MODEL_N1, MODEL_N2, MODEL_N3)

################################################################################
#
# ERROR
#
################################################################################

# error
error = SoftMaxCrossEntropy(MODEL_N3)

################################################################################
#
# UPDATE
#
################################################################################

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup
    if epoch < TRAIN_LR_INIT_EPOCHS:
        lr = (TRAIN_LR_MAX - TRAIN_LR_INIT)*(float(epoch)/TRAIN_LR_INIT_EPOCHS) + TRAIN_LR_INIT
    # 1/2 wave cosine decay
    else:
        lr = TRAIN_LR_FINAL + 0.5*(TRAIN_LR_MAX - TRAIN_LR_FINAL)*(1.0 + math.cos(((float(epoch) - TRAIN_LR_INIT_EPOCHS)/(TRAIN_LR_FINAL_EPOCHS - 1.0))*math.pi))

    return lr

################################################################################
#
# TRAIN
#
################################################################################

# initialize the epoch
start_epoch      = 0
start_time_epoch = time.time()

# initialize the display statistics
epochs   = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.int32)
avg_loss = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.float32)
accuracy = np.zeros(TRAIN_NUM_EPOCHS, dtype=np.float32)

# cycle through the epochs
for epoch in range(start_epoch, TRAIN_NUM_EPOCHS):

    # set the learning rate
    lr = np.float32(lr_schedule(epoch))

    # initialize the epoch statistics
    training_loss   = 0.0
    testing_correct = 0

    # cycle through the training data
    for t in range(data_loader_train.num()):

        # data
        img, label = data_loader_train.get(t)

        # network forward pass, error forward pass, error backward pass and network backward pass
        y    = network.forward(img)
        e    = error.forward(label, y)
        dedy = error.backward(e)
        dedx = network.backward(dedy)

        # weight update
        network.update(lr)

        # update statistics
        training_loss = training_loss + e

    # cycle through the testing data
    for t in range(data_loader_test.num()):

        # data
        img, label = data_loader_test.get(t)

        # network forward pass and prediction
        y          = network.forward(img)
        prediction = (np.argmax(y)).astype(np.int32)

        # update statistics
        if (label == prediction):
            testing_correct = testing_correct + 1
            
    # epoch statistics
    elapsed_time_epoch = time.time() - start_time_epoch
    start_time_epoch   = time.time()
    epochs[epoch]      = epoch
    avg_loss[epoch]    = training_loss/data_loader_train.num()
    accuracy[epoch]    = 100.0*testing_correct/data_loader_test.num()
    print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f} accuracy = {4:5.2f}'.format(epoch, elapsed_time_epoch, lr, avg_loss[epoch], accuracy[epoch]), flush=True)

################################################################################
#
# DISPLAY
#
################################################################################

# plot of loss and accuracy vs epoch
fig1, ax1 = plt.subplots()
ax1.plot(epochs, avg_loss, color='red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Avg loss', color='red')
ax1.set_title('Avg Loss And Accuracy Vs Epoch')
ax2 = ax1.twinx()
ax2.plot(epochs, accuracy, color='blue')
ax2.set_ylabel('Accuracy %', color='blue')

# initialize the display predictions
predictions = np.zeros(DISPLAY_NUM, dtype=np.int32)

# cycle through the display data
for t in range(DISPLAY_NUM):

    # data
    img, label = data_loader_test.get(t)

    # network forward pass and prediction
    y              = network.forward(img)
    predictions[t] = (np.argmax(y)).astype(np.int32)

# plot of display examples
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for t in range(DISPLAY_NUM):
    img, label = data_loader_test.get(t)
    img        = img.reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, t + 1))
    ax[-1].set_title('True: ' + str(label) + ' xNN: ' + str(predictions[t]))
    plt.imshow(img, cmap='Greys')

# show figures
plt.show()
