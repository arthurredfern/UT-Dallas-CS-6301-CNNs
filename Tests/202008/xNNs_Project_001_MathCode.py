################################################################################
#
# FILE
#
#    xNNs_Project_001_MathCode.py
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. This does not use PyTorch, TensorFlow or any other xNN library
#
#    2. Modifications
#
#       - Set a matrix dim to 100 (from 1000) for speed purposes during dev
#       - Add better data loading; maybe shuffling, batching, augmentation, ...
#       - Re write with indicated classes to bring in line with xNN libraries
#
#    3. Model
#
#       Layer                   Output
#       ---------------------   -----------
#       Data                    1x28x28
#       Division by 255.0       1x28x28
#       Vectorization           1x784
#       ---
#       Matrix multiplication   1x1000 (reduced to 100 for dev speed)
#       Addition                1x1000 (reduced to 100 for dev speed)
#       ReLU                    1x1000 (reduced to 100 for dev speed)
#       ---
#       Matrix multiplication   1x100
#       Addition                1x100
#       ReLU                    1x100
#       ---
#       Matrix multiplication   1x10
#       Addition                1x10
#       Softmax                 1x10
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
MODEL_N1 = 100 # 1000
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

# data class
    # init
    # shuffle, batch, augment, ...
    # load

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
# LAYERS
#
################################################################################

# layer class (for each different layer)
    # init
    # forward function
    # backward function

# loss class (include here as it's effectively a layer)
    # init
    # forward pass
    # backward pass

################################################################################
#
# NETWORK
#
################################################################################

# network class
    # init
    # forward pass
    # backward pass

# initialize data
x0 = np.zeros(MODEL_N0, dtype=np.float32)

# initialize weights
w1 = np.sqrt(2.0/(MODEL_N0 + MODEL_N1), dtype=np.float32)*np.random.standard_normal((MODEL_N0, MODEL_N1)).astype(np.float32)
w2 = np.zeros(MODEL_N1, dtype=np.float32)
w4 = np.sqrt(2.0/(MODEL_N1 + MODEL_N2), dtype=np.float32)*np.random.standard_normal((MODEL_N1, MODEL_N2)).astype(np.float32)
w5 = np.zeros(MODEL_N2, dtype=np.float32)
w7 = np.sqrt(2.0/(MODEL_N2 + MODEL_N3), dtype=np.float32)*np.random.standard_normal((MODEL_N2, MODEL_N3)).astype(np.float32)
w8 = np.zeros(MODEL_N3, dtype=np.float32)

# initialize feature maps
x1 = np.zeros(MODEL_N1, dtype=np.float32)
x2 = np.zeros(MODEL_N1, dtype=np.float32)
x3 = np.zeros(MODEL_N1, dtype=np.float32)
x4 = np.zeros(MODEL_N2, dtype=np.float32)
x5 = np.zeros(MODEL_N2, dtype=np.float32)
x6 = np.zeros(MODEL_N2, dtype=np.float32)
x7 = np.zeros(MODEL_N3, dtype=np.float32)
x8 = np.zeros(MODEL_N3, dtype=np.float32)

# initialize pmf
x9 = np.zeros(MODEL_N3, dtype=np.float32)

# initialize feature map gradients
dedx1 = np.zeros(MODEL_N1, dtype=np.float32)
dedx2 = np.zeros(MODEL_N1, dtype=np.float32)
dedx3 = np.zeros(MODEL_N1, dtype=np.float32)
dedx4 = np.zeros(MODEL_N2, dtype=np.float32)
dedx5 = np.zeros(MODEL_N2, dtype=np.float32)
dedx6 = np.zeros(MODEL_N2, dtype=np.float32)
dedx7 = np.zeros(MODEL_N3, dtype=np.float32)
dedx8 = np.zeros(MODEL_N3, dtype=np.float32)

# initialize weight gradients
dedw1 = np.zeros((MODEL_N0, MODEL_N1), dtype=np.float32)
dedw2 = np.zeros(MODEL_N1, dtype=np.float32)
dedw4 = np.zeros((MODEL_N1, MODEL_N2), dtype=np.float32)
dedw5 = np.zeros(MODEL_N2, dtype=np.float32)
dedw7 = np.zeros((MODEL_N2, MODEL_N3), dtype=np.float32)
dedw8 = np.zeros(MODEL_N3, dtype=np.float32)

################################################################################
#
# LOSS AND OPTIMIZER
#
################################################################################

# optimizer class
    # init
    # weight update

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
    for t in range(train_data.shape[0]):

        # data
        x0    = np.reshape(train_data[t, :, :, :], MODEL_N0)/DATA_NORM
        label = train_labels[t]

        # model
        x1 = x0.dot(w1)
        x2 = x1 + w2
        x3 = np.maximum(np.float32(0.0), x2)
        x4 = x3.dot(w4)
        x5 = x4 + w5
        x6 = np.maximum(np.float32(0.0), x5)
        x7 = x6.dot(w7)
        x8 = x7 + w8

        # loss
        x9   = np.exp(x8)
        x9   = x9/np.sum(x9)
        loss = -np.log(x9[label])

        # backward - loss
        dedx8        = np.copy(x9)
        dedx8[label] = dedx8[label] - np.float32(1.0)

        # backward - model feature maps
        dedx7        = np.copy(dedx8)
        dedx6        = dedx7.dot(w7.T)
        dedx5        = np.minimum(np.float32(1.0), np.ceil(np.maximum(np.float32(0.0), x5)))*dedx6
        dedx4        = np.copy(dedx5)
        dedx3        = dedx4.dot(w4.T)
        dedx2        = np.minimum(np.float32(1.0), np.ceil(np.maximum(np.float32(0.0), x2)))*dedx3
        dedx1        = np.copy(dedx2)

        # backward - model weights
        dedw8        = np.copy(dedx8)
        dedw7        = np.outer(x6, dedx7)
        dedw5        = np.copy(dedx5)
        dedw4        = np.outer(x3, dedx4)
        dedw2        = np.copy(dedx2)
        dedw1        = np.outer(x0, dedx1)

        # optimizer step
        w1 = w1 - lr*dedw1
        w2 = w2 - lr*dedw2
        w4 = w4 - lr*dedw4
        w5 = w5 - lr*dedw5
        w7 = w7 - lr*dedw7
        w8 = w8 - lr*dedw8

        # update statistics
        training_loss = training_loss + loss

    # cycle through the testing data
    for t in range(test_data.shape[0]):

        # data
        x0    = np.reshape(test_data[t, :, :, :], MODEL_N0)/DATA_NORM
        label = test_labels[t]

        # model
        x1 = x0.dot(w1)
        x2 = x1 + w2
        x3 = np.maximum(np.float32(0.0), x2)
        x4 = x3.dot(w4)
        x5 = x4 + w5
        x6 = np.maximum(np.float32(0.0), x5)
        x7 = x6.dot(w7)
        x8 = x7 + w8

        # prediction
        prediction = (np.argmax(x8)).astype(np.int32)

        # update statistics
        if (label == prediction):
            testing_correct = testing_correct + 1
            
    # epoch statistics
    elapsed_time_epoch = time.time() - start_time_epoch
    start_time_epoch   = time.time()
    epochs[epoch]      = epoch
    avg_loss[epoch]    = training_loss/train_data.shape[0]
    accuracy[epoch]    = 100.0*testing_correct/test_data.shape[0]
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
    x0 = np.reshape(test_data[t, :, :, :], MODEL_N0)/DATA_NORM

    # model
    x1 = x0.dot(w1)
    x2 = x1 + w2
    x3 = np.maximum(np.float32(0.0), x2)
    x4 = x3.dot(w4)
    x5 = x4 + w5
    x6 = np.maximum(np.float32(0.0), x5)
    x7 = x6.dot(w7)
    x8 = x7 + w8

    # prediction
    predictions[t] = (np.argmax(x8)).astype(np.int32)

# plot of display examples
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for t in range(DISPLAY_NUM):
    img = test_data[t, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, t + 1))
    ax[-1].set_title('True: ' + str(test_labels[t]) + ' xNN: ' + str(predictions[t]))
    plt.imshow(img, cmap='Greys')

# show figures
plt.show()
