################################################################################
#
# LOGISTICS
#
#    Arthur Redfern
#
# FILE
#
#    xNNs_Project_001_nn.py
#
# DESCRIPTION
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
DATA_URL_TRAIN_DATA    = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Tests/202101/train_data.gz'
DATA_URL_TRAIN_LABELS  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Tests/202101/train_labels.gz'
DATA_URL_TEST_DATA     = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Tests/202101/test_data.gz'
DATA_URL_TEST_LABELS   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Tests/202101/test_labels.gz'
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
        self.x      = x
        self.y      = y
        self.x_norm = x_norm

    # return normalized image t and label t
    def get(self, t):
        return self.x[t, :, :, :]/self.x_norm, self.y[t]

    # return the total number of images
    def num(self):
        return self.x.shape[0]

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
        self.x    = np.zeros(x_channels, dtype=np.float32)
        self.h    = np.sqrt(2.0/(x_channels + y_channels), dtype=np.float32)*np.random.standard_normal((x_channels, y_channels)).astype(np.float32)
        self.dedh = np.zeros((x_channels, y_channels), dtype=np.float32)

    # save the input x and return y = f(x, h)
    def forward(self, x):
        self.x = x
        return x.dot(self.h)

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        self.dedh = np.outer(self.x, dedy)
        return dedy.dot(self.h.T)

# vector vector addition layer
class VectorVectorAddition:

    # initialize parameters h and parameter gradient de/dh
    def __init__(self, x_channels):
        self.h    = np.zeros(x_channels, dtype=np.float32)
        self.dedh = np.zeros(x_channels, dtype=np.float32)
    
    # return y = f(x, h)
    def forward(self, x):
        return x + self.h

    # compute and save the parameter gradient de/dh and return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        self.dedh = np.copy(dedy)
        return np.copy(dedy)

# ReLU layer
class ReLU:

    # initialize input x
    def __init__(self, x_channels):
        self.x = np.zeros(x_channels, dtype=np.float32)
    
    # save the input x and return y = f(x, h)
    def forward(self, x):
        self.x = x
        return np.maximum(np.float32(0.0), x)

    # return the input gradient de/dx = de/dy * dy/dx
    def backward(self, dedy):
        return np.minimum(np.float32(1.0), np.ceil(np.maximum(np.float32(0.0), self.x)))*dedy

# soft max cross entropy layer
class SoftMaxCrossEntropy:

    # initialize probability p and label
    def __init__(self, y_channels):
        self.p     = np.zeros(y_channels, dtype=np.float32)
        self.label = 0
    
    # save the label, compute and save the probability p and return e = f(label, y)
    def forward(self, label, y):
        self.p     = np.exp(y)
        self.p     = self.p/np.sum(self.p)
        self.label = label + 0
        return -np.log(self.p[self.label])

    # compute and return the input gradient de/dx from the saved probability and label; e is not used
    def backward(self, e):
        dedy             = np.copy(self.p)
        dedy[self.label] = dedy[self.label] - np.float32(1.0)
        return dedy

################################################################################
#
# NETWORK
#
################################################################################

# network
class Network:

    # save the network description parameters and create all layers
    def __init__(self, rows, cols, n0, n1, n2, n3):
        self.rows  = rows
        self.cols  = cols
        self.n0    = n0
        self.n1    = n1
        self.n2    = n2
        self.n3    = n3
        self.vmm1  = VectorMatrixMultiplication(n0, n1)
        self.vva1  = VectorVectorAddition(n1)
        self.relu1 = ReLU(n1)
        self.vmm2  = VectorMatrixMultiplication(n1, n2)
        self.vva2  = VectorVectorAddition(n2)
        self.relu2 = ReLU(n2)
        self.vmm3  = VectorMatrixMultiplication(n2, n3)
        self.vva3  = VectorVectorAddition(n3)

    # connect layers forward functions together to map the input image to the network output
    # return the network output
    def forward(self, img):
        x = np.reshape(img, self.n0)
        x = self.vmm1.forward(x)
        x = self.vva1.forward(x)
        x = self.relu1.forward(x)
        x = self.vmm2.forward(x)
        x = self.vva2.forward(x)
        x = self.relu2.forward(x)
        x = self.vmm3.forward(x)
        x = self.vva3.forward(x)
        return x

    # connect layers backward functions together to map de/dy at the end of the network to de/dx at the beginning
    # note that inside the backward functions de/dh is computed for all parameters
    # optionally return de/dx (unused)
    def backward(self, dedy):
        dedx = self.vva3.backward(dedy)
        dedx = self.vmm3.backward(dedx)
        dedx = self.relu2.backward(dedx)
        dedx = self.vva2.backward(dedx)
        dedx = self.vmm2.backward(dedx)
        dedx = self.relu1.backward(dedx)
        dedx = self.vva1.backward(dedx)
        dedx = self.vmm1.backward(dedx)
        dedx = dedx.reshape((1, 1, self.rows, self.cols))
        return dedx

    # update all layers with trainable parameters via h = h - lr * de/dh
    def update(self, lr):
        self.vva3.h = self.vva3.h - lr*self.vva3.dedh
        self.vmm3.h = self.vmm3.h - lr*self.vmm3.dedh
        self.vva2.h = self.vva2.h - lr*self.vva2.dedh
        self.vmm2.h = self.vmm2.h - lr*self.vmm2.dedh
        self.vva1.h = self.vva1.h - lr*self.vva1.dedh
        self.vmm1.h = self.vmm1.h - lr*self.vmm1.dedh

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
