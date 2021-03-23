################################################################################
#
# LOGISTICS
#
#    <TO DO: first and last name as in eLearning>
#    <TO DO: UTD ID>
#    <TO DO: this comment block is included in each file that is submitted:
#    eff.py, eff_se.py (if done) and eff_se_cond.py (if done)>
#
# FILE
#
#    eff.py | eff_se.py | eff_se_cond.py
#
# DESCRIPTION
#
#    Grade = eff.py grade (max 90) + eff_se.py grade (max 10) + eff_se_cond.py grade (max 10)
#
#    A PyTorch implementation of the network described in section 3 of
#    xNNs_Project_002_NetworksPaper.doc/pdf trained in Google Colaboratory using
#    a GPU instance (Runtime - Change runtime type - Hardware accelerator - GPU)
#
# INSTRUCTIONS
#
#    1. a. eff.py
#
#          Complete all <TO DO: ...> code portions of this file to design and
#          train the network in table 1 of the paper with the standard inverted
#          residual building block in fig 2a and report the results
#
#    1. b. eff_se.py
#
#          Starting from eff.py, create a SE block per fig 3 in the paper and
#          add it to the inverted residual block per fig 2b in the paper to
#          create a SE enhanced building block; design and train the network
#          in table 1 with the SE enhanced building block and report the same
#          results as in the standard building block case
#
#    1. c. eff_se_cond.py
#
#          Starting from eff_se.py, create a conditional conv operation per
#          fig 4 in the paper and replace the building block convolutions with
#          the conditional convolution operation to create a SE and conditional
#          convolution enhanced building block per fig 2c in the paper; design
#          and train the network in table 1 with the SE and conditional
#          convolution enhanced building block and report the same results as in
#          the standard building block case; it may be required to reduce the
#          batch size in this case if there is an out of memory error
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics (for all networks trained: standard, SE enhanced and
#       SE and conditional convolution enhanced)
#
#       <TO DO: cut and paste per epoch statistics here>
#
#    3. Submit eff.py, eff_se.py (if done) and eff_se_cond.py (if done) via
#       eLearning (no zip files, no Jupyter / iPython notebooks, ...) with this
#       comment block at the top and all code from the IMPORT comment block to
#       the end; so if you implement all 3, you will submit 3 Python files
#
# HELP
#
#    1. If you're looking for a reference for block and network design, see
#       https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/blob/master/Tests/202008/xNNs_Project_002_Networks.py
#       which implemented a RegNetX style block and network; while the block and
#       network is different, that code should help with thinking about how to
#       organize this code
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
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
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 256
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model
# <TO DO: your code goes here>

# training
TRAIN_LR_MAX              = 0.2
TRAIN_LR_INIT_SCALE       = 0.01
TRAIN_LR_FINAL_SCALE      = 0.001
TRAIN_LR_INIT_EPOCHS      = 5
TRAIN_LR_FINAL_EPOCHS     = 50 # 100
TRAIN_NUM_EPOCHS          = TRAIN_LR_INIT_EPOCHS + TRAIN_LR_FINAL_EPOCHS
TRAIN_LR_INIT             = TRAIN_LR_MAX*TRAIN_LR_INIT_SCALE
TRAIN_LR_FINAL            = TRAIN_LR_MAX*TRAIN_LR_FINAL_SCALE
TRAIN_INTRA_EPOCH_DISPLAY = 10000

# file
FILE_NAME_CHECK      = 'EffNetStyleCheck.pt'
FILE_NAME_BEST       = 'EffNetStyleBest.pt'
FILE_SAVE            = True
FILE_LOAD            = False
FILE_EXTEND_TRAINING = False
FILE_NEW_OPTIMIZER   = False

################################################################################
#
# DATA
#
################################################################################

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False)

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# inverted residual block
class InvResBlock(nn.Module):

    # initialization
    def __init__(self, Ni, Ne, No, F, S):

        # parent initialization
        super(InvResBlock, self).__init__()

        # create all of the operators for the inverted residual block in fig 2a
        # of the paper; note that parameter names were chosen to match the paper
        # <TO DO: your code goes here>

    # forward path
    def forward(self, x):

        # map input x to output y for the inverted residual block in fig 2a of
        # the paper via connecting the operators defined in the initialization
        # and return output y
        # <TO DO: your code goes here>

        # return
        return y

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    # add necessary parameters to the init function to create the model defined
    # in table 1 of the paper
    def __init__(self, ): # <TO DO: your code goes here> inside the parenthesis

        # parent initialization
        super(Model, self).__init__()

        # create all of the operators for the network defined in table 1 of the
        # paper using a combination of Python, standard PyTorch operators and
        # the previously defined InvResBlock class
        # <TO DO: your code goes here>

    # forward path
    def forward(self, x):

        # map input x to output y for the network defined in table 1 of the
        # paper via connecting the operators defined in the initialization
        # and return output y
        # <TO DO: your code goes here>

        # return
        return y

# create
# add necessary parameters to the init function to create the model defined
# in table 1 of the paper
model = Model() # <TO DO: your code goes here> inside the parenthesis

# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)

################################################################################
#
# ERROR AND OPTIMIZER
#
################################################################################

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup followed by 1/2 wave cosine decay
    if epoch < TRAIN_LR_INIT_EPOCHS:
        lr = (TRAIN_LR_MAX - TRAIN_LR_INIT)*(float(epoch)/TRAIN_LR_INIT_EPOCHS) + TRAIN_LR_INIT
    else:
        lr = TRAIN_LR_FINAL + 0.5*(TRAIN_LR_MAX - TRAIN_LR_FINAL)*(1.0 + math.cos(((float(epoch) - TRAIN_LR_INIT_EPOCHS)/(TRAIN_LR_FINAL_EPOCHS - 1.0))*math.pi))

    return lr

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0.0, weight_decay=5e-5, nesterov=True)

################################################################################
#
# TRAINING
#
################################################################################

# start epoch
start_epoch = 0

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transfer the network to the device
model.to(device)

# load the last checkpoint
if (FILE_LOAD == True):
    checkpoint = torch.load(FILE_NAME_CHECK)
    model.load_state_dict(checkpoint['model_state_dict'])
    if (FILE_NEW_OPTIMIZER == False):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if (FILE_EXTEND_TRAINING == False):
        start_epoch = checkpoint['epoch'] + 1

# initialize the epoch
accuracy_best      = 0
start_time_display = time.time()
start_time_epoch   = time.time()

# cycle through the epochs
for epoch in range(start_epoch, TRAIN_NUM_EPOCHS):

    # initialize epoch training
    model.train()
    training_loss = 0.0
    num_batches   = 0
    num_display   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the training data set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1
        num_display   = num_display + DATA_BATCH_SIZE

        # display intra epoch results
        if (num_display > TRAIN_INTRA_EPOCH_DISPLAY):
            num_display          = 0
            elapsed_time_display = time.time() - start_time_display
            start_time_display   = time.time()
            print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f}'.format(epoch, elapsed_time_display, lr_schedule(epoch), (training_loss / num_batches) / DATA_BATCH_SIZE), flush=True)

    # initialize epoch testing
    model.eval()
    test_correct = 0
    test_total   = 0

    # no weight update / no gradient needed
    with torch.no_grad():

        # cycle through the testing data set
        for data in dataloader_test:

            # extract a batch of data and move it to the appropriate device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass and prediction
            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # update test set statistics
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()

    # epoch statistics
    elapsed_time_epoch = time.time() - start_time_epoch
    start_time_epoch   = time.time()
    print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f} accuracy = {4:5.2f}'.format(epoch, elapsed_time_epoch, lr_schedule(epoch), (training_loss/num_batches)/DATA_BATCH_SIZE, (100.0*test_correct/test_total)), flush=True)

    # save a checkpoint
    if (FILE_SAVE == True):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, FILE_NAME_CHECK)

    # save the best model
    accuracy_epoch = 100.0 * test_correct / test_total
    if ((FILE_SAVE == True) and (accuracy_epoch >= accuracy_best)):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, FILE_NAME_BEST)
