################################################################################
#
# LOGISTICS
#
#    Your name as in eLearning
#    Your UT Dallas identifier
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of 71.48% on epoch 103 with a learning
#    rate at that epoch of 0.000250 and time required for each epoch of ~ 190 s
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Change runtime type - Hardware accelerator - GPU
#    5. Runtime - Run all
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#       For reference, an implementation of this network took ~ 190 s per epoch
#       for training, validation and checkpoint saving on Oct 25, 2020 using a
#       free GPU runtime in Google Colab
#
#    3. Relative to the previous skeleton code, in dataloader_test I changed to
#       drop_last=False as there aren't many images per validation class and
#       without doing this the last 7 classes were skipped when the batch size
#       was set to 512
#
#    4. A training log is included at the bottom of this file showing training
#       set loss and test set accuracy per epoch
#
#    5. The choice of weight decay scale may be too high, in the future think
#       about reducing and re running the experiment; possibly also look into
#       increasing the max learning rate and shortening the number of epochs
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
DATA_BATCH_SIZE   = 512
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model
MODEL_LEVEL_1_BLOCKS   = 1 # used but ignored in model creation
MODEL_LEVEL_2_BLOCKS   = 1
MODEL_LEVEL_3_BLOCKS   = 1
MODEL_LEVEL_4_BLOCKS   = 4
MODEL_LEVEL_5_BLOCKS   = 7
MODEL_LEVEL_1_CHANNELS = 24
MODEL_LEVEL_2_CHANNELS = 24
MODEL_LEVEL_3_CHANNELS = 56
MODEL_LEVEL_4_CHANNELS = 152
MODEL_LEVEL_5_CHANNELS = 368
MODEL_LEVEL_1_GROUPS   = 1 # used but ignored in model creation
MODEL_LEVEL_2_GROUPS   = 3
MODEL_LEVEL_3_GROUPS   = 7
MODEL_LEVEL_4_GROUPS   = 19
MODEL_LEVEL_5_GROUPS   = 46

# training
TRAIN_LR_MAX              = 0.2
TRAIN_LR_INIT_SCALE       = 0.01
TRAIN_LR_FINAL_SCALE      = 0.001
TRAIN_LR_INIT_EPOCHS      = 5
TRAIN_LR_FINAL_EPOCHS     = 100
TRAIN_NUM_EPOCHS          = TRAIN_LR_INIT_EPOCHS + TRAIN_LR_FINAL_EPOCHS
TRAIN_LR_INIT             = TRAIN_LR_MAX*TRAIN_LR_INIT_SCALE
TRAIN_LR_FINAL            = TRAIN_LR_MAX*TRAIN_LR_FINAL_SCALE
TRAIN_INTRA_EPOCH_DISPLAY = 10000

# file
FILE_NAME_CHECK      = 'RegNetXCheck.pt'
FILE_NAME_BEST       = 'RegNetXBest.pt'
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
# dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
# dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False)

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, Ni, No, Fr, Fc, Sr, Sc, G):

        # parent initialization
        super(XBlock, self).__init__()

        # identity
        if ((Ni != No) or (Sr > 1) or (Sc > 1)):
            self.id    = True
            self.conv0 = nn.Conv2d(Ni, No, (1, 1), stride=(Sr, Sc), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
            self.bn0   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.id = False

        # residual
        Pr         = np.floor(Fr/2).astype(int)
        Pc         = np.floor(Fc/2).astype(int)
        self.conv1 = nn.Conv2d(Ni, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.bn1   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(No, No, (Fr, Fc), stride=(Sr, Sc), padding=(Pr, Pc), dilation=(1, 1), groups=G, bias=False, padding_mode='zeros')
        self.bn2   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(No, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.bn3   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # sum
        self.relu0 = nn.ReLU()

    # forward path
    def forward(self, x):

        # identity
        if (self.id == True):
            id = self.conv0(x)
            id = self.bn0(id)
        else:
            id = x

        # residual
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv3(res)
        res = self.bn3(res)

        # sum
        y = id + res
        y = self.relu0(y)

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
    def __init__(self,
                 data_num_channels,
                 model_level_1_blocks, model_level_1_channels, model_level_1_groups,
                 model_level_2_blocks, model_level_2_channels, model_level_2_groups,
                 model_level_3_blocks, model_level_3_channels, model_level_3_groups,
                 model_level_4_blocks, model_level_4_channels, model_level_4_groups,
                 model_level_5_blocks, model_level_5_channels, model_level_5_groups,
                 data_num_classes):

        # parent initialization
        super(Model, self).__init__()

        # stride
        stride1 = 1 # set to 2 for ImageNet
        stride2 = 1 # set to 2 for ImageNet
        stride3 = 2
        stride4 = 2
        stride5 = 2

        # encoder level 1
        self.enc_1 = nn.ModuleList()
        self.enc_1.append(nn.Conv2d(data_num_channels, model_level_1_channels, (3, 3), stride=(stride1, stride1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_1.append(nn.BatchNorm2d(model_level_1_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_1.append(nn.ReLU())

        # encoder level 2
        self.enc_2 = nn.ModuleList()
        self.enc_2.append(XBlock(model_level_1_channels, model_level_2_channels, 3, 3, stride2, stride2, model_level_2_groups))
        for n in range(model_level_2_blocks - 1):
            self.enc_2.append(XBlock(model_level_2_channels, model_level_2_channels, 3, 3, 1, 1, model_level_2_groups))

        # encoder level 3
        self.enc_3 = nn.ModuleList()
        self.enc_3.append(XBlock(model_level_2_channels, model_level_3_channels, 3, 3, stride3, stride3, model_level_3_groups))
        for n in range(model_level_3_blocks - 1):
            self.enc_3.append(XBlock(model_level_3_channels, model_level_3_channels, 3, 3, 1, 1, model_level_3_groups))

        # encoder level 4
        self.enc_4 = nn.ModuleList()
        self.enc_4.append(XBlock(model_level_3_channels, model_level_4_channels, 3, 3, stride4, stride4, model_level_4_groups))
        for n in range(model_level_4_blocks - 1):
            self.enc_4.append(XBlock(model_level_4_channels, model_level_4_channels, 3, 3, 1, 1, model_level_4_groups))

        # encoder level 5
        self.enc_5 = nn.ModuleList()
        self.enc_5.append(XBlock(model_level_4_channels, model_level_5_channels, 3, 3, stride5, stride5, model_level_5_groups))
        for n in range(model_level_5_blocks - 1):
            self.enc_5.append(XBlock(model_level_5_channels, model_level_5_channels, 3, 3, 1, 1, model_level_5_groups))

        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(model_level_5_channels, data_num_classes, bias=True))

    # forward path
    def forward(self, x):

        # encoder level 1
        for layer in self.enc_1:
            x = layer(x)

        # encoder level 2
        for layer in self.enc_2:
            x = layer(x)

        # encoder level 3
        for layer in self.enc_3:
            x = layer(x)

        # encoder level 4
        for layer in self.enc_4:
            x = layer(x)

        # encoder level 5
        for layer in self.enc_5:
            x = layer(x)

        # decoder
        for layer in self.dec:
            x = layer(x)

        # return
        return x

# create
model = Model(DATA_NUM_CHANNELS,
              MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_1_CHANNELS, MODEL_LEVEL_1_GROUPS,
              MODEL_LEVEL_2_BLOCKS, MODEL_LEVEL_2_CHANNELS, MODEL_LEVEL_2_GROUPS,
              MODEL_LEVEL_3_BLOCKS, MODEL_LEVEL_3_CHANNELS, MODEL_LEVEL_3_GROUPS,
              MODEL_LEVEL_4_BLOCKS, MODEL_LEVEL_4_CHANNELS, MODEL_LEVEL_4_GROUPS,
              MODEL_LEVEL_5_BLOCKS, MODEL_LEVEL_5_CHANNELS, MODEL_LEVEL_5_GROUPS,
              DATA_NUM_CLASSES)

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

################################################################################
#
# RESULTS
#
################################################################################

# Epoch   0 Time    184.7 lr = 0.002000 avg loss = 0.008522 accuracy =  9.30
# Epoch   1 Time    190.1 lr = 0.041600 avg loss = 0.007006 accuracy = 23.74
# Epoch   2 Time    190.0 lr = 0.081200 avg loss = 0.005858 accuracy = 32.84
# Epoch   3 Time    190.3 lr = 0.120800 avg loss = 0.005160 accuracy = 38.52
# Epoch   4 Time    190.0 lr = 0.160400 avg loss = 0.004715 accuracy = 42.22
# Epoch   5 Time    189.9 lr = 0.200000 avg loss = 0.004391 accuracy = 45.10
# Epoch   6 Time    190.1 lr = 0.199950 avg loss = 0.004073 accuracy = 49.66
# Epoch   7 Time    190.1 lr = 0.199799 avg loss = 0.003830 accuracy = 52.92
# Epoch   8 Time    190.0 lr = 0.199548 avg loss = 0.003649 accuracy = 52.42
# Epoch   9 Time    189.9 lr = 0.199196 avg loss = 0.003494 accuracy = 53.68
# Epoch  10 Time    189.9 lr = 0.198745 avg loss = 0.003362 accuracy = 52.78
# Epoch  11 Time    189.8 lr = 0.198195 avg loss = 0.003265 accuracy = 57.56
# Epoch  12 Time    189.5 lr = 0.197545 avg loss = 0.003158 accuracy = 60.64
# Epoch  13 Time    189.6 lr = 0.196798 avg loss = 0.003238 accuracy = 60.54
# Epoch  14 Time    189.6 lr = 0.195953 avg loss = 0.003063 accuracy = 58.34
# Epoch  15 Time    189.7 lr = 0.195012 avg loss = 0.002982 accuracy = 60.92
# Epoch  16 Time    189.6 lr = 0.193975 avg loss = 0.002899 accuracy = 60.86
# Epoch  17 Time    189.8 lr = 0.192844 avg loss = 0.002847 accuracy = 62.82
# Epoch  18 Time    189.9 lr = 0.191619 avg loss = 0.002789 accuracy = 61.34
# Epoch  19 Time    190.1 lr = 0.190302 avg loss = 0.002748 accuracy = 60.60
# Epoch  20 Time    190.0 lr = 0.188895 avg loss = 0.002690 accuracy = 63.42
# Epoch  21 Time    189.7 lr = 0.187398 avg loss = 0.002651 accuracy = 64.72
# Epoch  22 Time    189.4 lr = 0.185813 avg loss = 0.002605 accuracy = 63.44
# Epoch  23 Time    189.6 lr = 0.184141 avg loss = 0.002571 accuracy = 63.16
# Epoch  24 Time    189.3 lr = 0.182385 avg loss = 0.002527 accuracy = 63.56
# Epoch  25 Time    189.4 lr = 0.180546 avg loss = 0.002491 accuracy = 63.98
# Epoch  26 Time    189.0 lr = 0.178627 avg loss = 0.002457 accuracy = 64.62
# Epoch  27 Time    189.4 lr = 0.176628 avg loss = 0.002427 accuracy = 65.46
# Epoch  28 Time    189.7 lr = 0.174552 avg loss = 0.002399 accuracy = 62.50
# Epoch  29 Time    189.2 lr = 0.172401 avg loss = 0.002372 accuracy = 65.08
# Epoch  30 Time    189.7 lr = 0.170177 avg loss = 0.002343 accuracy = 63.48
# Epoch  31 Time    189.6 lr = 0.167883 avg loss = 0.002318 accuracy = 65.44
# Epoch  32 Time    189.5 lr = 0.165521 avg loss = 0.002294 accuracy = 64.94
# Epoch  33 Time    189.3 lr = 0.163092 avg loss = 0.002259 accuracy = 66.60
# Epoch  34 Time    189.3 lr = 0.160600 avg loss = 0.002238 accuracy = 66.68
# Epoch  35 Time    189.5 lr = 0.158048 avg loss = 0.002222 accuracy = 66.14
# Epoch  36 Time    189.7 lr = 0.155437 avg loss = 0.002202 accuracy = 66.36
# Epoch  37 Time    189.7 lr = 0.152770 avg loss = 0.002170 accuracy = 65.74
# Epoch  38 Time    189.7 lr = 0.150050 avg loss = 0.002139 accuracy = 65.04
# Epoch  39 Time    189.6 lr = 0.147280 avg loss = 0.002121 accuracy = 64.10
# Epoch  40 Time    189.6 lr = 0.144462 avg loss = 0.002095 accuracy = 67.66
# Epoch  41 Time    189.8 lr = 0.141600 avg loss = 0.002090 accuracy = 66.10
# Epoch  42 Time    189.7 lr = 0.138696 avg loss = 0.002063 accuracy = 67.70
# Epoch  43 Time    190.0 lr = 0.135753 avg loss = 0.002028 accuracy = 67.52
# Epoch  44 Time    190.1 lr = 0.132774 avg loss = 0.002013 accuracy = 67.14
# Epoch  45 Time    189.5 lr = 0.129762 avg loss = 0.001987 accuracy = 68.16
# Epoch  46 Time    189.8 lr = 0.126721 avg loss = 0.001973 accuracy = 68.10
# Epoch  47 Time    189.9 lr = 0.123652 avg loss = 0.001946 accuracy = 68.04
# Epoch  48 Time    189.9 lr = 0.120560 avg loss = 0.001943 accuracy = 66.64
# Epoch  49 Time    189.8 lr = 0.117447 avg loss = 0.001901 accuracy = 67.30
# Epoch  50 Time    189.4 lr = 0.114317 avg loss = 0.001888 accuracy = 68.40
# Epoch  51 Time    189.7 lr = 0.111173 avg loss = 0.001866 accuracy = 67.16
# Epoch  52 Time    189.7 lr = 0.108017 avg loss = 0.001830 accuracy = 68.50
# Epoch  53 Time    189.7 lr = 0.104853 avg loss = 0.001832 accuracy = 67.86
# Epoch  54 Time    189.6 lr = 0.101685 avg loss = 0.001795 accuracy = 68.68
# Epoch  55 Time    189.9 lr = 0.098515 avg loss = 0.001763 accuracy = 67.62
# Epoch  56 Time    189.7 lr = 0.095347 avg loss = 0.001754 accuracy = 68.60
# Epoch  57 Time    190.0 lr = 0.092183 avg loss = 0.001734 accuracy = 68.56
# Epoch  58 Time    189.6 lr = 0.089027 avg loss = 0.001707 accuracy = 68.24
# Epoch  59 Time    189.8 lr = 0.085883 avg loss = 0.001690 accuracy = 69.16
# Epoch  60 Time    189.5 lr = 0.082753 avg loss = 0.001654 accuracy = 69.44
# Epoch  61 Time    189.4 lr = 0.079640 avg loss = 0.001633 accuracy = 68.66
# Epoch  62 Time    189.8 lr = 0.076548 avg loss = 0.001608 accuracy = 68.20
# Epoch  63 Time    190.2 lr = 0.073479 avg loss = 0.001597 accuracy = 69.64
# Epoch  64 Time    189.6 lr = 0.070438 avg loss = 0.001567 accuracy = 69.10
# Epoch  65 Time    189.6 lr = 0.067426 avg loss = 0.001545 accuracy = 68.04
# Epoch  66 Time    189.8 lr = 0.064447 avg loss = 0.001522 accuracy = 69.46
# Epoch  67 Time    189.8 lr = 0.061504 avg loss = 0.001508 accuracy = 67.90
# Epoch  68 Time    189.6 lr = 0.058600 avg loss = 0.001475 accuracy = 68.92
# Epoch  69 Time    189.7 lr = 0.055738 avg loss = 0.001440 accuracy = 69.66
# Epoch  70 Time    189.8 lr = 0.052920 avg loss = 0.001427 accuracy = 70.30
# Epoch  71 Time    190.1 lr = 0.050150 avg loss = 0.001410 accuracy = 68.96
# Epoch  72 Time    189.9 lr = 0.047430 avg loss = 0.001391 accuracy = 68.52
# Epoch  73 Time    189.8 lr = 0.044763 avg loss = 0.001353 accuracy = 70.06
# Epoch  74 Time    190.2 lr = 0.042152 avg loss = 0.001325 accuracy = 70.30
# Epoch  75 Time    189.7 lr = 0.039600 avg loss = 0.001319 accuracy = 69.12
# Epoch  76 Time    190.0 lr = 0.037108 avg loss = 0.001286 accuracy = 69.78
# Epoch  77 Time    190.0 lr = 0.034679 avg loss = 0.001274 accuracy = 70.64
# Epoch  78 Time    190.1 lr = 0.032317 avg loss = 0.001250 accuracy = 69.14
# Epoch  79 Time    190.0 lr = 0.030023 avg loss = 0.001220 accuracy = 70.60
# Epoch  80 Time    189.9 lr = 0.027799 avg loss = 0.001197 accuracy = 70.56
# Epoch  81 Time    189.7 lr = 0.025648 avg loss = 0.001186 accuracy = 70.72
# Epoch  82 Time    189.7 lr = 0.023572 avg loss = 0.001149 accuracy = 70.84
# Epoch  83 Time    189.8 lr = 0.021573 avg loss = 0.001158 accuracy = 70.40
# Epoch  84 Time    189.9 lr = 0.019654 avg loss = 0.001140 accuracy = 70.56
# Epoch  85 Time    189.7 lr = 0.017815 avg loss = 0.001122 accuracy = 70.42
# Epoch  86 Time    189.8 lr = 0.016059 avg loss = 0.001091 accuracy = 70.54
# Epoch  87 Time    189.8 lr = 0.014387 avg loss = 0.001083 accuracy = 70.58
# Epoch  88 Time    189.9 lr = 0.012802 avg loss = 0.001064 accuracy = 70.64
# Epoch  89 Time    190.4 lr = 0.011305 avg loss = 0.001063 accuracy = 71.30
# Epoch  90 Time    189.8 lr = 0.009898 avg loss = 0.001031 accuracy = 71.24
# Epoch  91 Time    190.5 lr = 0.008581 avg loss = 0.001024 accuracy = 70.94
# Epoch  92 Time    190.0 lr = 0.007356 avg loss = 0.001011 accuracy = 70.68
# Epoch  93 Time    190.1 lr = 0.006225 avg loss = 0.001000 accuracy = 70.82
# Epoch  94 Time    190.1 lr = 0.005188 avg loss = 0.000999 accuracy = 70.68
# Epoch  95 Time    190.0 lr = 0.004247 avg loss = 0.000992 accuracy = 70.92
# Epoch  96 Time    190.0 lr = 0.003402 avg loss = 0.000973 accuracy = 71.34
# Epoch  97 Time    190.0 lr = 0.002655 avg loss = 0.000963 accuracy = 71.26
# Epoch  98 Time    189.5 lr = 0.002005 avg loss = 0.000974 accuracy = 71.26
# Epoch  99 Time    189.8 lr = 0.001455 avg loss = 0.000967 accuracy = 71.00
# Epoch 100 Time    189.9 lr = 0.001004 avg loss = 0.000967 accuracy = 71.42
# Epoch 101 Time    189.9 lr = 0.000652 avg loss = 0.000973 accuracy = 71.10
# Epoch 102 Time    189.9 lr = 0.000401 avg loss = 0.000960 accuracy = 71.14
# Epoch 103 Time    190.2 lr = 0.000250 avg loss = 0.000959 accuracy = 71.48
# Epoch 104 Time    190.1 lr = 0.000200 avg loss = 0.000958 accuracy = 71.12
