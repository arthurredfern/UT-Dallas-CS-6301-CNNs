################################################################################
#
# xNNs_Code_011_MNIST.py
#
# DESCRIPTION
#
#    MNIST image classification using PyTorch
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
#    1. This configuration achieves 95.5% accuracy in 9 epochs.  Accuracy can be
#       improved via
#       - Improved training data augmentation
#       - Improved network design
#       - Improved network training
#
# TO DO
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn    as nn
import torch.optim as optim

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import math
import numpy             as np
import matplotlib.pyplot as plt
# %matplotlib inline

# version check
# print(torch.__version__)

################################################################################
#
# PARAMETERS
#
################################################################################

# data (general)
DATA_NUM_CHANNELS = 1
DATA_CROP_ROWS    = 28
DATA_CROP_COLS    = 28
DATA_NUM_CLASSES  = 10

# data (for [0, 1] normalization of MNIST)
DATA_NORM = 255.0

# data (for display)
DATA_CLASS_NAMES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# model
MODEL_LEVEL_0_BLOCKS   = 1
MODEL_LEVEL_1_BLOCKS   = 1
MODEL_LEVEL_0_CHANNELS = 1000
MODEL_LEVEL_1_CHANNELS = 100

# training (general)
TRAINING_BATCH_SIZE  = 32
TRAINING_NUM_WORKERS = 4
TRAINING_LR_MAX      = 0.001

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 3
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 6
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# training (staircase learning rate)
# TRAINING_LR_SCALE   = 0.1
# TRAINING_LR_EPOCHS  = 3
# TRAINING_NUM_EPOCHS = 9

################################################################################
#
# DATA
#
################################################################################

# transforms for training and testing datasets
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, DATA_NORM)])
transform_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, DATA_NORM)])

# training and testing datasets with applied transform
dataset_train = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform_train)
dataset_test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# training and testing datasets loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAINING_BATCH_SIZE, shuffle=True,  num_workers=TRAINING_NUM_WORKERS, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=TRAINING_BATCH_SIZE, shuffle=False, num_workers=TRAINING_NUM_WORKERS, drop_last=True)

# debug - datasets
# print(dataset_train) # displays dataset info
# print(dataset_test)  # displays dataset info
# data_iterator_train = iter(dataloader_train)
# inputs, labels      = data_iterator_train.next()
# print(inputs.size())
# print(labels.size())
# data_iterator_test = iter(dataloader_test)
# inputs, labels     = data_iterator_test.next()
# print(inputs.size())
# print(labels.size())

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self, data_num_channels, data_crop_rows, data_crop_cols, data_num_classes, model_level_0_blocks, model_level_1_blocks, model_level_0_channels, model_level_1_channels):

        # parent initialization
        super(Model, self).__init__()

        # input
        self.input = nn.Flatten(start_dim=1, end_dim=-1)

        # encoder level 0
        self.encoder0 = nn.ModuleList()
        self.encoder0.append(nn.Linear(data_num_channels*data_crop_rows*data_crop_cols, model_level_0_channels, bias=True))
        self.encoder0.append(nn.ReLU())
        for n in range(model_level_0_blocks - 1):
            self.encoder0.append(nn.Linear(model_level_0_channels, model_level_0_channels, bias=True))
            self.encoder0.append(nn.ReLU())

        # encoder level 1
        self.encoder1 = nn.ModuleList()
        self.encoder1.append(nn.Linear(model_level_0_channels, model_level_1_channels, bias=True))
        self.encoder1.append(nn.ReLU())
        for n in range(model_level_1_blocks - 1):
            self.encoder1.append(nn.Linear(model_level_1_channels, model_level_1_channels, bias=True))
            self.encoder1.append(nn.ReLU())

        # decoder
        self.decoder = nn.Linear(model_level_1_channels, data_num_classes, bias=True)

    # forward path
    def forward(self, x):

        # input
        x = self.input(x)

        # encoder level 0
        for layer in self.encoder0:
            x = layer(x)

        # encoder level 1
        for layer in self.encoder1:
            x = layer(x)

        # decoder
        x = self.decoder(x)

        # return
        return x

# create
model = Model(DATA_NUM_CHANNELS, DATA_CROP_ROWS, DATA_CROP_COLS, DATA_NUM_CLASSES, MODEL_LEVEL_0_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_0_CHANNELS, MODEL_LEVEL_1_CHANNELS)

# visualization
# print(model)

################################################################################
#
# TRAIN
#
################################################################################

# start epoch
start_epoch = 0

# learning rate schedule
def lr_schedule(epoch):

    # staircase
    # lr = TRAINING_LR_MAX*math.pow(TRAINING_LR_SCALE, math.floor(epoch/TRAINING_LR_EPOCHS))

    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    # debug - learning rate display
    # print(epoch)
    # print(lr)

    return lr

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# transfer the network to the device
model.to(device)

# cycle through the epochs
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):

    # initialize train set statistics
    model.train()
    training_loss = 0.0
    num_batches   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the train set
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

    # initialize test set statistics
    model.eval()
    test_correct = 0
    test_total   = 0

    # no weight update / no gradient needed
    with torch.no_grad():

        # cycle through the test set
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
    print('Epoch {0:2d} avg loss = {1:8.6f} accuracy = {2:5.2f}'.format(epoch, (training_loss/num_batches)/TRAINING_BATCH_SIZE, (100.0*test_correct/test_total)))

################################################################################
#
# TEST
#
################################################################################

# initialize test set statistics
model.eval()
test_correct = 0
test_total   = 0

# initialize class statistics
class_correct = list(0. for i in range(DATA_NUM_CLASSES))
class_total   = list(0. for i in range(DATA_NUM_CLASSES))

# no weight update / no gradient needed
with torch.no_grad():

    # cycle through the test set
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

        # update class statistics
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label                 = labels[i]
            class_correct[label] += c[i].item()
            class_total[label]   += 1

# test set statistics
print('Accuracy of test set = {0:5.2f}'.format((100.0*test_correct/test_total)))
print('')

# class statistics
for i in range(DATA_NUM_CLASSES):
    print('Accuracy of {0:5s}    = {1:5.2f}'.format(DATA_CLASS_NAMES[i], (100.0*class_correct[i]/class_total[i])))

################################################################################
#
# DISPLAY
#
################################################################################

# set to evaluation mode
model.eval()

# extract a batch of data
data_iterator  = iter(dataloader_test)
inputs, labels = data_iterator.next()

# images and ground truth labels
images    = torchvision.utils.make_grid(inputs)*DATA_NORM
np_images = images.numpy()
plt.imshow(np.transpose(np_images, (1, 2, 0)))
print('Ground truth = ', ' '.join('%5s' % DATA_CLASS_NAMES[labels[j]] for j in range(TRAINING_BATCH_SIZE)))

# move it to the appropriate device
inputs, labels = inputs.to(device), labels.to(device)

# forward pass and prediction
outputs      = model(inputs)
_, predicted = torch.max(outputs, 1)

# predicted labels
print('Predicted    = ', ' '.join('%5s' % DATA_CLASS_NAMES[predicted[j]] for j in range(TRAINING_BATCH_SIZE)))
print('')