################################################################################
#
# xNNs_Code_031_CIFAR_ResNetV2b.py
#
# DESCRIPTION
#
#    PyTorch image classification using CIFAR
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
#    1. This configuration achieves 91.4% accuracy in 60 epochs with each epoch
#       taking ~ 70s on Google Colab.  Accuracy can be improved via
#       - Improved training data augmentation
#       - Improved network design
#       - Improved network training
#
#    2. Examples (currently commented out) are included for the following
#       - Computing the dataset mean and std dev
#       - Checkpointing during training and restarting training after a crash
#
# TO DO
#
#    1. Update class name display so names do not need to be manually entered
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

# visualization
# !pip install --quiet torchviz
# from   torchviz     import make_dot
# from   torchsummary import summary

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
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 10
DATA_CROP_ROWS    = 28
DATA_CROP_COLS    = 28

# data (for [-1, 1] or 0 mean 1 var normalization of CIFAR-10)
DATA_MEAN    = (0.5, 0.5, 0.5)
DATA_STD_DEV = (0.5, 0.5, 0.5)
# DATA_MEAN    = (0.49137914, 0.48213690, 0.44650456)
# DATA_STD_DEV = (0.24703294, 0.24348527, 0.26158544)

# data (for display)
DATA_CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model
MODEL_LEVEL_0_BLOCKS            = 4
MODEL_LEVEL_1_BLOCKS            = 6
MODEL_LEVEL_2_BLOCKS            = 3
MODEL_TAIL_END_CHANNELS         = 32
MODEL_LEVEL_0_IDENTITY_CHANNELS = 128
MODEL_LEVEL_0_RESIDUAL_CHANNELS = 32
MODEL_LEVEL_1_IDENTITY_CHANNELS = 256
MODEL_LEVEL_1_RESIDUAL_CHANNELS = 64
MODEL_LEVEL_2_IDENTITY_CHANNELS = 512
MODEL_LEVEL_2_RESIDUAL_CHANNELS = 128

# training (general)
TRAINING_BATCH_SIZE  = 32
TRAINING_NUM_WORKERS = 4
TRAINING_LR_MAX      = 0.001

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 55
# TRAINING_LR_FINAL_EPOCHS = 2 # uncomment for a quick test
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# training (staircase learning rate)
# TRAINING_LR_SCALE   = 0.1
# TRAINING_LR_EPOCHS  = 10
# TRAINING_NUM_EPOCHS = 30

# save
# SAVE_MODEL_PATH      = './save/model/'
# SAVE_CHECKPOINT_FILE = SAVE_MODEL_PATH + 'training_checkpoint.pt'
# !mkdir -p "$SAVE_MODEL_PATH"

################################################################################
#
# DATA
#
################################################################################

# transforms for training and testing datasets
transform_train = transforms.Compose([transforms.RandomCrop((DATA_CROP_ROWS, DATA_CROP_COLS)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.CenterCrop((DATA_CROP_ROWS, DATA_CROP_COLS)), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# training and testing datasets with applied transform
dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
dataset_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# training and testing datasets loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAINING_BATCH_SIZE, shuffle=True,  num_workers=TRAINING_NUM_WORKERS, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=TRAINING_BATCH_SIZE, shuffle=False, num_workers=TRAINING_NUM_WORKERS, drop_last=True)

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self, data_num_channels, data_num_classes, model_level_0_blocks, model_level_1_blocks, model_level_2_blocks, model_tail_end_channels, model_level_0_identity_channels, model_level_0_residual_channels, model_level_1_identity_channels, model_level_1_residual_channels, model_level_2_identity_channels, model_level_2_residual_channels):

        # parent initialization
        super(Model, self).__init__()

        # encoder tail
        self.enc_tail = nn.Conv2d(data_num_channels, model_tail_end_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

        # encoder level 0 special bottleneck
        self.enc_0a_residual = nn.ModuleList()
        self.enc_0a_residual.append(nn.BatchNorm2d(model_tail_end_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_0a_residual.append(nn.ReLU())
        self.enc_0a_residual.append(nn.Conv2d(model_tail_end_channels, model_level_0_residual_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_0a_residual.append(nn.BatchNorm2d(model_level_0_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_0a_residual.append(nn.ReLU())
        self.enc_0a_residual.append(nn.Conv2d(model_level_0_residual_channels, model_level_0_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_0a_residual.append(nn.BatchNorm2d(model_level_0_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_0a_residual.append(nn.ReLU())
        self.enc_0a_residual.append(nn.Conv2d(model_level_0_residual_channels, model_level_0_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_0a_identity = nn.Conv2d(model_tail_end_channels, model_level_0_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

        # encoder level 0 standard bottleneck
        self.enc_0b_residual = nn.ModuleList()
        for n in range(model_level_0_blocks - 1):
            self.enc_0b_residual.append(nn.BatchNorm2d(model_level_0_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_0b_residual.append(nn.ReLU())
            self.enc_0b_residual.append(nn.Conv2d(model_level_0_identity_channels, model_level_0_residual_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_0b_residual.append(nn.BatchNorm2d(model_level_0_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_0b_residual.append(nn.ReLU())
            self.enc_0b_residual.append(nn.Conv2d(model_level_0_residual_channels, model_level_0_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_0b_residual.append(nn.BatchNorm2d(model_level_0_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_0b_residual.append(nn.ReLU())
            self.enc_0b_residual.append(nn.Conv2d(model_level_0_residual_channels, model_level_0_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))

        # encoder level 1 down sampling bottleneck
        self.enc_1a_residual = nn.ModuleList()
        self.enc_1a_residual.append(nn.BatchNorm2d(model_level_0_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_1a_residual.append(nn.ReLU())
        self.enc_1a_residual.append(nn.Conv2d(model_level_0_identity_channels, model_level_1_residual_channels, (1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_1a_residual.append(nn.BatchNorm2d(model_level_1_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_1a_residual.append(nn.ReLU())
        self.enc_1a_residual.append(nn.Conv2d(model_level_1_residual_channels, model_level_1_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_1a_residual.append(nn.BatchNorm2d(model_level_1_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_1a_residual.append(nn.ReLU())
        self.enc_1a_residual.append(nn.Conv2d(model_level_1_residual_channels, model_level_1_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_1a_identity = nn.Conv2d(model_level_0_identity_channels, model_level_1_identity_channels, (1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

        # encoder level 1 standard bottleneck
        self.enc_1b_residual = nn.ModuleList()
        for n in range(model_level_1_blocks - 1):
            self.enc_1b_residual.append(nn.BatchNorm2d(model_level_1_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_1b_residual.append(nn.ReLU())
            self.enc_1b_residual.append(nn.Conv2d(model_level_1_identity_channels, model_level_1_residual_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_1b_residual.append(nn.BatchNorm2d(model_level_1_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_1b_residual.append(nn.ReLU())
            self.enc_1b_residual.append(nn.Conv2d(model_level_1_residual_channels, model_level_1_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_1b_residual.append(nn.BatchNorm2d(model_level_1_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_1b_residual.append(nn.ReLU())
            self.enc_1b_residual.append(nn.Conv2d(model_level_1_residual_channels, model_level_1_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))

        # encoder level 2 down sampling bottleneck
        self.enc_2a_residual = nn.ModuleList()
        self.enc_2a_residual.append(nn.BatchNorm2d(model_level_1_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_2a_residual.append(nn.ReLU())
        self.enc_2a_residual.append(nn.Conv2d(model_level_1_identity_channels, model_level_2_residual_channels, (1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_2a_residual.append(nn.BatchNorm2d(model_level_2_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_2a_residual.append(nn.ReLU())
        self.enc_2a_residual.append(nn.Conv2d(model_level_2_residual_channels, model_level_2_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_2a_residual.append(nn.BatchNorm2d(model_level_2_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_2a_residual.append(nn.ReLU())
        self.enc_2a_residual.append(nn.Conv2d(model_level_2_residual_channels, model_level_2_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_2a_identity = nn.Conv2d(model_level_1_identity_channels, model_level_2_identity_channels, (1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')

        # encoder level 2 standard bottleneck
        self.enc_2b_residual = nn.ModuleList()
        for n in range(model_level_2_blocks - 1):
            self.enc_2b_residual.append(nn.BatchNorm2d(model_level_2_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_2b_residual.append(nn.ReLU())
            self.enc_2b_residual.append(nn.Conv2d(model_level_2_identity_channels, model_level_2_residual_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_2b_residual.append(nn.BatchNorm2d(model_level_2_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_2b_residual.append(nn.ReLU())
            self.enc_2b_residual.append(nn.Conv2d(model_level_2_residual_channels, model_level_2_residual_channels, (3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
            self.enc_2b_residual.append(nn.BatchNorm2d(model_level_2_residual_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.enc_2b_residual.append(nn.ReLU())
            self.enc_2b_residual.append(nn.Conv2d(model_level_2_residual_channels, model_level_2_identity_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))

        # encoder finalize before decoder
        self.enc_final = nn.ModuleList()
        self.enc_final.append(nn.BatchNorm2d(model_level_2_identity_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_final.append(nn.ReLU())

        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(model_level_2_identity_channels, data_num_classes, bias=True))

    # forward path
    def forward(self, x):

        # encoder tail
        x = self.enc_tail(x)

        # encoder level 0 special bottleneck
        identity = self.enc_0a_identity(x)
        for layer in self.enc_0a_residual:
            x = layer(x)
        x = x + identity

        # encoder level 0 standard bottleneck
        identity = x
        for layer in self.enc_0b_residual:
            x = layer(x)
        x = x + identity

        # encoder level 1 down sampling bottleneck
        identity = self.enc_1a_identity(x)
        for layer in self.enc_1a_residual:
            x = layer(x)
        x = x + identity

        # encoder level 1 standard bottleneck
        identity = x
        for layer in self.enc_1b_residual:
            x = layer(x)
        x = x + identity

        # encoder level 2 down sampling bottleneck
        identity = self.enc_2a_identity(x)
        for layer in self.enc_2a_residual:
            x = layer(x)
        x = x + identity

        # encoder level 2 standard bottleneck
        identity = x
        for layer in self.enc_2b_residual:
            x = layer(x)
        x = x + identity

        # encoder finalize before decoder
        for layer in self.enc_final:
            x = layer(x)

        # decoder
        for layer in self.dec:
            x = layer(x)

        # return
        return x

# create
model = Model(DATA_NUM_CHANNELS, DATA_NUM_CLASSES, MODEL_LEVEL_0_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_2_BLOCKS, MODEL_TAIL_END_CHANNELS, MODEL_LEVEL_0_IDENTITY_CHANNELS, MODEL_LEVEL_0_RESIDUAL_CHANNELS, MODEL_LEVEL_1_IDENTITY_CHANNELS, MODEL_LEVEL_1_RESIDUAL_CHANNELS, MODEL_LEVEL_2_IDENTITY_CHANNELS, MODEL_LEVEL_2_RESIDUAL_CHANNELS)

# visualization
# print(model)
# x = torch.zeros(1, DATA_CHANNELS, DATA_CROP_ROWS, DATA_CROP_COLS, dtype=torch.float)
# y = model(x)
# make_dot(y)

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

# checkpoint loading
# checkpoint = torch.load(SAVE_CHECKPOINT_FILE)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch'] + 1

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# transfer the network to the device
model.to(device)
# summary(model, (DATA_CHANNELS, DATA_CROP_ROWS, DATA_CROP_COLS))

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

    # checkpoint saving
    # torch.save({
    #     'epoch':                epoch,
    #     'model_state_dict':     model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    #     }, SAVE_CHECKPOINT_FILE)

    # epoch statistics
    print('Epoch {0:2d} avg loss = {1:8.6f} accuracy = {2:5.2f}'.format(epoch, (training_loss/num_batches)/TRAINING_BATCH_SIZE, (100.0*test_correct/test_total)))

    # checkpoint test
    # if epoch == 2:
    #     break

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
images    = torchvision.utils.make_grid(inputs)/2 + 0.5
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
