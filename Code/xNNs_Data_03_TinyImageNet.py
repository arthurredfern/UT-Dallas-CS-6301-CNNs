################################################################################
#
# xNNs_Data_03_TinyImageNet.py
#
# DESCRIPTION
#
#    Download the Tiny ImageNet dataset, unzip it, and pack it into tfrecords
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory:
#       https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (ok to divide into multiple cells)
#    4. Runtime - Run all
#
# RESULTS
#
#    Number of training images:   100000
#    Number of validation images: 10000
#    Mean*:                       [0.47593436 0.44813890 0.39262872]
#    Std dev:                     [0.27633505 0.26869268 0.28134818]
#
#    *After dividing image by 255.0
#
################################################################################


################################################################################
#
# IMPORT
#
################################################################################

import tensorflow as tf
import numpy as np
import os
import re
import sys
import random
import urllib.request
import zipfile
import matplotlib.pyplot as plt
%matplotlib inline


################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_USE_GOOGLE_COLAB = True
DATA_URL              = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATA_DOWNLOAD_DIR     = '/content/gdrive/My Drive/download/'
DATA_TFRECORDS_DIR    = '/content/gdrive/My Drive/data/tiny-imagenet-200/'
DATA_TFRECORDS_TRAIN  = '/content/gdrive/My Drive/data/tiny-imagenet-200/tiny_imagenet_train_{}.tfrecords'
DATA_TFRECORDS_VAL    = '/content/gdrive/My Drive/data/tiny-imagenet-200/tiny_imagenet_val_{}.tfrecords'
DATA_RANDOM_SEED      = 42
DATA_NUM_SHARDS_TRAIN = 20
DATA_NUM_SHARDS_VAL   = 2
DATA_IMAGE_HEIGHT     = 64
DATA_IMAGE_WIDTH      = 64

# training
TRAINING_SHOW_SAMPLE         = True
TRAINING_BATCH_SIZE          = 32
TRAINING_NUM_SAMPLES_TO_SHOW = 10  # must be <= TRAINING_BATCH_SIZE
TRAINING_BUFFER_SIZE         = 1000


################################################################################
#
# DATA DOWNLOAD FUNCTIONS
#
################################################################################

# display download progress
def print_download_progress(count, block_size, total_size):

    # percentage complete
    pct_complete = float(count*block_size)/total_size
    pct_complete = min(1.0, pct_complete)

    # status message
    msg = "\rDownload progress: {0:.1%}".format(pct_complete)

    # display
    sys.stdout.write(msg)
    sys.stdout.flush()

# download and extract data
def maybe_download_and_extract(url, download_dir):

    # filename for saving the downloaded file
    # use the filename from the URL and add it to the download_dir
    filename  = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # download and unzip file if not downloaded yet
    if not os.path.exists(file_path):
        
        # create the directory if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # download the file
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=print_download_progress)

        # display
        print()
        print("Download finished, unzipping files (possibly slow)")

        # extracting files
        zipfile.ZipFile(file=file_path, mode='r').extractall(download_dir)

        # display
        print("Done")

    # file already exists
    else:
        
        # display
        print("Data has already been downloaded and unzipped")


################################################################################
#
# CONVERT TO TFRECORD FUNCTIONS
#
################################################################################

# wrap values to int64
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# wrap values to bytes
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# display conversion progress
def print_conversion_progress(count, total):
    
    # percentage complete
    pct_complete = float(count)/total
    pct_complete = min(1.0, pct_complete)

    # status message
    msg = "\rConversion progress: {0:.1%}".format(pct_complete)

    # display
    sys.stdout.write(msg)
    sys.stdout.flush()

# convert data to tfrecords
def convert(image_paths, labels, out_path):
    
    # display
    print("Converting: " + out_path)
    
    # number of images (used for tracking progress)
    num_images = len(image_paths)
    
    # open a TFRecordWriter for the output file
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        # iterate over all the image paths and class labels
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            
            # display the progress
            print_conversion_progress(count=i, total=num_images-1)

            # read the image file
            with tf.gfile.GFile(path, 'rb') as fid:
                image_data = fid.read()

            # create a dictionary with the data to be save in the TFRecords file
            data = \
                {
                    'image': wrap_bytes(image_data),
                    'label': wrap_int64(label)
                }

            # wrap the data as TensorFlow features
            feature = tf.train.Features(feature=data)

            # wrap again as a TensorFlow example
            example = tf.train.Example(features=feature)

            # serialize the data
            serialized = example.SerializeToString()
            
            # write the serialized data to the TFRecords file
            writer.write(serialized)

    # display
    print()


################################################################################
#
# DOWNLOAD DATA AND CONVERT TO TFRECORD
#
################################################################################

# mount Google Drive
# may need to go to web site and enter authorization code
if DATA_USE_GOOGLE_COLAB == True:
    from google.colab import drive
    drive.mount('/content/gdrive')

# download data
maybe_download_and_extract(url=DATA_URL, download_dir=DATA_DOWNLOAD_DIR)

# set the training and validation directories
train_dir = os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/train')
val_dir   = os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/val')

# initialize file paths and labels
train_paths  = []
train_labels = []
val_paths    = []
val_labels   = []

# find the class ids in training set (ie. n01442537, etc)
classes = [x for x in os.listdir(train_dir)]

# create an id to label (0-199) dictionary
classes_id_label = {}
with open(os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/wnids.txt')) as f:
    content = f.readlines()
content = [x.strip() for x in content]
for counter, value in enumerate(content):
    classes_id_label[value] = counter

# iterate through all training images and save their paths and labels
for c in classes:
    for image in os.listdir(os.path.join(train_dir, c, 'images')):
        train_paths.append(os.path.join(train_dir, c, 'images', image))
        train_labels.append(classes_id_label[c])

# permutate the paths and labels
train_paths  = np.random.RandomState(seed=DATA_RANDOM_SEED).permutation(train_paths)
train_labels = np.random.RandomState(seed=DATA_RANDOM_SEED).permutation(train_labels)

# read all validation images names into a list
val_images = [x for x in os.listdir(os.path.join(val_dir,'images'))]

# record the paths and labels (using the val_annotation.txt in tiny-imagenet-200) of the validation images
with open(os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/val/val_annotations.txt')) as f:
    content = f.readlines()
val_image_class = {}
for line in content:
    val_image_class[re.split(r'\t+', line.strip())[0]] = re.split(r'\t+', line.strip())[1]
for image in val_images:
    val_paths.append(os.path.join(val_dir, 'images', image))
    val_labels.append(classes_id_label[val_image_class[image]])

# number of training and validation images
train_data_size = len(train_paths)
val_data_size   = len(val_paths)

# number of images in each of the shards
nt = int(train_data_size/DATA_NUM_SHARDS_TRAIN)
nv = int(val_data_size/DATA_NUM_SHARDS_VAL)

# create the tfrecords directory if it does not exist
if not os.path.exists(DATA_TFRECORDS_DIR):
    os.makedirs(DATA_TFRECORDS_DIR)

# pack the training images into tfrecords
for i in range(DATA_NUM_SHARDS_TRAIN):
    tfrecord_path = os.path.join(DATA_TFRECORDS_DIR, 'tiny_imagenet_train_{}.tfrecords'.format(i))
    convert(train_paths[i*nt:(i+1)*nt], train_labels[i*nt:(i+1)*nt], tfrecord_path)

# pack the validation images into tfrecords
for i in range(DATA_NUM_SHARDS_VAL):
    tfrecord_path = os.path.join(DATA_TFRECORDS_DIR, 'tiny_imagenet_val_{}.tfrecords'.format(i))
    convert(val_paths[i*nv:(i+1)*nv], val_labels[i*nv:(i+1)*nv], tfrecord_path)


################################################################################
#
# COUNT THE NUM OF TRAIN AND VAL IMAGES
#
################################################################################

# parser
def parser(record):
    
    # feature definition
    features = \
    {'image': tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64)}

    # extract a single example
    sample = tf.parse_single_example(record, features)

    # image decode
    image = tf.image.decode_image(sample['image'], channels=3)

    # normalization to [0, 1]
    image = tf.cast(image, tf.float32)/255.0

    # label conversion
    label = tf.cast(sample['label'], tf.int32)

    return image, label

# training tfrecords
tfrecords_train = []
for i in range(DATA_NUM_SHARDS_TRAIN):
    tfrecords_train.append(DATA_TFRECORDS_TRAIN.format(i))

# validation tfrecords
tfrecords_val = []
for i in range(DATA_NUM_SHARDS_VAL):
    tfrecords_val.append(DATA_TFRECORDS_VAL.format(i))

# display the number of training images
train_num = 0
for fn in tfrecords_train:
    for record in tf.python_io.tf_record_iterator(fn):
        train_num += 1
print("Number of training images:   {}".format(train_num))

# display the number of validation images
val_num = 0
for fn in tfrecords_val:
    for record in tf.python_io.tf_record_iterator(fn):
        val_num += 1
print("Number of validation images: {}".format(val_num))


################################################################################
#
# COMPUTE THE MEAN AND STD DEV
#
################################################################################

# number of batches
num_batch = train_num/TRAINING_BATCH_SIZE

# dataset
dataset_train = tf.data.TFRecordDataset(tfrecords_train)

# transformation
dataset_train = dataset_train.repeat(1).map(parser).batch(TRAINING_BATCH_SIZE)

# iterator
iterator            = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
iterator_init_train = iterator.make_initializer(dataset_train)

# example
images, labels = iterator.get_next()

# create a session
session = tf.Session()

# initialize global variables
session.run(tf.global_variables_initializer())

# channel average
channel_avg = tf.reduce_mean(images, axis=[1,2])

# initialize the total sum
tot_sum = np.array([0.0,0.0,0.0])

# initialize the iterator to the training dataset
session.run(iterator_init_train)

# accumulate the total sum
try:
    while True:
        batch_channel_avg  = session.run([channel_avg])
        batch_avg          = np.mean(batch_channel_avg, axis=1)
        tot_sum           += batch_avg[0]

except tf.errors.OutOfRangeError:
    pass

# display the mean
tot_mean = tot_sum/num_batch
print("Mean:    {}".format(tot_mean))

# close the session
session.close()

# reshape the mean to a (1, 1, 3) tensor
mean_channel = tf.constant(tot_mean, dtype=tf.float32)
mean_channel = tf.reshape(mean_channel, [1,1,3])

# create a session
session = tf.Session()

# initialize global variables
session.run(tf.global_variables_initializer())

# compute the variance
image_sub_mean_sq = tf.reduce_sum(tf.math.square(tf.math.subtract(images, mean_channel)), axis=[1,2])

# initialize the total sum
tot_sum = np.array([0.0, 0.0, 0.0])

# initialize the iterator to the training dataset
session.run(iterator_init_train)

# accumulate the total sum
try:
    while True:
        batch_image_sub_mean_sq  = session.run([image_sub_mean_sq])
        batch_channel_sum        = np.sum(batch_image_sub_mean_sq, axis=1)
        tot_sum                 += batch_channel_sum[0]
except tf.errors.OutOfRangeError:
    pass

# close the session
session.close()

# display the standard deviation
std = np.sqrt(tot_sum/(DATA_IMAGE_WIDTH*DATA_IMAGE_HEIGHT*train_num))
print("Std dev: {}".format(std))


################################################################################
#
# VISUALIZE IMAGES AND LABELS TO DOUBLE CHECK
#
################################################################################

# parser
def parser_visual(record):
    
    # feature definition
    features = \
    {'image': tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64)}

    # extract a single example
    sample = tf.parse_single_example(record, features)

    # image decode
    image = tf.image.decode_image(sample['image'], channels=3)

    # label conversion
    label = tf.cast(sample['label'], tf.int32)

    return image, label

# display
if TRAINING_SHOW_SAMPLE == True:

    # dataset
    dataset_train = tf.data.TFRecordDataset(tfrecords_train)
    dataset_val   = tf.data.TFRecordDataset(tfrecords_val)

    # transformation
    dataset_train = dataset_train.shuffle(TRAINING_BUFFER_SIZE).repeat(1).map(parser_visual).batch(TRAINING_BATCH_SIZE)
    dataset_val   = dataset_val.shuffle(TRAINING_BUFFER_SIZE).repeat(1).map(parser_visual).batch(TRAINING_BATCH_SIZE)

    # iterator
    iterator            = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    iterator_init_train = iterator.make_initializer(dataset_train)
    iterator_init_val   = iterator.make_initializer(dataset_val)

    # example
    images, labels = iterator.get_next()

    # create a session
    session = tf.Session()

    # initialize global variables
    session.run(tf.global_variables_initializer())

    # initialize the iterator to the training dataset and collect a batch of samples
    session.run(iterator_init_train)
    images_sample_train, labels_sample_train = session.run([images, labels])

    # initialize the iterator to the validation dataset and collect a batch of samples
    session.run(iterator_init_val)
    images_sample_val, labels_sample_val = session.run([images, labels])

    # close the session
    session.close()

    # initialize the paths and labels
    train_paths  = []
    train_labels = []
    val_paths    = []
    val_labels   = []

    # read wordnet ids
    classes_name_label = {}
    with open(os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/wnids.txt')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    classes_name_id = {}
    for counter, value in enumerate(content):
        classes_name_label[value] = counter

    # read words names
    with open(os.path.join(DATA_DOWNLOAD_DIR, 'tiny-imagenet-200/words.txt')) as f:
        con = f.readlines()
    con = [x.strip() for x in con]
    classes_name_words = {}
    for i in range(len(con)):
        classes_name_words[con[i].split('\t')[0]] = con[i].split('\t')[1]
    
    # display training images and labels
    for i in range(TRAINING_NUM_SAMPLES_TO_SHOW):
        plt.imshow(images_sample_train[i, :, :, :], interpolation='nearest')
        plt.show()
        for name, label in classes_name_label.items():
            if label == labels_sample_train[i]:
                print(name, classes_name_words[name])

    # display validation images and labels
    for i in range(TRAINING_NUM_SAMPLES_TO_SHOW):
        plt.imshow(images_sample_val[i, :, :, :], interpolation='nearest')
        plt.show()
        for name, label in classes_name_label.items():
            if label == labels_sample_val[i]:
                print(name, classes_name_words[name])
