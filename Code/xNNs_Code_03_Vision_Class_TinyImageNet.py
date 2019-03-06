################################################################################
#
# xNNs_Code_02_Vision_Class_TinyImageNet.py
#
# DESCRIPTION
#
#    TensorFlow Tiny ImageNet example
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Change runtime type - Hardware accelerator - GPU
#    5. Runtime - Run all
#
################################################################################


################################################################################
#
# IMPORT
#
################################################################################

# tenorflow
import tensorflow         as     tf
from   tensorflow         import contrib
from   tensorflow.contrib import autograph

# additional libraries
import numpy             as np
import matplotlib.pyplot as plt
%matplotlib inline


################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_USE_GOOGLE_COLAB  = True
DATA_TFRECORDS_TRAIN   = '/content/gdrive/My Drive/data/tiny-imagenet-200/tiny_imagenet_train_{}.tfrecords'
DATA_TFRECORDS_VAL     = '/content/gdrive/My Drive/data/tiny-imagenet-200/tiny_imagenet_val_{}.tfrecords'
DATA_NUM_SHARDS_TRAIN  = 20
DATA_NUM_SHARDS_VAL    = 2
DATA_NUM_CLASSES       = 200
DATA_NUM_TRAIN         = 500*DATA_NUM_CLASSES
DATA_NUM_VAL           = 50*DATA_NUM_CLASSES
DATA_MEAN_CHANNEL_0    = 0.47593436
DATA_MEAN_CHANNEL_1    = 0.44813890
DATA_MEAN_CHANNEL_2    = 0.39262872
DATA_STD_DEV_CHANNEL_0 = 0.27633505
DATA_STD_DEV_CHANNEL_1 = 0.26869268
DATA_STD_DEV_CHANNEL_2 = 0.28134818

# model
MODEL_LEVEL_0_BLOCKS = 3
MODEL_LEVEL_1_BLOCKS = 4
MODEL_LEVEL_2_BLOCKS = 6
MODEL_LEVEL_3_BLOCKS = 3

# training
TRAINING_IMAGE_SIZE        = 64
TRAINING_CROP_SIZE         = 56
TRAINING_SHUFFLE_BUFFER    = 5000
TRAINING_BATCH_SIZE        = 32
TRAINING_NUM_EPOCHS        = 112                    # 144
TRAINING_MOMENTUM          = 0.9                    # currently not used
TRAINING_REGULARIZER_SCALE = 0.1                    # currently not used
TRAINING_LR_INITIAL        = 0.001
TRAINING_LR_SCALE          = 0.1
TRAINING_LR_EPOCHS         = 48                     # 64
TRAINING_LR_STAIRCASE      = True
TRAINING_MAX_CHECKPOINTS   = 5
TRAINING_CHECKPOINT_FILE   = './logs/model_{}.ckpt' # currently not used


################################################################################
#
# PRE PROCESSING
#
################################################################################

# center crop
def center_crop(image, crop_height, crop_width):
    
    # image shape
    shape  = tf.shape(image)
    height = shape[0]
    width  = shape[1]
    
    # center crop top left point
    amount_to_be_cropped_h = (height - crop_height)
    crop_top               = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left              = amount_to_be_cropped_w // 2
    
    # crop
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

# pre processing - validation
def pre_processing_val(record):

    # feature definition
    features = \
    {'image': tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64)}

    # extract a single example
    sample = tf.parse_single_example(record, features)

    # image decode
    image = tf.image.decode_image(sample['image'], channels=3)

    # center crop
    image = center_crop(image, TRAINING_CROP_SIZE, TRAINING_CROP_SIZE)
    
    # normaliztion
    data_mean = tf.constant([DATA_MEAN_CHANNEL_0, DATA_MEAN_CHANNEL_1, DATA_MEAN_CHANNEL_2], dtype=tf.float32)
    data_mean = tf.reshape(data_mean, [1, 1, 3])
    data_std  = tf.constant([DATA_STD_DEV_CHANNEL_0, DATA_STD_DEV_CHANNEL_1, DATA_STD_DEV_CHANNEL_2], dtype=tf.float32)
    data_std  = tf.reshape(data_std, [1, 1, 3])
    image     = tf.math.divide(tf.math.subtract(tf.cast(image, tf.float32)/255.0, data_mean), data_std)

    # label conversion
    label = tf.cast(sample['label'], tf.int32)

    # return
    return image, label

# pre processing - training
def pre_processing_train(record):
    
    # feature definition
    features = \
    {'image': tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64)}
    
    # extract a single example
    sample = tf.parse_single_example(record, features)

    # image decode
    image = tf.image.decode_image(sample['image'], channels=3)

    # random flip and crop
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[TRAINING_CROP_SIZE, TRAINING_CROP_SIZE, 3])

    # normaliztion
    data_mean = tf.constant([DATA_MEAN_CHANNEL_0, DATA_MEAN_CHANNEL_1, DATA_MEAN_CHANNEL_2], dtype=tf.float32)
    data_mean = tf.reshape(data_mean, [1, 1, 3])
    data_std  = tf.constant([DATA_STD_DEV_CHANNEL_0, DATA_STD_DEV_CHANNEL_1, DATA_STD_DEV_CHANNEL_2], dtype=tf.float32)
    data_std  = tf.reshape(data_std, [1, 1, 3])
    image     = tf.math.divide(tf.math.subtract(tf.cast(image, tf.float32)/255.0, data_mean), data_std)

    # label conversion
    label = tf.cast(sample['label'], tf.int32)

    # return
    return image, label


################################################################################
#
# DATASET
#
################################################################################

# mount Google Drive
# may need to go to web site and enter authorization code
if DATA_USE_GOOGLE_COLAB == True:
    from google.colab import drive
    drive.mount('/content/gdrive')

# training tfrecords
tfrecords_train = []
for i in range(DATA_NUM_SHARDS_TRAIN):
    tfrecords_train.append(DATA_TFRECORDS_TRAIN.format(i))

# validation tfrecords
tfrecords_val = []
for i in range(DATA_NUM_SHARDS_VAL):
    tfrecords_val.append(DATA_TFRECORDS_VAL.format(i))

# dataset
dataset_train = tf.data.TFRecordDataset(tfrecords_train)
dataset_val   = tf.data.TFRecordDataset(tfrecords_val)

# transformation
dataset_train = dataset_train.shuffle(buffer_size=TRAINING_SHUFFLE_BUFFER).repeat().map(pre_processing_train).batch(TRAINING_BATCH_SIZE)
# dataset_val   = dataset_val.shuffle(buffer_size=TRAINING_SHUFFLE_BUFFER).repeat().map(pre_processing_val).batch(TRAINING_BATCH_SIZE)
dataset_val   = dataset_val.repeat().map(pre_processing_val).batch(TRAINING_BATCH_SIZE)


################################################################################
#
# ITERATOR
#
################################################################################

# iterator
iterator            = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
iterator_init_train = iterator.make_initializer(dataset_train)
iterator_init_test  = iterator.make_initializer(dataset_val)

# example
data, labels = iterator.get_next()


################################################################################
#
# MODEL - SEQUENTIAL
#
################################################################################

# sequential model
def model_sequential(data, train_state, num_classes):
    
    # data
    # TRAINING_BATCH_SIZE x rows x cols x channels
    
    # encoder - level 0
    fm       = tf.layers.conv2d(data, 32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    
    # encoder - level 1 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')
    
    # encoder - level 1
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    
    # encoder - level 2 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')
    
    # encoder - level 2
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    
    # encoder - level 3 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')
    
    # encoder - level 3
    fm       = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    fm       = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    features = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu, use_bias=True)
    
    # decoder
    # predictions.shape = TRAINING_BATCH_SIZE x num_classes
    features    = tf.reduce_mean(features, axis=[1, 2])
    predictions = tf.layers.dense(features, num_classes, activation=None, use_bias=True)
    
    # return
    return predictions


################################################################################
#
# MODEL - SEQUENTIAL BATCH NORM
#
################################################################################

# sequential batch norm model
def model_sequential_bn(data, train_state, num_classes):
    
    # data
    # TRAINING_BATCH_SIZE x rows x cols x channels
    
    # encoder - level 0
    fm       = tf.layers.conv2d(data, 32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)

    # encoder - level 1 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')
    
    # encoder - level 1
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    
    # encoder - level 2 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')

    # encoder - level 2
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    
    # encoder - level 3 down sampling
    fm       = tf.layers.max_pooling2d(fm, (3, 3), (2, 2), padding='same', data_format='channels_last')
    
    # encoder - level 3
    fm       = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    fm       = tf.nn.relu(fm)
    fm       = tf.layers.conv2d(fm,   256, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm       = tf.layers.batch_normalization(fm, training=train_state)
    features = tf.nn.relu(fm)
    
    # decoder
    # predictions.shape = TRAINING_BATCH_SIZE x num_classes
    features    = tf.reduce_mean(features, axis=[1, 2])
    predictions = tf.layers.dense(features, num_classes, activation=None, use_bias=True)
    
    # return
    return predictions


################################################################################
#
# MODEL - RESNET V2
#
################################################################################

# resnet model
@autograph.convert()
def model_resnet(data, train_state, level_0_blocks, level_1_blocks, level_2_blocks, level_3_blocks, num_classes):
    
    # data
    # TRAINING_BATCH_SIZE x rows x cols x channels

    # encoder - tail
    fm_id       = tf.layers.conv2d(data, 32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)

    # encoder - level 0 special bottleneck x1
    # input:  32 x 64 x 64
    # filter: 16 x 32 x 1 x 1 / 1
    # filter: 16 x 16 x 3 x 3
    # filter: 64 x 16 x 1 x 1
    # main:   64 x 32 x 1 x 1 / 1
    # output: 64 x 64 x 64
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 16, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 16, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 0 standard bottleneck x(level_0_blocks - 1)
    # input:  64 x 64 x 64
    # filter: 16 x 64 x 1 x 1
    # filter: 16 x 16 x 3 x 3
    # filter: 64 x 16 x 1 x 1
    # main:   identity
    # output: 64 x 64 x 64
    for block_repeat_0 in range(level_0_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 16, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 16, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 1 down sampling bottleneck x1
    # input:   64 x 64 x 64
    # filter:  32 x 64 x 1 x 1 / 2
    # filter:  32 x 32 x 3 x 3
    # filter: 128 x 32 x 1 x 1
    # main:   128 x 64 x 1 x 1 / 2
    # output: 128 x 32 x 32
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  32, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       128, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 1 standard bottleneck x(level_1_blocks - 1)
    # input:  128 x  32 x 32
    # filter:  32 x 128 x 1 x 1
    # filter:  32 x  32 x 3 x 3
    # filter: 128 x  32 x 1 x 1
    # main:   identity
    # output: 128 x  32 x 32
    for block_repeat_1 in range(level_1_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  32, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)
    
    # encoder - level 2 down sampling bottleneck x1
    # input:  128 x  32 x 32
    # filter:  64 x 128 x 1 x 1 / 2
    # filter:  64 x  64 x 3 x 3
    # filter: 256 x  64 x 1 x 1
    # main:   256 x 128 x 1 x 1 / 2
    # output: 256 x  16 x 16
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  64, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       256, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 2 standard bottleneck x(level_2_blocks - 1)
    # input:  256 x  16 x 16
    # filter:  64 x 256 x 1 x 1
    # filter:  64 x  64 x 3 x 3
    # filter: 256 x  64 x 1 x 1
    # main:   identity
    # output: 256 x  16 x 16
    for block_repeat_2 in range(level_2_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  64, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)
    
    # encoder - level 3 down sampling bottleneck x1
    # input:  256 x  16 x 16
    # filter: 128 x 256 x 1 x 1 / 2
    # filter: 128 x 128 x 3 x 3
    # filter: 512 x 128 x 1 x 1
    # main:   512 x 256 x 1 x 1 / 2
    # output: 512 x   8 x 8
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 512, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       512, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 3 standard bottleneck x(level_2_blocks - 1)
    # input:  512 x   8 x 8
    # filter: 128 x 512 x 1 x 1
    # filter: 128 x 128 x 3 x 3
    # filter: 512 x 128 x 1 x 1
    # main:   identity
    # output: 512 x   8 x 8
    for block_repeat_3 in range(level_3_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 128, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 512, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)
    
    # encoder - level 3 special block x1
    # input:  512 x   8 x 8
    # output: 512 x   8 x 8
    fm_id       = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_id       = tf.nn.relu(fm_id)

    # decoder
    # predictions.shape = TRAINING_BATCH_SIZE x num_classes
    fm_id       = tf.reduce_mean(fm_id, axis=[1, 2])
    predictions = tf.layers.dense(fm_id, num_classes, activation=None, use_bias=True)
    
    # return
    return predictions


################################################################################
#
# TRAINING
#
################################################################################

# state
train_state = tf.placeholder(tf.bool, name='train_state')

# data
# num_train         = len(data_train)
# num_test          = len(data_test)
num_train         = DATA_NUM_TRAIN
num_test          = DATA_NUM_VAL
num_batches_train = int(num_train/TRAINING_BATCH_SIZE)
num_batches_test  = int(num_test/TRAINING_BATCH_SIZE)

# display
# print(num_train)
# print(num_test)
# print(num_batches_train)
# print(num_batches_test)

# model
# predictions      = model_sequential(data, train_state, DATA_NUM_CLASSES)
# predictions      = model_sequential_bn(data, train_state, DATA_NUM_CLASSES)
predictions      = model_resnet(data, train_state, MODEL_LEVEL_0_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_2_BLOCKS, MODEL_LEVEL_3_BLOCKS, DATA_NUM_CLASSES)
predictions_test = np.zeros((num_test, DATA_NUM_CLASSES), dtype=np.float32)

# accuracy
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.cast(labels, tf.int64)), tf.float32))

# loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predictions)

# optimizer
global_step   = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(TRAINING_LR_INITIAL, global_step, TRAINING_LR_EPOCHS*num_batches_train, TRAINING_LR_SCALE, staircase=TRAINING_LR_STAIRCASE)
update_ops    = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, TRAINING_MOMENTUM, use_nesterov=True).minimize(loss, global_step=global_step)

# saver
# saver = tf.train.Saver(max_to_keep=TRAINING_MAX_CHECKPOINTS)

# create a session
session = tf.Session()
    
# initialize global variables
session.run(tf.global_variables_initializer())

# cycle through the epochs
for epoch_index in range(TRAINING_NUM_EPOCHS):
    
    # train
    # initialize the iterator to the training dataset
    # cycle through the training batches
    # example, encoder, decoder, error, gradient computation and update
    session.run(iterator_init_train)
    for batch_index in range(num_batches_train):
        session.run(optimizer, feed_dict={train_state: True})

    # validate
    # initialize the iterator to the testing dataset
    # reset the accuracy statistics
    # cycle through the testing batches
    # example, encoder, decoder, accuracy
    session.run(iterator_init_test)
    num_correct = 0
    for batch_index in range(num_batches_test):
        num_correct_batch, predictions_batch    = session.run([accuracy, predictions], feed_dict={train_state: False})
        num_correct                            += num_correct_batch
        row_start                               = batch_index*TRAINING_BATCH_SIZE
        row_end                                 = (batch_index + 1)*TRAINING_BATCH_SIZE
        predictions_test[row_start:row_end, :]  = predictions_batch

    # display
    print('Epoch {0:3d}: top 1 accuracy on the test set is {1:5.2f} %'.format(epoch_index, (100.0*num_correct)/(TRAINING_BATCH_SIZE*num_batches_test)))

    # save
    # saver.save(session, TRAINING_CHECKPOINT_FILE.format(epoch_index))

# close the session
session.close()


################################################################################
#
# DISPLAY
#
################################################################################

# batch mean and standard deviation
batch_mean = np.array([DATA_MEAN_CHANNEL_0, DATA_MEAN_CHANNEL_1, DATA_MEAN_CHANNEL_2]).reshape((1, 1, 1, 3))
batch_std  = np.array([DATA_STD_DEV_CHANNEL_0, DATA_STD_DEV_CHANNEL_1, DATA_STD_DEV_CHANNEL_2]).reshape((1, 1, 1, 3))

# create a session
session = tf.Session()

# initialize global variables
session.run(tf.global_variables_initializer())

# initialize the test iterator
session.run(iterator_init_test)

# cycle through a few batches
for batch_index in range(1):
    
    # generate data and labels
    data_batch, labels_batch = session.run([data, labels])
    
    # normalize to [0, 1]
    data_batch = ((data_batch*batch_std) + batch_mean);
    
    # convert the final saved predictions to labels
    row_start          = batch_index*TRAINING_BATCH_SIZE
    row_end            = (batch_index + 1)*TRAINING_BATCH_SIZE
    predictions_labels = np.argmax(predictions_test[row_start:row_end, :], axis=1)
    
    # cycle through the images in the batch
    for image_index in range(TRAINING_BATCH_SIZE):
        
        # display the predicted label, actual label and image
        print('Predicted label: {0:1d} and actual label: {1:1d}'.format(predictions_labels[image_index], labels_batch[image_index]))
        plt.imshow(data_batch[image_index, :, :, :])
        plt.show()

# close the session
session.close()

