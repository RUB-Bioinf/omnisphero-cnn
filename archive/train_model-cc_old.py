'''Train a CNN to predict binary decisions on Omnisphero data.
Can be used for either neuron or oligo detection.

JOSHUA BUTKE, SEPTEMBER 2019
'''

# IMPORTS
#########
import numpy as np
import h5py
import os
import time
import math

from datetime import datetime

from tensorflow import keras

from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras import optimizers, regularizers
from keras.callbacks import Callback
from keras.callbacks import *
from keras.utils import *
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_curve, roc_auc_score, auc

from scramblePaths import *
from misc_omnisphero import *

from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import sys

vis_device_indexes = "1,2,3"
#vis_device_indexes = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = vis_device_indexes

# Custom Module
###############
import misc_omnisphero as misc

# =========== List of all available neuron experiments on SAS15 ============================================================================
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'

# =========== List of all available oligo experiments on SAS15 ============================================================================
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK95_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK153_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK155_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK156_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM10_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/MP70_trainingData_oligo/'

allNeurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'
]

allOligos = [
    # Defect experiments
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK95_trainingData_oligo/',

    # PRODI Paths
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ELS81_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK122_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK153_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK155_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK156_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/EKB5_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ESM9_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ESM10_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/MP66_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/MP67_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/MP70_trainingData_oligo/'

    # LOCAL PATHS
    # '/home/nilfoe/CNN/oligo/ELS81_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/JK122_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/JK153_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/JK155_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/JK156_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/EKB5_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/ESM9_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/ESM10_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/MP66_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/MP67_trainingData_oligo/',
    # '/home/nilfoe/CNN/oligo/MP70_trainingData_oligo/'
]

# FINAL NERON & OLIGO PATHS
finalNeurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/FJK130_trainingData_neuron/',

    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/JK96_trainingData_neuron/',

    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/JK122_trainingData_neuron/'
]

finalOligos = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    #'/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ELS81_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ESM9_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ESM10_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/JK95_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/JK122_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/JK155_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/JK156_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/MP66_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/MP67_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/MP70_trainingData_oligo/'
]

debugOligos = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/ELS81_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/EKB5_trainingData_oligo/'
]

debugNeurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/ESM9_trainingData_neuron/'
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK122_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK155_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK156_trainingData_oligo/',
]

testDataSet = '';


def gct():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


# CNN MODELS
############
def omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format):
    """prototype model for single class decision
    """
    # Input
    img_input = Input(shape=(input_height, input_width, input_depth), name='input_layer')

    # Convolutional Blocks (FEATURE EXTRACTION)

    # Conv Block 1
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=data_format)(img_input)
    bn1 = BatchNormalization(name='batch_norm_1')(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(bn1)
    bn2 = BatchNormalization(name='batch_norm_2')(c2)
    p1 = MaxPooling2D((2, 2), name='block1_pooling', data_format=data_format)(bn2)
    block1 = p1

    # Dave's Idee:
    # #Conv Block 1
    # c1 = Conv2D(32, (3,3), padding='same', name='block1_conv1', data_format=data_format)(img_input)
    # bn1 = BatchNormalization(name='batch_norm_1')(c1)
    # act1 = Activation('relu', alpha=0.0, max_value=None, threshold=0.0)(bn1)
    # 
    # c2 = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(act1)
    # bn2 = BatchNormalization(name='batch_norm_2')(c2)
    # p1 = MaxPooling2D((2,2), name='block1_pooling', data_format=data_format)(bn2)
    # block1 = p1

    #Idea to test out: kernel_initializer = 'he_uniform' for Conv2D nodes

    # Conv Block 2
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=data_format)(block1)
    bn3 = BatchNormalization(name='batch_norm_3')(c3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=data_format)(bn3)
    bn4 = BatchNormalization(name='batch_norm_4')(c4)
    p2 = MaxPooling2D((2, 2), name='block2_pooling', data_format='channels_last')(bn4)
    block2 = p2

    # Conv Block 3
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=data_format)(block2)
    bn5 = BatchNormalization(name='batch_norm_5')(c5)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=data_format)(bn5)
    bn6 = BatchNormalization(name='batch_norm_6')(c6)
    p3 = MaxPooling2D((2, 2), name='block3_pooling', data_format='channels_last')(bn6)
    block3 = p3

    # Conv Block 4
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=data_format)(block3)
    bn7 = BatchNormalization(name='batch_norm_7')(c7)
    c8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=data_format)(bn7)
    bn8 = BatchNormalization(name='batch_norm_8')(c8)
    p4 = MaxPooling2D((2, 2), name='block4_pooling', data_format='channels_last')(bn8)
    block4 = p4

    # Fully-Connected Block (CLASSIFICATION)
    flat = Flatten(name='flatten')(block3)
    fc1 = Dense(256, activation='relu', name='fully_connected1')(flat)
    drop_fc_1 = Dropout(0.5)(fc1)

    if n_classes == 1:
        prediction = Dense(n_classes, activation='sigmoid', name='output_layer')(drop_fc_1)
    else:
        prediction = Dense(n_classes, activation='softmax')(drop_fc_1)

    # Construction
    model = Model(inputs=img_input, outputs=prediction)

    return model


# ROC stuff
# Source: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
class plot_callback(Callback):
    def __init__(self, training_data, validation_data, file_handle,reduce_rate=0.5):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

        self.logs = []
        self.file_handle = file_handle

        self.batchCount = 0
        self.epochCount = 0
        self.reduce_rate = reduce_rate

    def on_train_begin(self, logs={}):
        self.file_handle.write('Training start.' + '\n')
        return

    def on_train_end(self, logs={}):
        self.file_handle.write('Training finished.' + '\n')
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epochCount = self.epochCount + 1
        return

    def on_epoch_end(self, epoch, logs={}):
        print('Timestamp: ', gct())
        return

    def on_batch_begin(self, batch, logs={}):
        self.batchCount = self.batchCount + 1
        self.file_handle.write('Beginning Batch: ' + str(self.batchCount) + '\n')
        return

    def on_batch_end(self, batch, logs={}):
        self.file_handle.write('Finished Batch: ' + str(self.batchCount) + '\n')
        return

# Initiating dummy variables
X = 0
y = 0
model = 0

#####################################################################

# SCRABLING
#################

scrambleResults = scramble_paths(path_candidate_list=debugOligos, validation_count=0, test_count=1)
# outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/results/roc_results_no81/'
outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/models/debug/stepsPerEpoch/'
#outPath = '/bph/home/nilfoe/Documents/CNN/results/neurons_final_softmax400/'

print('Saving results here: ' + outPath)
os.makedirs(outPath, exist_ok=True);

print('Timestamp: ', gct())
print('Sleep...')
time.sleep(10)

#####################################################################

# HYPERPARAMETERS
#################
batch_size = 100
n_classes = 1
input_height = 64
input_width = 64
input_depth = 3
data_format = 'channels_last'
optimizer_name = 'adadelta'
loss_enum = 'binary_crossentropy'
learn_rate = 0.0001
epochs = 25
# Erfahrung zeigt: 300 Epochen für Oligos, 400 für Neurons

#train_generator = datagenTrain.flow_from_directory(
#        augmentPathTrain,
#        target_size=(input_height, input_width),
#        color_mode='rgb',
#        batch_size=batch_size,
#        class_mode='binary')

genSeedTrain = 737856000
genSeedVal = genSeedTrain + 1

for n in range(len(scrambleResults)):
    # Remove previous iteration
    del X
    del y
    del model

    # AUGMENTATION
    datagen = ImageDataGenerator(
        #mode = "train"
        rotation_range=360,
        validation_split=0.5,
        brightness_range=[0.9,1.35],
        horizontal_flip=True,
        vertical_flip=True
        )

    scrambles = scrambleResults[n]
    label = scrambles['label']
    training_path_list = scrambles['train']
    validation_path_list = scrambles['val']
    test_path = scrambles['test']
    test_path = test_path[0]
    outPathCurrent = outPath + str(n) + '_' + label + os.sep
    os.makedirs(outPathCurrent, exist_ok=True);

    augmentPath = outPathCurrent + 'augments/'
    outImgPath = outPathCurrent + 'fig/'
    augmentPathTrain = augmentPath + 'train/'
    augmentPathVal = augmentPath + 'val/'
    os.makedirs(augmentPath, exist_ok=True);
    os.makedirs(augmentPathTrain, exist_ok=True);
    os.makedirs(augmentPathVal, exist_ok=True);
    os.makedirs(outImgPath, exist_ok=True);

    f = open(outPathCurrent + label + '_training_info.txt', 'w+')
    f.write('Label: ' + label + '\nTraining paths:\n')
    for i in range(len(training_path_list)):
        f.write(training_path_list[i] + '\n')
    f.write('\nValidation paths:\n')
    for i in range(len(validation_path_list)):
        f.write(validation_path_list[i] + '\n')
    f.write('\nTest path: ' + test_path + '\n')
    f.close()

    print('Round: ' + str(n + 1) + '/' + str(len(scrambleResults)) + ' -> ' + label)
    print('Writing results here: ' + outPathCurrent)
    print('Timestamp: ', gct())
    time.sleep(5)

    # TRAINING DATA
    ###############
    print("Traing data size: " + str(len(training_path_list)))
    X, y = misc.multiple_hdf5_loader(training_path_list)  # load datasets

    print(y.shape)
    if n_classes == 2:
        y = np.append(y, 1 - y, axis=1)
    print("Loaded data has shape: ")
    print(X.shape)
    print(y.shape)

    # # Loading temp data
    # data = np.load('/bph/puredata4/bioinfdata/work/omnisphero/CNN/temp/temp.npz')
    # X_pre = data.f.arr_0
    # y_pre = data.f.arr_1
    # data.close()
    # 
    # X = np.concatenate((X_pre, X), axis=0)
    # y = np.concatenate((y_pre, y), axis=0)
    # del X_pre, y_pre
    # # ==============================

    print("Correcting axes...")
    X = np.moveaxis(X, 1, 3)
    y = y.astype(np.int)
    print(X.shape)

    # np.savez('/bph/puredata4/bioinfdata/work/omnisphero/CNN/temp/temp2', X, y)

    print("Fitting data to generator.");
    X = misc.normalize_rgb_pixels(X)  # preprocess data
    datagen.fit(X)
    print("Done.");

    # VALIDATION DATA
    #################
    print("Validation data size: " + str(len(validation_path_list)))
    X_val, y_val = misc.multiple_hdf5_loader(validation_path_list)
    print(y_val.shape)
    if n_classes == 2:
        y_val = np.append(y_val, 1 - y_val, axis=1)
    #################

    print("Loaded validation data has shape: ")
    print(X_val.shape)
    print(y_val.shape)
    print("Correcting axes...")
    X_val = np.moveaxis(X_val, 1, 3)
    X_val = misc.normalize_rgb_pixels(X_val)
    y_val = y_val.astype(np.int)
    print(X_val.shape)
    print(y_val.shape)

    # CONSTRUCTION
    ##############

    print("Building model...")
    deviceCount = list(vis_device_indexes.replace(",",""))
    print("Devices index: " + vis_device_indexes + " -> " + str(deviceCount) + " GPU idexes used. Count: " + str(len(deviceCount)))
    steps_per_epoch = math.nan
    validation_steps = math.nan

    model = omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format)
    if len(deviceCount) > 1:
        model = multi_gpu_model(model, gpus=len(deviceCount))
        steps_per_epoch = len(X) / epochs
        validation_steps = len(y) / epochs
        print('Training Steps per epoch: ' + str(steps_per_epoch))
        print('Validation Steps per epoch: ' + str(validation_steps))
    model.compile(loss=loss_enum, optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    model.summary()
    print("Model output shape: ", model.output_shape)
    # plot_model(model, to_file=outPathCurrent + label + '_model.png', show_shapes=True, show_layer_names=True)
    f = open(outPathCurrent + label + '_model.txt', 'w+')

    f.write('Optimizer: SGD\n')
    f.write('Metrics: accuracy\n')
    f.write('GPUs used: ' + vis_device_indexes + '\n')
    f.write('Steps per epoch: ' + str(steps_per_epoch) + '\n')
    f.write('Loss: ' + loss_enum + '\n')
    f.write('Model shape: ' + str(model.output_shape) + '\n')
    f.write('Batch size: ' + str(batch_size) + '\n')
    f.write('Classes: ' + str(n_classes) + '\n')
    f.write('Input height: ' + str(input_height) + '\n')
    f.write('Input depth: ' + str(input_depth) + '\n')
    f.write('Data Format: ' + str(data_format) + '\n')
    f.write('Learn Rate: ' + str(learn_rate) + '\n')
    f.write('Epocs: ' + str(epochs) + '\n')
    f.close()

    # def myprint(s):
    #    with open(outPathCurrent + label + '_model.txt','a+') as f:
    #        print(s, file=f)
    # model.summary(print_fn=myprint)

    f = open(outPathCurrent + label + '_model.json', 'w+')
    f.write(model.to_json())
    f.close()
    model.save(outPathCurrent + label + '_model_init.h5')

    # class weighting
    from sklearn.utils.class_weight import compute_class_weight

    if n_classes == 1:
        y_order = y.reshape(y.shape[0])
        class_weights = compute_class_weight('balanced', np.unique(y), y_order)
        print("Class weights: ", class_weights)

    # CALLBACKS
    ###########
    csv_logger_filename = outPathCurrent + label + '_training_log.csv'
    if os.path.exists(csv_logger_filename):
        os.remove(csv_logger_filename)
    f = open(csv_logger_filename, 'w')
    f.write(gct()+"\n")
    f.close()

    print('Timestamp: ', gct())
    print('Reminder. Training for label: ' + label)
    f = open(outPathCurrent + label + '_training_progress.txt', 'w+')
    model_checkpoint = ModelCheckpoint(outPathCurrent + label + '_weights_best.h5', monitor='val_loss', verbose=1,save_best_only=True, mode='min')
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=80, verbose=0, mode='auto', min_delta=0.00000001, cooldown=0, min_lr=0)
    csv_logger = CSVLogger(filename = csv_logger_filename,separator=';',append=True)


    callbacks_list = [model_checkpoint,
                      lrCallBack,
                      csv_logger,
                      plot_callback(training_data=(X, y), validation_data=(X_val, y_val), file_handle=f)]


    # TRAINING
    ##########
    #history_all = model.fit(X, y,
    #                        validation_data=(X_val, y_val),
    #                        callbacks=callbacks_list,
    #                        epochs=epochs,
    #                        batch_size=batch_size,
    #                        class_weight=class_weights
    #                        )

    history_all = model.fit_generator(datagen.flow(
                                                X,y,
                                                batch_size=batch_size
                                                #save_to_dir=augmentPathTrain,
                                                #save_prefix='tr',
                                                ),
                            validation_data=(X_val, y_val),
                            #validation_data=datagen.flow(
                            #                    X_val,y_val,
                            #                    batch_size=batch_size,
                            #                    save_to_dir=augmentPathVal,
                            #                    save_prefix='vl',
                            #                    ),
                            callbacks=callbacks_list,
                            epochs=epochs,
                            #batch_size=batch_size,
                            class_weight=class_weights,
                            validation_steps=validation_steps,
                            steps_per_epoch=steps_per_epoch
                            )
    print('Done. Postprocessing and saving results.');

    history = [np.zeros((epochs, 4), dtype=np.float32)]
    history[0][:, 0] = history_all.history['loss']
    history[0][:, 1] = history_all.history['acc']
    history[0][:, 2] = history_all.history['val_loss']
    history[0][:, 3] = history_all.history['val_acc']
    f.close()

    # SAVING
    ########

    print('Timestamp: ', gct())

    model.save(outPathCurrent + label + '.h5')
    model.save_weights(outPathCurrent + label + '_weights_last.h5')
    print('Saved model: ' + outPathCurrent + label)

    print('Loading best weights again.')
    model.load_weights(outPathCurrent + label + '_weights_best.h5')

    # Validate the trained model.
    scores = model.evaluate(X_val, y_val, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # np.save(np.stack([history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']]),outPathCurrent + label + '_history.npy')

    # Plot training & validation accuracy values
    plt.plot(history_all.history['acc'])
    plt.plot(history_all.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')

    plt.savefig(outImgPath + label + '_accuracy.png',transparent=True)
    plt.savefig(outImgPath + label + '_accuracy.svg',transparent=True)
    plt.savefig(outImgPath + label + '_accuracy.pdf',transparent=True)
    plt.clf()
    print('Saved accuracy.')

    # Plot training & validation loss values
    plt.plot(history_all.history['loss'])
    plt.plot(history_all.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')

    plt.savefig(outImgPath + label + '_loss.png',transparent=True)
    plt.savefig(outImgPath + label + '_loss.svg',transparent=True)
    plt.savefig(outImgPath + label + '_loss.pdf',transparent=True)
    plt.clf()
    print('Saved loss.')

    # SAVING HISTORY
    np.save(outPathCurrent + label + "_history.npy", history)
    print('Saved history.')

    # SAVING ON MEMORY
    del X_val
    del X
    del y

    # TEST DATA
    #################
    print("Loading Test data")
    y_test = np.empty((0, 1))
    X_test, y_test = misc.hdf5_loader(test_path, gp_current=1, gp_max=1)
    y_test = np.asarray(y_test)
    y_test = y_test.astype(np.int)

    X_test = np.asarray(X_test)
    print(X_test.shape)
    X_test = np.moveaxis(X_test, 1, 3)
    X_test = misc.normalize_rgb_pixels(X_test)

    print("Loaded test data has shape: ")
    print(X_test.shape)
    print(y_test.shape)
    #################

    # ROC Curve
    print('Calculating roc curve...')
    try:
        y_pred_roc = model.predict(X_test)  # .ravel()
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred_roc)
        auc_roc = auc(fpr_roc, tpr_roc)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_roc, tpr_roc, label='ROC (area = {:.3f})'.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(outImgPath + label + '_roc.png',transparent=True)
        plt.savefig(outImgPath + label + '_roc.svg',transparent=True)
        plt.savefig(outImgPath + label + '_roc.pdf',transparent=True)
        plt.clf()

        np.save(outPathCurrent + label + "_roc_predictions.npy", y_pred_roc)
        np.savetxt(outPathCurrent + label + "_roc_predictions.csv", y_pred_roc, delimiter=';')

        # HISTOGRAM

        hist_pos = y_pred_roc[np.where(y_pred_roc > 0.5)]
        plt.hist(hist_pos, bins='auto')
        plt.title("Histogram: Positive")
        plt.savefig(outImgPath + label + '_histogram_1.png',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_1.svg',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_1.pdf',transparent=True)
        plt.clf()

        hist_neg = y_pred_roc[np.where(y_pred_roc <= 0.5)]
        plt.hist(hist_neg, bins='auto')
        plt.title("Histogram: Negative")
        plt.savefig(outImgPath + label + '_histogram_0.png',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_0.svg',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_0.pdf',transparent=True)
        plt.clf()

        plt.hist(y_pred_roc, bins='auto')
        plt.title("Histogram: All")
        plt.savefig(outImgPath + label + '_histogram_all.png',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_all.svg',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_all.pdf',transparent=True)
        plt.clf()

        plt.hist(label, bins='auto')
        plt.title("Histogram: All [Capped]")
        axes = plt.gca()
        plt.ylim(0, 2000)
        plt.xlim(0,1)
        plt.savefig(outImgPath + label + '_histogram_all2.png',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_all2.svg',transparent=True)
        plt.savefig(outImgPath + label + '_histogram_all2.pdf',transparent=True)
        plt.clf()
    except Exception as e:
        # Printing the exception message to file.
        print("Failed to calculate roc curve for: " + label + ".")
        f = open(outPathCurrent + label + "_rocError.txt", 'w+')
        f.write(str(e))

        try:
            # Printing the stack trace to the file
            exc_info = sys.exc_info()
            f.write('\n')
            f.write(str(exc_info))
        finally:
            pass

        f.close()

    del X_test
    del y_test
    del y_val

# END OF FILE
#############

print('Training done.')
print('Your results here: ' + outPath)
