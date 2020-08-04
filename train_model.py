'''Train a CNN to predict binary decisions on Omnisphero data.
Can be used for either neuron or oligo detection.

Nils Foerster
Joshua Butke
2019 - 2020
'''

# IMPORTS
#########
import math
import sys
import time

import socket
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend
from keras.backend import tensorflow_backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger, EarlyStopping
from keras.layers import Dense
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from keras.backend.tensorflow_backend import set_session

# Custom Imports
from misc_omnisphero import *
from scramblePaths import *
from test_model import test_cnn

# Keras session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
# Code snipit credit: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96

gpu_index_string = "0"
# gpuIndexString = "0,1,2"

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

all_neurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'
]

all_oligos = [
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
final_neurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/FJK130_trainingData_neuron/',

    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK96_trainingData_neuron/',

    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK122_trainingData_neuron/'
]

final_neurons_validated_validation_set = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_val/']

final_neurons_validated = [
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/ELS470_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/ESM49_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/FJK130_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/JK122_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/JK242_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/MP149_trainingData_neuron/'
]

final_neurons_adjusted_only = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_adjustedOnly/FJK130_trainingData_neuron/',

    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK96_trainingData_neuron/',

    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK122_trainingData_neuron/'
]

#########################################################################################################
#########################################################################################################
#########################################################################################################

final_oligos = [
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ELS81_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
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

final_oligos_validated_validation_set = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_val/']

final_oligos_validated = [
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/ELS470_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/ESM10_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/ESM49_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/ESM9_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK122_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK153_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK155_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK156_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK242_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/JK95_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/MP149_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/MP66_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/MP67_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/MP70_trainingData_oligo/'
]

final_oligos_adjusted_only = [
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo/ELS81_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/ESM9_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/ESM10_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/JK95_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/JK122_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/JK155_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/JK156_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/MP66_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/MP67_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_adjustedOnly/MP70_trainingData_oligo/'
]

debug_neurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/EKB5_trainingData_neuron/'
]

debug_oligos = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/EKB5_trainingData_oligo/'
]

debug_oligos_validation = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug_val/'
]


# CNN MODELS
############
def omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format):
    """prototype model for single class decision
    """
    # Input
    img_input = Input(shape=(input_height, input_width, input_depth), name='input_layer')

    # Convolution Blocks (FEATURE EXTRACTION)

    # Conv Block 1
    c1 = Conv2D(32, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv1',
                data_format=data_format)(img_input)
    bn1 = BatchNormalization(name='batch_norm_1')(c1)
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv2',
                data_format=data_format)(bn1)
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

    # Conv Block 2
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv1',
                data_format=data_format)(block1)
    bn3 = BatchNormalization(name='batch_norm_3')(c3)
    c4 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv2',
                data_format=data_format)(bn3)
    bn4 = BatchNormalization(name='batch_norm_4')(c4)
    p2 = MaxPooling2D((2, 2), name='block2_pooling', data_format='channels_last')(bn4)
    block2 = p2

    # Conv Block 3
    c5 = Conv2D(128, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv1',
                data_format=data_format)(block2)
    bn5 = BatchNormalization(name='batch_norm_5')(c5)
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv2',
                data_format=data_format)(bn5)
    bn6 = BatchNormalization(name='batch_norm_6')(c6)
    p3 = MaxPooling2D((2, 2), name='block3_pooling', data_format='channels_last')(bn6)
    block3 = p3

    # Conv Block 4
    c7 = Conv2D(256, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv1',
                data_format=data_format)(block3)
    bn7 = BatchNormalization(name='batch_norm_7')(c7)
    c8 = Conv2D(256, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv2',
                data_format=data_format)(bn7)
    bn8 = BatchNormalization(name='batch_norm_8')(c8)
    p4 = MaxPooling2D((2, 2), name='block4_pooling', data_format='channels_last')(bn8)
    block4 = p4

    # Fully-Connected Block (CLASSIFICATION)
    flat = Flatten(name='flatten')(block3)
    fc1 = Dense(256, kernel_initializer='he_uniform', activation='relu', name='fully_connected1')(flat)
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

# Custom Callback: Canary Interrupt
class CanaryInterruptCallback(Callback):

    def __init__(self, path: str, starts_active: bool = True, label: str = None):
        self.active: bool = starts_active
        self.label: str = label
        self.shutdown_source: bool = False
        os.makedirs(path, exist_ok=True)

        self.__canary_file = path + os.sep + 'canary_interrupt.txt'
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)

        f = open(self.__canary_file, 'w')
        f.write(
            'Canary interrupt for CNN training started at ' + gct() + '.\nDelete this file to safely stop your training.')
        if label is not None:
            f.write('\nLabel: ' + str(label).strip())
        f.write('\n\nCreated by Nils Förster.')
        f.close()

        print('Placed canary file here:' + str(self.__canary_file))

    def on_epoch_end(self, epoch, logs={}):
        if self.active:
            if not os.path.exists(self.__canary_file):
                print('Canary file not found! Shutting down training!')
                self.shutdown_source = True
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)


# Custom callback: Live Plotting
class PlotTrainingLiveCallback(Callback):
    # packages required: os, socket, matplotlib as plt

    def __init__(self, out_path, gpu_index_string):
        super().__init__()
        self.gpu_index_string = gpu_index_string
        self.file_name = out_path + 'training_custom_progress.csv'
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

        self.live_plot_dir = out_path + 'live_plot' + os.sep
        os.makedirs(self.live_plot_dir, exist_ok=True)

        self.epoch_start_timestamp = time.time()
        self.epoch_duration_list = []
        self.out_path = out_path
        self.host_name = str(socket.gethostname())

        self.epochCount = 0

        self.history_list_loss = []
        self.history_list_loss_val = []
        self.history_list_acc = []
        self.history_list_acc_val = []
        self.history_list_lr = []

    def on_train_begin(self, logs={}):
        self.write_line('Training start;;' + gct() + '\n')
        self.write_line('Epoch;Timestamp\n')

    def on_train_end(self, logs={}):
        self.write_line('Training finished;;' + gct())
        self.plot_training_time()
        self.plot_training_history_live()

    def on_epoch_begin(self, epoch, logs={}):
        self.epochCount = self.epochCount + 1
        self.epoch_start_timestamp = time.time()
        self.write_line()

    def on_epoch_end(self, epoch, logs={}):
        t = int(time.time() - self.epoch_start_timestamp)
        self.epoch_duration_list.append(t)

        self.history_list_loss.append(logs['loss'])
        self.history_list_loss_val.append(logs['val_loss'])
        self.history_list_acc.append(logs['acc'])
        self.history_list_acc_val.append(logs['val_acc'])
        self.history_list_lr.append(logs['lr'])

        self.plot_training_time()
        self.plot_training_history_live()

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def write_line(self, line=None):
        try:
            f = open(self.file_name, 'a')
            if line is None:
                line = str(self.epochCount) + ';' + gct()

            f.write(line + '\n')
            f.close()
        except Exception as e:
            # TODO print stacktrace
            pass

    def plot_training_time(self):
        # Plotting epoch duration
        try:
            plt.plot(self.epoch_duration_list)
            plt.title('Training on ' + self.host_name + ': ' + gct() + '. GPUS: [' + self.gpu_index_string + ']')
            plt.ylabel('Duration (sec)')
            plt.xlabel('Epoch')

            plt.savefig(self.out_path + 'training_time.png', dpi=400)
            plt.savefig(self.live_plot_dir + 'training_time_live.png', dpi=400)
            plt.savefig(self.live_plot_dir + 'training_time_live.svg', dpi=400, transparent=True)
            plt.savefig(self.live_plot_dir + 'training_time_live.pdf', dpi=400, transparent=True)
            plt.clf()
        except Exception as e:
            # TODO print stacktrace
            pass

    def plot_training_history_live(self):
        # Plotting epoch duration
        try:
            # Plot training & validation loss values
            plt.plot(self.history_list_loss)
            plt.plot(self.history_list_loss_val)
            plt.title('Live: Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='best')

            plt.savefig(self.live_plot_dir + 'loss_live.png', dpi=400)
            plt.savefig(self.live_plot_dir + 'loss_live.pdf', dpi=400, transparent=True)
            plt.savefig(self.live_plot_dir + 'loss_live.svg', dpi=400, transparent=True)
            plt.clf()

            # Plot accuracy loss values
            plt.plot(self.history_list_acc)
            plt.plot(self.history_list_acc_val)
            plt.title('Live: Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='best')

            plt.savefig(self.live_plot_dir + 'accuracy_live.png', dpi=400)
            plt.savefig(self.live_plot_dir + 'accuracy_live.svg', dpi=400, transparent=True)
            plt.savefig(self.live_plot_dir + 'accuracy_live.pdf', dpi=400, transparent=True)
            plt.clf()

            # Plot accuracy loss values
            plt.plot(self.history_list_lr)
            plt.title('Live: Model learn rate')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')

            plt.savefig(self.live_plot_dir + 'learn_rate_live.png', dpi=400)
            plt.savefig(self.live_plot_dir + 'learn_rate_live.svg', dpi=400, transparent=True)
            plt.savefig(self.live_plot_dir + 'learn_rate_live.pdf', dpi=400, transparent=True)
            plt.clf()
        except Exception as e:
            # TODO print stacktrace
            pass


# Initiating dummy variables
# X = 0
# y = 0
# model = 0
# TODO delete this?

#####################################################################

# SCRABLING
#################

# outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/results/roc_results_no81/'
# outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/models/debug-kontrolliert-weighted/neuron-n4-ep1500/'
# outPath = '/bph/home/nilfoe/Documents/CNN/results/neurons_final_softmax400/'

# outPath = '/bph/puredata3/work/sas15_mirror_nils/cnn/models/debug-kontrolliert-unweighted/neuron-n4-ep1500/'
out_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/'

# Test Data Old
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_trainingData_oligo/'
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_trainingData_neuron/'

# Test Data Validated
test_data_path_oligo = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test/'
test_data_path_neuron = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'

#####################################################################

# HYPERPARAMETERS
#################
img_dpi_default = 550
batch_size = 100
n_classes = 1
data_format = 'channels_last'
learn_rate = 0.0001
epochs = 2000
# Erfahrung zeigt: 300 Epochen für Oligos, 400 für Neurons

# We want to train on 64x64x3 RGB images. Thus, our height, width and depth should be adjusted accordingly
input_height = 64
input_width = 64
input_depth = 3

# Loss enum determines the loss function used during fitting.
# Possible entries:
# 'binary_crossentropy', 'mse'
lossEnum = 'binary_crossentropy'

# Optimizer name determines the optimizer used during fitting.
# Possible entries:
# 'adadelta', 'adam', 'SGD'
optimizer = SGD(lr=learn_rate)
# TODO actually use this


# Metrics name determines the metrics used during fitting.
# Possible entries:
# 'mean_sqaure_error, 'accuracy'
metrics = ['accuracy']
# TODO actually use this

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum = 4


def train_model_scrambling(path_candidate_list: [str], out_path: str, test_data_path: str, validation_count: int = 2):
    scramble_results = scramble_paths(path_candidate_list=path_candidate_list, test_count=0,
                                      validation_count=validation_count)

    scramble_size = len(scramble_results)
    for n in range(scramble_size):
        # Decoding the scrambling
        scrambles = scramble_results[n]
        label = scrambles['label']
        training_path_list = scrambles['train']
        validation_path_list = scrambles['val']

        out_path_current = out_path + str(n) + '_' + label + os.sep
        os.makedirs(out_path_current, exist_ok=True)

        print('Round: ' + str(n + 1) + '/' + str(len(scramble_results)) + ' -> ' + label)
        print('Writing results here: ' + out_path_current)
        print('Timestamp: ', gct())
        time.sleep(5)

        # AUGMENTATION
        data_gen = get_default_augmenter()

        print("Starting scrambling round: " + str(n + 1) + " out of " + str(scramble_size))
        train_model(training_path_list=training_path_list,
                    validation_path_list=validation_path_list,
                    test_data_path=test_data_path,
                    out_path=out_path_current,
                    global_progress_current=(n + 1),
                    global_progress_max=scramble_size, label=label, data_gen=data_gen)

    print("Finished high throughput training of " + str(scramble_size) + " models!")
    print(gct())


def train_model(training_path_list: [str], validation_path_list: [str], out_path: str,
                test_data_path: str,
                lossEnum: str = lossEnum, normalize_enum: int = normalize_enum, n_classes: int = n_classes,
                input_height: int = input_height,
                input_width: int = input_width, input_depth: int = input_depth, data_format: str = data_format,
                batch_size: int = batch_size, learn_rate: int = learn_rate,
                epochs: int = epochs, global_progress_current: int = 1, global_progress_max: int = 1,
                gpu_index_string: str = gpu_index_string, img_dpi: int = img_dpi_default,
                optimizer=optimizer, metrics=metrics, n_jobs: int = 1,
                data_gen: ImageDataGenerator = None, label: str = None, use_SMOTE: bool = False):
    # Creating specific out dirs
    print('Saving results here: ' + out_path)
    os.makedirs(out_path, exist_ok=True)

    augment_path = out_path + 'augments' + os.sep
    augment_smote_path = augment_path + 'smote' + os.sep
    fig_path = out_path + 'fig' + os.sep
    fig_path_model = fig_path + 'model' + os.sep

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(augment_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(fig_path_model, exist_ok=True)
    os.makedirs(augment_smote_path, exist_ok=True)

    # Logging the directories used for training
    f = open(out_path + 'training_data_used.txt', 'w+')
    f.write('Current time: ' + gct() + '\n')
    f.write('Training paths:\n')
    for i in range(len(training_path_list)):
        f.write(training_path_list[i] + '\n')
    f.write('\nValidation paths:\n')
    for i in range(len(validation_path_list)):
        f.write(validation_path_list[i] + '\n')
    f.write('\n\nTest data paths:\n' + test_data_path)
    f.close()

    # TRAINING DATA
    ###############
    print("Loading training data. Folder count: " + str(len(training_path_list)))
    X, y = multiple_hdf5_loader(training_path_list, gp_current=global_progress_current, gp_max=global_progress_max,
                                normalize_enum=normalize_enum)  # load datasets

    # print(y.shape)
    if n_classes == 2:
        y = np.append(y, 1 - y, axis=1)
    print("Finished loading training data. Loaded data has shape: ")
    print("X-shape: " + str(X.shape))
    print("y-shape: " + str(y.shape))

    print("Correcting axes...")
    X = np.moveaxis(X, 1, 3)
    y = y.astype(np.int)
    print("X-shape (corrected): " + str(X.shape))

    print('SMOTE mode: ' + str(use_SMOTE))
    n_samples, n_x, n_y, n_z = X.shape
    k_neighbors = max(int(n_samples / 1000), 150)
    smote_params: str = 'No SMOTE used'
    smote_error_text = 'N/A.'
    if use_SMOTE:
        smote_error_text = 'None. All went well.'
        smh = create_SMOTE_handler(n_jobs=n_jobs, k_neighbors=k_neighbors)
        smote_params = str(smh.get_params())
        smote_out_file = out_path + 'smote_progress.txt'
        f = open(smote_out_file, 'w')

        print('Starting SMOTE. Threads: ' + str(n_jobs) + '. ' + gct())
        try:
            X_smote = X.reshape(n_samples, n_x * n_y * n_z)
            y_smote = y.reshape(y.shape[0])

            f.write('Starting time: ' + gct() + '\n')
            f.write('Params: ' + str(smote_params) + '\n')
            f.write('X shape: ' + str(X.shape) + '\n')
            f.write('y shape: ' + str(y.shape) + '\n')
            f.write('Read class 0 count: ' + str(np.count_nonzero(y == 0)) + '\n')
            f.write('Read class 1 count: ' + str(np.count_nonzero(y == 1)) + '\n')
            f.write('Read samples: ' + str(n_samples) + '\n\n')

            X_smote, y_smote = smh.fit_sample(X_smote, y_smote)
            new_samples = X_smote.shape[0]

            f.write('Finished time: ' + gct() + '\n')
            f.write('X_smote shape: ' + str(X_smote.shape) + '\n')
            f.write('y_smote shape: ' + str(y_smote.shape) + '\n')
            f.write('New class 0 count: ' + str(np.count_nonzero(y_smote == 0)) + '\n')
            f.write('New class 1 count: ' + str(np.count_nonzero(y_smote == 1)) + '\n')
            f.write("New samples: " + str(new_samples) + '\n\n')

            X_smote = X_smote.reshape(new_samples, n_x, n_y, n_z)
            y_smote = y_smote.reshape(new_samples, 1)

            f.write('X_smote shape [reshaped]: ' + str(X_smote.shape) + '\n')
            f.write('y_smote shape [reshaped]: ' + str(y_smote.shape) + '\n')
        except Exception as e:
            # TODO display stacktrace
            smote_error_text = str(e.__class__.__name__) + ': "' + str(e) + '"'
            print('ERROR WHILE SMOTE!! (Reverting to un-smote)')
            print(smote_error_text)
            X_smote = X
            y_smote = y
            n_samples = np.nan
            new_samples = np.nan
            f.write('\nError! -> ' + smote_error_text)

        try:
            save_smote_samples(X_smote, y_smote, n_samples, new_samples, augment_smote_path,
                               out_samples=k_neighbors + 5)
        except Exception as e:
            # TODO display stacktrace
            print('Failed to save smote samples!')
            print(str(e))
            smote_error_text = smote_error_text + '\nSmote sample error: ' + str(e)
            f.write('\nError! -> ' + smote_error_text)

        f.close()

        X = X_smote
        y = y_smote
        del X_smote
        del y_smote

        print(
            'Finished smote. Old sample size: ' + str(n_samples) + '. New Samples: ' + str(new_samples) + '. ' + gct())
        print("Finished SMOTE. New data has shape: ")
        print("X-shape: " + str(X.shape))
        print("y-shape: " + str(y.shape))

        del n_samples
        del new_samples

    if data_gen is not None:
        print("Fitting X to the data-gen.")
        data_gen.fit(X)
        print("Done.")

    # VALIDATION DATA
    #################
    print("Loading validation data. Source folder count: " + str(len(validation_path_list)))
    X_val, y_val = multiple_hdf5_loader(validation_path_list, gp_current=global_progress_current,
                                        gp_max=global_progress_max,
                                        normalize_enum=normalize_enum)
    print("Validation data shape: " + str(y_val.shape))
    if n_classes == 2:
        y_val = np.append(y_val, 1 - y_val, axis=1)
    #################

    y_val_class1_size = len(y_val[y_val == 0])
    y_val_class2_size = len(y_val[y_val == 1])
    y_train_class1_size = len(y[y == 0])
    y_train_class2_size = len(y[y == 1])

    print("Loaded validation data has shape: ")
    print("X_val shape: " + str(X_val.shape))
    print("y_val shape: " + str(y_val.shape))

    print("Correcting axes...")
    X_val = np.moveaxis(X_val, 1, 3)
    # X_val = misc.normalize_RGB_pixels(X_val)
    y_val = y_val.astype(np.int)
    print("X_val corrected shape: " + str(X_val.shape))
    print("y_val corrected shape: " + str(y_val.shape))

    # CONSTRUCTION
    ##############
    steps_per_epoch = math.nan
    print("Building model...")
    gpu_indexes = list(gpu_index_string.replace(",", ""))
    gpu_index_count = len(gpu_indexes)
    print("Visible GPUs: '" + gpu_index_string + "'. Count: " + str(gpu_index_count))

    model = omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format)
    if gpu_index_count > 1:
        model = multi_gpu_model(model, gpus=gpu_index_count)
        steps_per_epoch = len(X) / epochs
        print("Model has been set up to run on multiple GPUs.")
        print("Steps per epoch: " + str(steps_per_epoch))

        print('WARNING! There seems to be an issue with multi gpu and batch validation size!!')
        # https://github.com/keras-team/keras/issues/11434#issuecomment-439832556

    print("Compiling model...")
    # model.compile(loss=custom_loss_mse, optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    model.compile(loss=lossEnum, optimizer=optimizer, metrics=metrics)
    # TODO use dynamic params
    model.summary()
    print("Model output shape: ", model.output_shape)
    print("Model metric names: " + str(model.metrics_names))

    # Printing the model summary. To a file.
    # Yea, it's that complicated. Thanks keras... >.<
    orig_stdout = sys.stdout
    f = open(out_path + 'model_summary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    # plot_model(model, to_file=outPathCurrent + label + '_model.png', show_shapes=True, show_layer_names=True)
    f = open(out_path + 'model_training.txt', 'w+')
    data_gen_description = 'None.'
    if data_gen is not None:
        data_gen_description = 'Used: ' + str(data_gen)

    f.write('Training time: ' + gct() + '\n')
    if label is not None:
        f.write('Label: ' + label + '\n')
    f.write('Loss: ' + lossEnum + '\n')
    f.write('GPUs: ' + gpu_index_string + '\n')
    f.write('Steps per epoch: ' + str(steps_per_epoch) + '\n')
    f.write('Model shape: ' + str(model.output_shape) + '\n')
    f.write('Batch size: ' + str(batch_size) + '\n')
    f.write('Classes: ' + str(n_classes) + '\n')
    f.write('Input height: ' + str(input_height) + '\n')
    f.write('Input depth: ' + str(input_depth) + '\n')
    f.write('Data Format: ' + str(data_format) + '\n')
    f.write('Learn Rate: ' + str(learn_rate) + '\n')
    f.write('Epochs: ' + str(epochs) + '\n')
    f.write('Normalization mode: ' + str(normalize_enum) + '\n')
    f.write('Model metrics: ' + str(model.metrics_names) + '\n')
    f.write('Model optimizer: ' + str(optimizer) + '\n')
    f.write('Model metrics raw: ' + str(metrics) + '\n')
    f.write('Data Generator used: ' + data_gen_description + '\n')
    f.write('SMOTE Parameters: ' + str(smote_params) + '\n')
    f.write('SMOTE Error: ' + smote_error_text + '\n')

    f.write('\n == DATA: ==\n')
    f.write("X shape: " + str(X.shape) + '\n')
    f.write("y shape: " + str(y.shape) + '\n')
    f.write("X_val shape: " + str(X_val.shape) + '\n')
    f.write("y_val shape: " + str(y_val.shape) + '\n')

    f.close()

    f = open(out_path + 'model.json', 'w+')
    f.write(model.to_json())
    f.close()

    # TODO make weighting optional
    # class weighting
    f = open(out_path + 'class_weights.csv', 'w+')
    f.write(';Validation;Training\n')
    f.write('Class 0 count;' + str(y_val_class1_size) + ';' + str(y_train_class1_size) + '\n')
    f.write('Class 1 count;' + str(y_val_class2_size) + ';' + str(y_train_class2_size) + '\n')
    f.write('All count;' + str(y_val_class1_size + y_val_class2_size) + ';' + str(
        y_train_class1_size + y_train_class2_size) + '\n')
    f.write('Class Ratio;' + str(y_val_class2_size / y_val_class1_size) + ';' + str(
        y_train_class2_size / y_train_class1_size) + '\n')
    f.write('1:x Ratio;' + str(y_val_class1_size / y_val_class2_size) + ';' + str(
        y_train_class1_size / y_train_class2_size) + '\n\n')

    f.write('Number classes;' + str(n_classes) + '\n')
    class_weights = np.asarray([1, 1])
    if n_classes == 1:
        weights_aim = 'balanced'
        y_order = y.reshape(y.shape[0])
        class_weights = compute_class_weight(weights_aim, np.unique(y), y_order)
        print("Class weights: ", class_weights)

        f.write('Weight aim;' + weights_aim + '\n')
        f.write('Weights Class 0;' + str(class_weights[0]) + '\n')
        f.write('Weights Class 1;' + str(class_weights[1]) + '\n')
    f.close()

    checkpoint_out_path = out_path + 'checkpoints' + os.sep
    os.makedirs(checkpoint_out_path, exist_ok=True)

    log_out_path = out_path + 'training_log.csv'
    f = open(log_out_path, 'w+')
    f.write(gct() + '\nEpoch;Accuracy;Loss;??;Validation Accuracy; Validation Loss\n')
    f.close()

    # CALLBACKS
    ###########
    learn_rate_reduction_patience = 110
    learn_rate_factor = 0.5
    es_patience = int(float(learn_rate_reduction_patience) * 2.1337)
    print('Learn rate reduction by factor ' + str(learn_rate_factor) + ' if improvement within ' + str(
        learn_rate_reduction_patience) + ' epochs.')
    print('Early stopping patience: ' + str(es_patience))

    weights_best_filename = out_path + 'weights_best.h5'
    model_checkpoint = ModelCheckpoint(checkpoint_out_path + 'weights_ep{epoch:08d}.h5', verbose=1,
                                       save_weights_only=True, period=50)
    model_checkpoint_best = ModelCheckpoint(weights_best_filename, monitor='val_loss', verbose=1, save_best_only=True,
                                            mode='min')
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=learn_rate_factor, patience=learn_rate_reduction_patience,
                                   verbose=1,
                                   mode='auto', min_delta=0.000001, cooldown=0, min_lr=0.000001)
    csv_logger = CSVLogger(log_out_path, separator=';', append=True)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1, mode='auto', baseline=None,
                                        restore_best_weights=True)  # early stopping
    canary_interrupt_callback = CanaryInterruptCallback(path=out_path)
    live_plot_callback = PlotTrainingLiveCallback(out_path=out_path, gpu_index_string=gpu_index_string)

    callbacks_list = [model_checkpoint,
                      model_checkpoint_best,
                      lrCallBack,
                      csv_logger,
                      # early_stop_callback,
                      canary_interrupt_callback,
                      live_plot_callback
                      ]

    # TRAINING
    ##########
    if label is not None:
        print('Reminder. Training for label: ' + label)
    print('Saving model here: ' + out_path)
    print('Training started: ' + gct())

    # Setting the os cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_string

    # Checking if a data generator exists. If so, datagen mode will be used. If not, classic training.
    history_all = None
    if data_gen is None:
        print('Fitting model without a data gen!')
        history_all = model.fit(X, y,
                                validation_data=(X_val, y_val),
                                callbacks=callbacks_list,
                                epochs=epochs,
                                batch_size=batch_size,
                                class_weight=class_weights
                                )
    else:
        print('Fitting model and using a data gen!')
        history_all = model.fit_generator(data_gen.flow(
            X, y,
            batch_size=batch_size,
            # save_to_dir=augment_path,
            # save_prefix='aug'
        ),
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            epochs=epochs,
            # batch_size=batch_size,
            # class_weight=class_weights,
            steps_per_epoch=len(X) / epochs
        )

    # SAVING
    ########
    print("Saving history & plots to disc: " + out_path)
    print('Timestamp: ', gct())

    model.save(out_path + 'model.h5')
    model.save_weights(out_path + 'weights.h5')
    print('Saved model: ' + out_path + 'model.h5')

    # Saving plots
    try:
        plot_training_history(history_all=history_all, fig_path=fig_path, img_dpi=img_dpi)
    except Exception as e:
        print(gct() + " Failed plot history!!")
        # TODO print stacktrace

    # SAVING ON MEMORY
    del X_val
    del X
    del y
    del y_val
    del model

    # TEST DATA
    #################
    # Not needed anymore, since testing has been outsourced into its own function
    # print("Loading best weights again to be tested.")
    # model.load_weights(weights_best_filename)
    # print("Finished loading weights.")

    try:
        print('Test started')
        test_cnn(out_path, test_data_path, normalize_enum, img_dpi, gpu_index_string, True, label='train-test')
        print('Test finished')
    except Exception as e:
        print(gct() + " Failed to execute CNN TEST!")
        # TODO print stacktrace

    # END OF Training
    #############
    print('Training done.')
    print('Timestamp: ', gct())
    print('Your results here: ' + out_path)


def save_smote_samples(X_smote, y_smote, n_samples, new_samples, augment_smote_path, out_samples: int = 5):
    print('Saving smote samples: ' + augment_smote_path)
    smote_indexes = list(range(out_samples))
    smote_indexes.extend(range(n_samples - out_samples, n_samples + out_samples))
    smote_indexes.extend(range(new_samples - out_samples * 2, new_samples))

    class1_indices = np.where(y_smote == 1)[0]

    for i in range(len(smote_indexes)):
        current_index = smote_indexes[i]
        current_img = X_smote[current_index]

        current_img_path = augment_smote_path + 'smote_' + str(i) + '_' + str(y_smote[current_index][0]) + '.png'
        plt.imsave(current_img_path, current_img)

    out_image_bounds = int((math.sqrt(out_samples) + 1) * 2.1337)
    combined_sample_count = out_image_bounds * out_image_bounds
    combined_img = np.zeros((out_image_bounds * 64, out_image_bounds * 64, 3), dtype='uint8')
    y = -1
    x = -1
    for i in range(new_samples - combined_sample_count, new_samples):
        x = (x + 1) % out_image_bounds
        if x == 0:
            y = y + 1
        print(x)
        print(y)
        current_img = (X_smote[i] * 255).astype(np.uint8)
        combined_img[x * 64:x * 64 + 64, y * 64:y * 64 + 64] = current_img
    plt.imsave(augment_smote_path + 'smote_last_' + str(out_image_bounds) + 'x' + str(out_image_bounds) + '.png',
               combined_img)


def get_default_augmenter() -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=360,
        validation_split=0.0,
        # brightness_range=[1.0,1.0],
        horizontal_flip=True,
        vertical_flip=True
    )


def plot_training_history(history_all, fig_path, img_dpi=img_dpi_default):
    epochs = len(history_all.epoch)
    hist_key_set = history_all.history.keys()
    history = [np.zeros((epochs, len(hist_key_set)), dtype=np.float32)]

    # Plot training & validation accuracy values
    i = 0
    out_file_header = "Epoch;"

    for hist_key in hist_key_set:
        label = decode_history_key(hist_key)
        out_file_header = out_file_header + label + ";"
        history[0][:, i] = history_all.history[hist_key]

        plt.plot(history_all.history[hist_key])
        plt.title(label)
        plt.ylabel(label)
        plt.xlabel('Epoch')

        plt.savefig(fig_path + 'raw_' + hist_key + '.png', dpi=img_dpi)
        plt.savefig(fig_path + 'raw_' + hist_key + '.svg', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'raw_' + hist_key + '.pdf', dpi=img_dpi, transparent=True)
        plt.clf()

        i = i + 1
        print('Saved raw data for: ' + hist_key + ' [' + label + '].')

        val_hist_key = 'val_' + hist_key
        if val_hist_key in hist_key_set:
            val_label = decode_history_key(val_hist_key)

            plt.plot(history_all.history[hist_key])
            plt.plot(history_all.history[val_hist_key])
            plt.title('Model ' + label)
            plt.ylabel(label)
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='best')

            plt.savefig(fig_path + 'val_' + hist_key + '.png', dpi=img_dpi)
            plt.savefig(fig_path + 'val_' + hist_key + '.svg', dpi=img_dpi, transparent=True)
            plt.savefig(fig_path + 'val_' + hist_key + '.pdf', dpi=img_dpi, transparent=True)
            plt.clf()
            print('Saved combined validation: ' + label + ' & ' + val_label)

    if 'acc' in hist_key_set and 'val_acc' in hist_key_set:
        plt.plot(history_all.history['acc'])
        plt.plot(history_all.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')

        plt.savefig(fig_path + 'accuracy.png', dpi=img_dpi)
        plt.savefig(fig_path + 'accuracy.svg', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'accuracy.pdf', dpi=img_dpi, transparent=True)
        plt.clf()
        print('Saved accuracy.')

    if 'loss' in hist_key_set and 'val_loss' in hist_key_set:
        # Plot training & validation loss values
        plt.plot(history_all.history['loss'])
        plt.plot(history_all.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='best')

        plt.savefig(fig_path + 'loss.png', dpi=img_dpi)
        plt.savefig(fig_path + 'loss.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'loss.svg', dpi=img_dpi, transparent=True)
        plt.clf()
        print('Saved loss.')

    # SAVING HISTORY
    np.save(fig_path + "history.npy", history)
    print('Saved history.')

    # Saving raw plot data
    print('Saving raw plot data.')
    f = open(fig_path + "plot_data_raw.csv", 'w+')
    f.write(out_file_header + '\n')
    for i in range(epochs):
        out_line = str(i + 1) + ';'
        for hist_key in hist_key_set:
            try:
                out_line = out_line + str(history_all.history[hist_key][i]) + ';'
            except Exception as e:
                # TODO print stacktrace
                out_line = 'Error'
    f.close()


def decode_history_key(key: str) -> str:
    if key == 'lr':
        return "Learn Rate"
    if key == 'acc':
        return "Accuracy"
    if key == 'mean_squared_error':
        return "Mean Squared Error"
    if key == 'val_mean_squared_error':
        return "Validation Mean Squared Error"
    if key == 'loss':
        return "Loss"
    if key == 'val_acc':
        return "Validation Accuracy"
    if key == 'val_loss':
        return "Validation Loss"

    print("Warning! Key: " + key + " has no extended figure label!")
    return key


def main():
    # AUGMENTATION
    data_gen = get_default_augmenter()

    out_path_base = out_path + 'paper-final_no-datagen' + os.sep
    out_path_oligo = out_path_base + 'oligo' + os.sep
    out_path_neuron = out_path_base + 'neuron' + os.sep

    out_path_oligo_debug = out_path_base + 'oligo_debug' + os.sep

    oligo_mode = True
    neuron_mode = False
    debug_mode = False
    n_jobs = 1

    print('Sleeping....')
    # time.sleep(18000)

    if debug_mode:
        train_model(
            training_path_list=debug_oligos,
            validation_path_list=debug_oligos_validation,
            test_data_path=test_data_path_oligo,
            # data_gen=data_gen,
            # out_path=out_path_oligo_debug,
            out_path=out_path + 'paper-debug-smote' + os.sep + 'oligo-debug' + os.sep,
            use_SMOTE=True,
            n_jobs=19,
            gpu_index_string="2",
            epochs=5
        )

    if oligo_mode:
        train_model(
            training_path_list=final_oligos_validated,
            validation_path_list=final_oligos_validated_validation_set,
            test_data_path=test_data_path_oligo,
            # data_gen=data_gen,
            use_SMOTE=True,
            out_path=out_path + 'paper-final_smote_no-datagen' + os.sep + 'oligo' + os.sep,
            n_jobs=n_jobs,
            gpu_index_string="0",
            epochs=2500
        )

    if neuron_mode:
        train_model(
            training_path_list=final_neurons_validated,
            validation_path_list=final_neurons_validated_validation_set,
            test_data_path=test_data_path_neuron,
            # data_gen=data_gen,
            use_SMOTE=True,
            out_path=out_path + 'paper-final_smote_no-datagen' + os.sep + 'neuron' + os.sep,
            n_jobs=n_jobs,
            gpu_index_string="1",
            epochs=2500
        )

    # out_path_oligo = out_path+'oligo'+os.sep
    # out_path_neuron = out_path+'neuron'+os.sep


#
# train_model_scrambling(path_candidate_list=final_oligos_validated,
#                       test_data_path=test_data_path_oligo,
#                       out_path=out_path_oligo,
#                       validation_count=1)
# train_model_scrambling(path_candidate_list=final_neurons_validated,
#                       test_data_path=test_data_path_neuron,
#                       out_path=out_path_neuron,
#                       validation_count=1)


if __name__ == "__main__":
    main()
