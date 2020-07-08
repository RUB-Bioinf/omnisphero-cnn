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

import keras.backend as K
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger
from keras.layers import Dense
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
# Custom Module
###############
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

import misc_omnisphero as misc
from misc_omnisphero import *
from scramblePaths import *
from test_model import test_cnn

gpu_index_string = "2"
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

final_neurons_validated = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert/combinedVal_trainingData_neuron/',
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

final_oligos_validated = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert/combinedVal_trainingData_oligo/',
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
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/ESM9_trainingData_neuron/'
]

debug_oligos = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/ELS81_trainingData_oligo/'
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK122_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK155_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK156_trainingData_oligo/',
]

debug_oligos_validation = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/combinedVal_trainingData_oligo/'
]


# CNN MODELS
############
def omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format):
    """prototype model for single class decision
    """
    # Input
    img_input = Input(shape=(input_height, input_width, input_depth), name='input_layer')

    # Convolutional Blocks (FEATURE EXTRACTION)

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


def custom_loss_debug(y_true, y_pred):
    pass


def custom_loss_mse(y_true, y_pred):
    # shape of y_true -> (batch_size,units)
    # shape of y_pred -> (batch_size,units)

    loss = K.square(y_pred - y_true)
    loss = K.mean(loss, axis=1)

    return loss


# ROC stuff
# Source: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
class CustomCallback(Callback):
    def __init__(self, training_data, validation_data, file_handle, reduce_rate=0.5):
        super().__init__()
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

        self.logs = []

        self.batchCount = 0
        self.epochCount = 0
        self.reduce_rate = reduce_rate

    def on_train_begin(self, logs={}):
        f.write('Training start.' + '\n')
        return

    def on_train_end(self, logs={}):
        f.write('Training finished.' + '\n')
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epochCount = self.epochCount + 1
        return

    def on_epoch_end(self, epoch, logs={}):
        print('Timestamp: ', gct())
        return

    def on_batch_begin(self, batch, logs={}):
        self.batchCount = self.batchCount + 1
        f.write('Beginning Batch: ' + str(self.batchCount) + '\n')
        return

    def on_batch_end(self, batch, logs={}):
        f.write('Finished Batch: ' + str(self.batchCount) + '\n')
        return


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
out_path = '/bph/puredata3/work/sas15_mirror_nils/cnn/models/debug/custom_loss2/'

# Test Data Old
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_trainingData_oligo/'
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_trainingData_neuron/'

# Test Data Validated
test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'

print('Saving results here: ' + out_path)
os.makedirs(out_path, exist_ok=True)

#####################################################################

# HYPERPARAMETERS
#################
img_dpi_default = 550
batch_size = 100
n_classes = 1
lossEnum = 'binary_crossentropy'
data_format = 'channels_last'
optimizer_name = 'adadelta'
learn_rate = 0.0001
epochs = 5
# Erfahrung zeigt: 300 Epochen für Oligos, 400 für Neurons

# We want to train on 64x64x3 RGB images. Thus, our height, width and depth should be adjusted accordingly
input_height = 64
input_width = 64
input_depth = 3

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum = 4


def train_model_scrambling(path_candidate_list: [str], out_path: str, validation_count: int = 2, predict_count: int = 0,
                           test_data_path: str = test_data_path):
    scramble_results = scramble_paths(path_candidate_list=path_candidate_list, test_count=predict_count,
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

        print("Starting scrambling round: " + str(n) + " out of " + str(scramble_size))
        train_model(training_path_list=training_path_list,
                    validation_path_list=validation_path_list,
                    test_data_path=test_data_path,
                    out_path=out_path_current,
                    global_progress_current=n,
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
                data_gen: ImageDataGenerator = None, label: str = None):
    # Creating specific out dirs
    augment_path = out_path + 'augments' + os.sep
    fig_path = out_path + 'fig' + os.sep
    fig_path_model = fig_path + 'model' + os.sep
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(augment_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(fig_path_model, exist_ok=True)

    # Logging the directories used for training
    f = open(out_path + 'training_data_used.txt', 'w+')
    f.write('Current time: ' + gct())
    f.write('Training paths:\n')
    for i in range(len(training_path_list)):
        f.write(training_path_list[i] + '\n')
    f.write('\nValidation paths:\n')
    for i in range(len(validation_path_list)):
        f.write(validation_path_list[i] + '\n')
    f.close()

    # TRAINING DATA
    ###############
    print("Loading. Training data size: " + str(len(training_path_list)))
    X, y = misc.multiple_hdf5_loader(training_path_list, gp_current=global_progress_current, gp_max=global_progress_max,
                                     normalize_enum=normalize_enum)  # load datasets

    print(y.shape)
    if n_classes == 2:
        y = np.append(y, 1 - y, axis=1)
    print("Finished loading training data. Loaded data has shape: ")
    print("X-shape: " + str(X.shape))
    print("y-shape: " + str(y.shape))

    print("Correcting axes...")
    X = np.moveaxis(X, 1, 3)
    y = y.astype(np.int)
    print("X-shape (corrected): " + str(X.shape))

    print("Fitting X to the data-gen.")
    data_gen.fit(X)
    print("Done.")

    # VALIDATION DATA
    #################
    print("Validation data size: " + str(len(validation_path_list)))
    X_val, y_val = misc.multiple_hdf5_loader(validation_path_list, gp_current=global_progress_current,
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

    print("Compiling model...")
    # model.compile(loss=custom_loss_mse, optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    # TODO use dynamic params
    model.summary()
    print("Model output shape: ", model.output_shape)

    # Printing the model summary. To a file.
    # Yea, it's that complicated. Thanks keras... >.<
    orig_stdout = sys.stdout
    f = open(out_path + 'model_summary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    # plot_model(model, to_file=outPathCurrent + label + '_model.png', show_shapes=True, show_layer_names=True)
    f = open(out_path + 'model.txt', 'w+')

    f.write('Training time: ' + gct())
    if label is not None:
        f.write('Label: ' + label + '\n')
    f.write('Optimizer: SGD\n')
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

    log_out_path = out_path + 'training_log.csv'
    f = open(log_out_path, 'w+')
    f.write(gct() + '\nEpoch;Accuracy;Loss;??;Validation Accuracy; Validation Loss\n')
    f.close()

    # CALLBACKS
    ###########
    weights_best_filename = out_path + 'weights_best.h5'
    f = open(out_path + 'training_progress.txt', 'w+')
    model_checkpoint = ModelCheckpoint(weights_best_filename, monitor='val_loss', verbose=1,
                                       save_best_only=True, mode='min')
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=60, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)
    csv_logger = CSVLogger(log_out_path, separator=';', append=True)

    callbacks_list = [model_checkpoint,
                      lrCallBack,
                      csv_logger,
                      CustomCallback(training_data=(X, y), validation_data=(X_val, y_val), file_handle=f)]

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
    plot_training_history(history_all=history_all, fig_path=fig_path, epochs=epochs, img_dpi=img_dpi)

    # SAVING ON MEMORY
    del X_val
    del X
    del y

    # TEST DATA
    #################
    print("Loading best weights again to be tested.")
    model.load_weights(weights_best_filename)
    print("Finished loading weights.")

    try:
        test_cnn(out_path, test_data_path, normalize_enum, img_dpi, gpu_index_string, True, label='train-test')
    except Exception as e:
        print(gct() + " Failed to execute CNN TEST!")
        pass
        # TODO print stacktrace

    # END OF Training
    #############
    print('Training done.')
    print('Timestamp: ', gct())
    print('Your results here: ' + out_path)


def get_default_augmenter() -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=360,
        validation_split=0.0,
        # brightness_range=[1.0,1.0],
        horizontal_flip=True,
        vertical_flip=True
    )


def plot_training_history(history_all, fig_path, epochs, img_dpi=img_dpi_default):
    history = [np.zeros((epochs, 4), dtype=np.float32)]
    history[0][:, 0] = history_all.history['loss']
    history[0][:, 1] = history_all.history['acc']
    history[0][:, 2] = history_all.history['val_loss']
    history[0][:, 3] = history_all.history['val_acc']

    # SAVING HISTORY
    np.save(fig_path + "history.npy", history)
    print('Saved history.')

    # Plot training & validation accuracy values
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

    # Saving raw plot data
    print('Saving raw plot data.')
    f = open(fig_path + "plot_data_raw.csv", 'w+')
    f.write('Epoch;Accuracy;Validation Accuracy;Loss;Validation Loss\n')
    for i in range(epochs):
        f.write(
            str(i + 1) + ';' + str(history_all.history['acc'][i]) + ';' + str(history_all.history['val_acc'][i]) + ';' +
            str(history_all.history['loss'][i]) + ';' + str(history_all.history['val_loss'][i]) + ';' + str(
                history_all.history['acc'][i]) + ';\n')
    f.close()


def main():
    # AUGMENTATION
    data_gen = get_default_augmenter()

    train_model(
        training_path_list=debug_oligos,
        validation_path_list=debug_oligos_validation,
        test_data_path=test_data_path,
        data_gen=data_gen,
        out_path=out_path
    )


if __name__ == "__main__":
    main()
