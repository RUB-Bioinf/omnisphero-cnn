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
from keras.utils import plot_model

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
from sklearn.metrics import *

from scramblePaths import *
from misc_omnisphero import *
from sklearn.utils.class_weight import compute_class_weight

from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import sys

gpuIndexString = "1"
# gpuIndexString = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuIndexString

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

    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK96_trainingData_neuron/',

    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron/JK122_trainingData_neuron/'
]

finalNeuronsKontrolliert = [
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

finalNeuronsAdjustedOnly = [
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

finalOligos = [
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

finalOligosKontrolliert = [
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

finalOligosAdjustedOnly = [
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

debugNeurons = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/combinedVal_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_debug/ESM9_trainingData_neuron/'
]

debugOligos = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/combinedVal_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_debug/ELS81_trainingData_oligo/'
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK122_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK155_trainingData_oligo/',
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo/JK156_trainingData_oligo/',
]


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
class plot_callback(Callback):
    def __init__(self, training_data, validation_data, file_handle, reduce_rate=0.5):
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
X = 0
y = 0
model = 0

#####################################################################

# SCRABLING
#################

scrambleResults = scramblePaths(pathCandidateList=finalNeuronsKontrolliert, validation_count=0, predict_count=1)
# outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/results/roc_results_no81/'
outPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/models/debug-kontrolliert-weighted/neuron-n4-ep1500/'
# outPath = '/bph/home/nilfoe/Documents/CNN/results/neurons_final_softmax400/'

#Test Data Old
#testDataPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_trainingData_oligo/'
#testDataPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_trainingData_neuron/'

#Test Data Kontrolliert
#testDataPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test/'
testDataPath = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'

print('Saving results here: ' + outPath)
os.makedirs(outPath, exist_ok=True)

print('Timestamp: ', gct())
print('sleeep...')
time.sleep(4)

#####################################################################

# HYPERPARAMETERS
#################
batch_size = 100
n_classes = 1
input_height = 64
input_width = 64
input_depth = 3
lossEnum = 'binary_crossentropy'
data_format = 'channels_last'
optimizer_name = 'adadelta'
learn_rate = 0.0001
epochs = 1500
# Erfahrung zeigt: 300 Epochen für Oligos, 400 für Neurons

normalize_enum = 4
# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well

img_dpi = 450
genSeedTrain = 737856000
genSeedVal = genSeedTrain + 1

for n in range(len(scrambleResults)):
    # Remove previous iteration
    del X
    del y
    del model

    # AUGMENTATION
    datagen = ImageDataGenerator(
        rotation_range=360,
        validation_split=0.0,
        # brightness_range=[1.0,1.0],
        horizontal_flip=True,
        vertical_flip=True
    )

    scrambles = scrambleResults[n]
    label = scrambles['label']
    training_path_list = scrambles['train']
    validation_path_list = scrambles['val']
    outPathCurrent = outPath + str(n) + '_' + label + os.sep
    os.makedirs(outPathCurrent, exist_ok=True)

    augmentPath = outPathCurrent + 'augments' + os.sep
    figPath = outPathCurrent + 'fig' + os.sep
    figPathModel = figPath + 'model' + os.sep
    os.makedirs(augmentPath, exist_ok=True)
    os.makedirs(figPath, exist_ok=True)
    os.makedirs(figPathModel, exist_ok=True)

    f = open(outPathCurrent + 'training_info.txt', 'w+')
    f.write('Label: ' + label + '\nTraining paths:\n')
    for i in range(len(training_path_list)):
        f.write(training_path_list[i] + '\n')
    f.write('\nValidation paths:\n')
    for i in range(len(validation_path_list)):
        f.write(validation_path_list[i] + '\n')
    f.close()

    print('Round: ' + str(n + 1) + '/' + str(len(scrambleResults)) + ' -> ' + label)
    print('Writing results here: ' + outPathCurrent)
    print('Timestamp: ', gct())
    time.sleep(5)

    # TRAINING DATA
    ###############
    print("Traing data size: " + str(len(training_path_list)))
    X, y = misc.multiple_hdf5_loader(training_path_list, gpCurrent=n, gpMax=len(scrambleResults),
                                     normalize_enum=normalize_enum)  # load datasets

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

    # X = misc.normalize_RGB_pixels(X)  # preprocess data
    datagen.fit(X)

    # VALIDATION DATA
    #################
    print("Validation data size: " + str(len(validation_path_list)))
    X_val, y_val = misc.multiple_hdf5_loader(validation_path_list, gpCurrent=n, gpMax=len(scrambleResults),
                                             normalize_enum=normalize_enum)
    print(y_val.shape)
    if n_classes == 2:
        y_val = np.append(y_val, 1 - y_val, axis=1)
    #################

    y_val_class1_size = len(y_val[y_val == 0])
    y_val_class2_size = len(y_val[y_val == 1])
    y_train_class1_size = len(y[y == 0])
    y_train_class2_size = len(y[y == 1])

    print("Loaded validation data has shape: ")
    print(X_val.shape)
    print(y_val.shape)
    print("Correcting axes...")
    X_val = np.moveaxis(X_val, 1, 3)
    # X_val = misc.normalize_RGB_pixels(X_val)
    y_val = y_val.astype(np.int)
    print(X_val.shape)
    print(y_val.shape)

    # CONSTRUCTION
    ##############

    steps_per_epoch = math.nan
    print("Building model...")
    gpuIndexes = list(gpuIndexString.replace(",", ""))
    gpuIndexCount = len(gpuIndexes)
    print("Visible GPUs: '" + gpuIndexString + "'. Count: " + str(gpuIndexCount))

    model = omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format)
    if gpuIndexCount > 1:
        model = multi_gpu_model(model, gpus=gpuIndexCount)
        steps_per_epoch = len(X) / epochs
        print("Model has been set up to run on multiple GPUs.")
        print("Steps per epoch: " + str(steps_per_epoch))

    model.compile(loss=lossEnum, optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    model.summary()
    print("Model output shape: ", model.output_shape)

    orig_stdout = sys.stdout
    f = open(outPathCurrent + 'model_summary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    # plot_model(model, to_file=outPathCurrent + label + '_model.png', show_shapes=True, show_layer_names=True)
    f = open(outPathCurrent + 'model.txt', 'w+')

    f.write('Label: ' + label + '\n')
    f.write('Optimizer: SGD\n')
    f.write('Loss: ' + lossEnum + '\n')
    f.write('GPUs: ' + gpuIndexString + '\n')
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

    # def myprint(s):
    #    with open(outPathCurrent + label + '_model.txt','a+') as f:
    #        print(s, file=f)
    # model.summary(print_fn=myprint)

    f = open(outPathCurrent + 'model.json', 'w+')
    f.write(model.to_json())
    f.close()

    # class weighting

    f = open(outPathCurrent + 'class_weights.csv', 'w+')
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

    logOutPath = outPathCurrent + 'training_log.csv'
    f = open(logOutPath, 'w+')
    f.write(gct() + '\nEpoch;Accuracy;Loss;??;Validation Accuracy; Validation Loss\n')
    f.close()

    # CALLBACKS
    ###########

    weights_best_filename = outPathCurrent + 'weights_best.h5'
    print('Timestamp: ', gct())
    print('Reminder. Training for label: ' + label)
    print('Saving model here: ' + outPathCurrent)
    f = open(outPathCurrent + 'training_progress.txt', 'w+')
    model_checkpoint = ModelCheckpoint(weights_best_filename, monitor='val_loss', verbose=1,
                                       save_best_only=True, mode='min')
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=60, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)
    csv_logger = CSVLogger(logOutPath, separator=';', append=True)

    callbacks_list = [model_checkpoint,
                      lrCallBack,
                      csv_logger,
                      plot_callback(training_data=(X, y), validation_data=(X_val, y_val), file_handle=f)]

    # TRAINING
    ##########
    # history_all = model.fit(X, y,
    #                        validation_data=(X_val, y_val),
    #                        callbacks=callbacks_list,
    #                        epochs=epochs,
    #                        batch_size=batch_size,
    #                        class_weight=class_weights
    #                        )

    history_all = model.fit_generator(datagen.flow(
        X, y,
        batch_size=batch_size,
        # save_to_dir=augmentPath,
        # save_prefix='aug'
    ),
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        epochs=epochs,
        # batch_size=batch_size,
        class_weight=class_weights,
        steps_per_epoch=len(X) / epochs
    )

    history = [np.zeros((epochs, 4), dtype=np.float32)]
    history[0][:, 0] = history_all.history['loss']
    history[0][:, 1] = history_all.history['acc']
    history[0][:, 2] = history_all.history['val_loss']
    history[0][:, 3] = history_all.history['val_acc']

    # SAVING
    ########

    print('Timestamp: ', gct())

    model.save(outPathCurrent + 'model.h5')
    model.save_weights(outPathCurrent + 'weights.h5')
    print('Saved model: ' + outPathCurrent + 'model.h5')

    print('Loading best weights again.')
    model.load_weights(weights_best_filename)

    # Validate the trained model.
    # print('Evaluating trained data...')
    # scores = model.evaluate(X_val, y_val, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

    # np.save(np.stack([history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']]),outPathCurrent + label + '_history.npy')

    # Plot training & validation accuracy values
    plt.plot(history_all.history['acc'])
    plt.plot(history_all.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')

    plt.savefig(figPath + 'accuracy.png', dpi=img_dpi)
    plt.savefig(figPath + 'accuracy.svg', dpi=img_dpi, transparent=True)
    plt.savefig(figPath + 'accuracy.pdf', dpi=img_dpi, transparent=True)
    plt.clf()
    print('Saved accuracy.')

    # Plot training & validation loss values
    plt.plot(history_all.history['loss'])
    plt.plot(history_all.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')

    plt.savefig(figPath + 'loss.png', dpi=img_dpi)
    plt.savefig(figPath + 'loss.pdf', dpi=img_dpi, transparent=True)
    plt.savefig(figPath + 'loss.svg', dpi=img_dpi, transparent=True)
    plt.clf()
    print('Saved loss.')

    # Saving raw plot data
    print('Saving raw plot data.')
    f = open(figPath + "plot_data_raw.csv", 'w+')
    f.write('Epoch;Accuracy;Validation Accuracy;Loss;Validation Loss\n')
    for i in range(epochs):
        f.write(
            str(i + 1) + ';' + str(history_all.history['acc'][i]) + ';' + str(history_all.history['val_acc'][i]) + ';' +
            str(history_all.history['loss'][i]) + ';' + str(history_all.history['val_loss'][i]) + ';' + str(
                history_all.history['acc'][i]) + ';\n')
    f.close()

    # SAVING HISTORY
    np.save(outPathCurrent + "history.npy", history)
    print('Saved history.')

    # SAVING ON MEMORY
    del X_val
    del X
    del y

    # TEST DATA
    #################
    print('Loading Test data: ' + testDataPath)
    y_test = np.empty((0, 1))
    X_test, y_test = misc.hdf5_loader(testDataPath, gpCurrent=1, gpMax=1, normalize_enum=normalize_enum)
    print('Done. Preprocessing test data.')
    y_test = np.asarray(y_test)
    y_test = y_test.astype(np.int)

    X_test = np.asarray(X_test)
    print(X_test.shape)
    X_test = np.moveaxis(X_test, 1, 3)
    # X_test = misc.normalize_RGB_pixels(X_test)

    print("Loaded test data has shape: ")
    print(X_test.shape)
    print(y_test.shape)
    #################

    # ROC Curve
    try:
        print('Trying to predict test data')
        y_pred_roc = model.predict(X_test)  # .ravel()

        # PRECISION RECALL CURVE
        lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_test, y_pred_roc)
        lr_auc = auc(lr_recall, lr_precision)
        lr_noskill = len(y_test[y_test == 1]) / len(y_test)

        plt.plot([0, 1], [lr_noskill, lr_noskill], linestyle='--')
        plt.plot(lr_recall, lr_precision, label='PR (Area = {:.3f})'.format(lr_auc))
        plt.xlabel('Recall (TPR)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.savefig(figPath + 'pr.png', dpi=img_dpi)
        plt.savefig(figPath + 'pr.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(figPath + 'pr.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        # Raw PR data
        print('Saving raw PR data')
        f = open(figPath + "pr_data_raw.csv", 'w+')
        f.write('Baseline: ' + str(lr_noskill) + '\n')
        f.write('i;Recall;Precision;Thresholds\n')
        for i in range(len(lr_precision)):
            f.write(
                str(i + 1) + ';' + str(lr_recall[i]) + ';' + str(lr_precision[i]) + ';' + str(lr_precision[i]) + ';\n')
        f.close()

        # ROC CURVE
        print('Calculating roc curve.')
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred_roc)

        print('Calculating AUC.')
        auc_roc = auc(fpr_roc, tpr_roc)

        print('Plotting roc curve.')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_roc, tpr_roc, label='ROC (Area = {:.3f})'.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(figPath + 'roc.png', dpi=img_dpi)
        plt.savefig(figPath + 'roc.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(figPath + 'roc.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        # Raw ROC data
        print('Saving raw ROC data')
        f = open(figPath + "roc_data_raw.csv", 'w+')
        f.write('i;FPR;TPR;Thresholds\n')
        for i in range(len(thresholds_roc)):
            f.write(
                str(i + 1) + ';' + str(fpr_roc[i]) + ';' + str(tpr_roc[i]) + ';' + str(thresholds_roc[i]) + ';\n')
        f.close()

        # Try this out: https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/
        # Precision / Recall Curve

        # HISTOGRAM

        hist_pos = y_pred_roc[np.where(y_pred_roc > 0.5)]
        plt.hist(hist_pos, bins='auto')
        plt.title("Histogram: Positive")
        plt.savefig(figPath + 'histogram_1.png', dpi=img_dpi)
        plt.clf()

        hist_neg = y_pred_roc[np.where(y_pred_roc <= 0.5)]
        plt.hist(hist_neg, bins='auto')
        plt.title("Histogram: Negative")
        plt.savefig(figPath + 'histogram_0.png', dpi=img_dpi)
        plt.clf()

        plt.hist(y_pred_roc, bins='auto')
        plt.title("Histogram: All")
        plt.savefig(figPath + 'histogram_all.png', dpi=img_dpi)
        plt.clf()

        plt.hist(label, bins='auto')
        plt.title("Histogram: All [Capped]")
        axes = plt.gca()
        plt.ylim(0, 2000)
        plt.xlim(0, 1)
        plt.savefig(figPath + 'histogram_all2.png', dpi=img_dpi)
        plt.clf()

        # TPR / FNR
        print("Calculating TPR / TNR, etc. for: " + label + ".")

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        bp = 0
        bn = 0
        pp = 0
        pn = 0

        y_baseline_values = sigmoid_binary(y_test)
        y_prediction_values = sigmoid_binary(y_pred_roc)
        for i in range(len(y_baseline_values)):
            current_baseline = y_baseline_values[i][0]
            current_prediction = y_prediction_values[i][0]

            if current_baseline == 1:
                bp = bp + 1
                if current_prediction == 1:
                    tp = tp + 1
                    pp = pp + 1
                else:
                    fn = fn + 1
                    pn = pn + 1
            else:
                bn = bn + 1
                if current_prediction == 1:
                    fp = fp + 1
                    pp = pp + 1
                else:
                    tn = tn + 1
                    pn = pn + 1

        f = open(outPathCurrent + "test_data_statistics.csv", 'w+')
        f.write('Count;Baseline;Predicted\n')
        f.write('All;' + str(len(y_baseline_values)) + ';' + str(len(y_prediction_values)) + '\n')
        f.write('Positive;' + str(bp) + ';' + str(pp) + '\n')
        f.write('Negative;' + str(bn) + ';' + str(pn) + '\n\n')

        f.write('TPR;' + str(tp / bp) + '\n')
        f.write('TNR;' + str(tn / bn) + '\n')
        f.write('FPR;' + str(fp / bn) + '\n')
        f.write('FNR;' + str(fn / bp) + '\n')

        f.write('ACC;' + str((tp + tn) / (bp + bn)) + '\n')
        f.write('BACC;' + str(((tp / bp) + (tn / bn)) / 2) + '\n')
        f.write('F1;' + str((2 * tp) / (2 * tp + fp + fn)) + '\n')

        f.close()

    except Exception as e:
        # Printing the exception message to file.
        print("Failed to calculate roc curve for: " + label + ".")
        f = open(figPath + "rocError.txt", 'w+')
        f.write(str(e))

        try:
            # Printing the stack trace to the file
            exc_info = sys.exc_info()
            f.write('\n')
            f.write(str(exc_info))
        except Exception as e2:
            print('Failed to write the whole stack trace into the error file. Reason:')
            print(str(e2))
            pass

        f.close()

    del X_test

# END OF FILE
#############

print('Training done.')
print('Timestamp: ', gct())
print('Your results here: ' + outPath)
