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

from keras.backend import tensorflow_backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Custom Imports
import misc_cnn
import models
from misc_omnisphero import *
from scramblePaths import *
from test_model import test_cnn

p_allow_growth: bool = False
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

final_neurons_validated_validation_set_reorder = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_val_reorder/']

final_neurons_validated = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/ELS470_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/ESM49_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/FJK130_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/JK122_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/JK242_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train/MP149_trainingData_neuron/'
]

final_neurons_validated_reorder = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/EKB5_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/ELS470_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/ELS81_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/ESM49_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/ESM9_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/FJK125_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/JK242_trainingData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_train_reorder/MP149_trainingData_neuron/'
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

final_oligos_validated_validation_set_reorder = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_val_reorder/']

final_oligos_validated = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/ELS470_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/ESM10_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/ESM49_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/ESM9_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK122_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK153_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK155_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK156_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK242_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/JK95_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/MP149_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/MP66_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/MP67_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train/MP70_trainingData_oligo/'
]

final_oligos_validated_reorder = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/EKB5_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/ESM10_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/ESM49_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/ESM9_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/JK153_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/JK155_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/JK156_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/JK242_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/JK95_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/MP149_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/MP66_trainingData_oligo/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_train_reorder/MP70_trainingData_oligo/'
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
# 'adadelta', 'adam', 'SGD(lr=learn_rate)'
optimizer = 'SGD'
allowed_optimizers = ['adam', 'adadelta', 'SGD']
# TODO actually use this


# Metrics name determines the metrics used during fitting.
# Possible entries:
# 'mean_sqaure_error, 'accuracy'
metrics = ['accuracy']

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


def train_model(training_path_list: [str], validation_path_list: [str], out_path: str, test_data_path: str,
                # Multi param
                lossEnum: str = lossEnum, normalize_enum: int = normalize_enum, n_classes: int = n_classes,
                batch_size: int = batch_size,
                # Input data
                input_height: int = input_height, input_width: int = input_width, input_depth: int = input_depth,
                data_format: str = data_format, epochs: int = epochs,
                # optimizer params
                optimizer=optimizer, sgd_momentum: float = 0.9, sgd_nesterov: bool = False, metrics=metrics,
                learn_rate: int = learn_rate,
                # multi-threadding
                n_jobs: int = 1, single_thread_loading: bool = False,
                # data ugmentation
                data_gen: ImageDataGenerator = None, use_SMOTE: bool = False,
                # Training data split params
                split_proportion: float = None, split_stratify: bool = False,
                # misc params
                gpu_index_string: str = gpu_index_string, p_allow_growth:bool=p_allow_growth,
                img_dpi: int = img_dpi_default,
                example_sample_count: int = 25,
                label: str = None, global_progress_current: int = 1, global_progress_max: int = 1):
    # Creating specific out dirs

    # Importing Tensorflow and setting the session & GPU Management
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index_string
    gpu_indexes = list(gpu_index_string.replace(",", ""))
    gpu_index_count = len(gpu_indexes)
    print("Visible GPUs: '" + gpu_index_string + "'. Count: " + str(gpu_index_count))

    # Important! Set GPU Index String before importing tensorflow!
    import tensorflow as tf

    # Keras session growth
    if p_allow_growth:
        print('CUDA GPU Mem Allocation Growth enabled!')
    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
        # Code snipit credit: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
    else:
        print('CUDA GPU Mem Allocation Growth disabled! Hogging all the memory for myself!')


    print('Saving results here: ' + out_path)
    os.makedirs(out_path, exist_ok=True)

    if str(optimizer) not in allowed_optimizers:
        raise Exception('Cannot train model with given optimizer: "' + str(optimizer) + '"! Valid optimizers: ' + str(
            allowed_optimizers))

    has_lr_adjusting_optimizer = True
    if optimizer == 'SGD':
        has_lr_adjusting_optimizer = False
        optimizer = SGD(lr=learn_rate, momentum=sgd_momentum, nesterov=sgd_nesterov)
    else:
        sgd_nesterov = 'N / A'
        sgd_momentum = 'N / A'

    augment_path = out_path + 'augments' + os.sep
    augment_smote_path = augment_path + 'smote' + os.sep
    fig_path = out_path + 'fig' + os.sep
    sample_path = fig_path + 'samples' + os.sep
    fig_path_model = fig_path + 'model' + os.sep

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(augment_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(fig_path_model, exist_ok=True)
    os.makedirs(augment_smote_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

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
    X, y, _ = multiple_hdf5_loader(training_path_list, gp_current=global_progress_current, gp_max=global_progress_max,
                                   normalize_enum=normalize_enum, n_jobs=len(training_path_list),
                                   single_thread_loading=single_thread_loading)  # load datasets

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

    minority_count = min(np.count_nonzero(y == 1), np.count_nonzero(y == 0))
    k_neighbors = int(min(max(int(minority_count / 25) + 1, 150), minority_count / 2))
    print('Minority count for smote: ' + str(minority_count) + '. k_neighbour candidates: ' + str(
        int(minority_count * 0.25)) + '. Actually: ' + str(k_neighbors))

    print('SMOTE mode: ' + str(use_SMOTE))
    n_samples, n_x, n_y, n_z = X.shape

    # TRAIN TEST SPLIT
    ###############
    split_out_file = out_path + 'data_splitting.txt'
    f = open(split_out_file, 'w')

    y_split = None
    X_split = None
    if split_proportion is not None and split_proportion > 0.0:
        print('Splitting traing / val data in ' + str(split_proportion) + '-ratio. Stratify: ' + str(split_stratify))
        f.write('Splitting traing / val data in ' + str(split_proportion) + '-ratio. Stratify: ' + str(
            split_stratify) + '\n')
        param_strat = None
        if split_stratify:
            param_strat = y

        print('Pre splitting shape:')
        f.write("X-shape: " + str(X.shape) + '\n')
        f.write("y-shape: " + str(y.shape) + '\n')
        f.write('y==0 count: ' + str(np.count_nonzero(y == 0)) + '\n')
        f.write('y==1 count: ' + str(np.count_nonzero(y == 1)) + '\n')

        X, X_split, y, y_split = train_test_split(X, y, test_size=split_proportion, stratify=param_strat)
        print('Finished splitting.')

        f.write('Post splitting shape:\n')
        f.write("X-shape: " + str(X.shape) + '\n')
        f.write("y-shape: " + str(y.shape) + '\n')
        f.write("X_split-shape: " + str(X_split.shape) + '\n')
        f.write("y_split-shape: " + str(y_split.shape) + '\n')
        f.write('Post splitting y==0 count: ' + str(np.count_nonzero(y == 0)) + '\n')
        f.write('Post splitting y==1 count: ' + str(np.count_nonzero(y == 1)) + '\n')
        f.write('Post splitting y_split==0 count: ' + str(np.count_nonzero(y_split == 0)) + '\n')
        f.write('Post splitting y_split==1 count: ' + str(np.count_nonzero(y_split == 1)) + '\n')
    else:
        f.write('Not splitting.')
    f.close()

    # SMOTE DATA
    ###############
    smote_params: str = 'No SMOTE used'
    smote_error_text = 'N/A.'
    if use_SMOTE:
        # Smote for image classification: https://medium.com/swlh/how-to-use-smote-for-dealing-with-imbalanced-image-dataset-for-solving-classification-problems-3aba7d2b9cad
        # TODO out-sorce this as an independent function?

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

    # Data Augmentation
    if data_gen is not None:
        print("Fitting X to the data-gen.")
        data_gen.fit(X)
        print("Done.")

    # VALIDATION DATA
    #################
    print("Loading validation data. Source folder count: " + str(len(validation_path_list)))
    X_val, y_val, _ = multiple_hdf5_loader(validation_path_list, gp_current=global_progress_current,
                                           gp_max=global_progress_max,
                                           normalize_enum=normalize_enum,
                                           single_thread_loading=single_thread_loading,
                                           n_jobs=len(validation_path_list))
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

    split_out_file_val = out_path + 'data_splitting_val.txt'
    f = open(split_out_file_val, 'w')
    if y_split is not None and X_split is not None:
        print('Merging plit training data with validation data.')
        f.write('Merging plit training data with validation data.\n')
        f.write('X_val original shape: ' + str(X_val.shape) + '\n')
        f.write('y_val original shape: ' + str(y_val.shape) + '\n')

        f.write('Training data split off:\n')
        f.write('X_split shape: ' + str(X_split.shape) + '\n')
        f.write('y_split shape: ' + str(y_split.shape) + '\n')

        X_val = np.concatenate((X_val, X_split), axis=0)
        y_val = np.concatenate((y_val, y_split), axis=0)

        f.write('Post merging data:\n')
        f.write('X_val merged shape: ' + str(X_val.shape) + '\n')
        f.write('y_val merged shape: ' + str(y_val.shape) + '\n')

        del y_split
        del X_split
        y_split = None
        X_split = None
    else:
        f.write('Not merging.')

    f.close()

    # CONSTRUCTION
    ##############
    steps_per_epoch = math.nan
    print("Building model...")
    model: Model = models.omnisphero_model(n_classes, input_height, input_width, input_depth, data_format)
    if gpu_index_count > 1:
        model = multi_gpu_model(model, gpus=gpu_index_count)
        steps_per_epoch = len(X) / epochs
        print("Model has been set up to run on multiple GPUs.")
        print("Steps per epoch: " + str(steps_per_epoch))

        print('WARNING! There seems to be an issue with multi gpu and batch validation size!!')
        # https://github.com/keras-team/keras/issues/11434#issuecomment-439832556

    print("Compiling model...")
    model.compile(loss=lossEnum, optimizer=optimizer, metrics=metrics)
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
    f = open(out_path + 'model_training_params.txt', 'w+')
    data_gen_description = 'None.'
    if data_gen is not None:
        data_gen_description = 'Used: ' + str(data_gen)

    f.write('Training start time: ' + gct() + '\n')
    f.write('Model: ' + str(model) + '\n')
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
    f.write('SGD Momentum: ' + str(sgd_momentum) + '\n')
    f.write('SGD Nesterov: ' + str(sgd_nesterov) + '\n')
    f.write('Epochs: ' + str(epochs) + '\n')
    f.write('Normalization mode: ' + str(normalize_enum) + '\n')
    f.write('Model metrics: ' + str(model.metrics_names) + '\n')
    f.write('Model optimizer: ' + str(optimizer) + '\n')
    f.write('Has lr-adjusting optimizer: ' + str(has_lr_adjusting_optimizer) + '\n')
    f.write('Model metrics raw: ' + str(metrics) + '\n')
    f.write('Data Generator used: ' + data_gen_description + '\n')
    f.write('SMOTE Parameters: ' + str(smote_params) + '\n')
    f.write('SMOTE Error: ' + smote_error_text + '\n')
    f.write('Train-Test Split: Proportions: ' + str(split_proportion) + '\n')
    f.write('Train-Test Split: Stragize: ' + str(split_stratify) + '\n')

    f.write('\n == DATA: ==\n')
    f.write("X shape: " + str(X.shape) + '\n')
    f.write("y shape: " + str(y.shape) + '\n')
    f.write('y==0 count: ' + str(np.count_nonzero(y == 0)) + '\n')
    f.write('y==1 count: ' + str(np.count_nonzero(y == 1)) + '\n')
    f.write("X_val shape: " + str(X_val.shape) + '\n')
    f.write("y_val shape: " + str(y_val.shape) + '\n')
    f.write('y_val==0 count: ' + str(np.count_nonzero(y_val == 0)) + '\n')
    f.write('y_val==1 count: ' + str(np.count_nonzero(y_val == 1)) + '\n')

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

    if example_sample_count > 0:
        save_random_samples(X, y, count=example_sample_count, path=sample_path + os.sep + 'train' + os.sep)
        save_random_samples(X_val, y_val, count=example_sample_count, path=sample_path + os.sep + 'val' + os.sep)

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
    model_checkpoint_best = ModelCheckpoint(weights_best_filename, monitor='val_loss', verbose=1,
                                            save_best_only=True,
                                            mode='min')
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=learn_rate_factor,
                                   patience=learn_rate_reduction_patience,
                                   verbose=1,
                                   mode='auto', min_delta=0.000001, cooldown=0, min_lr=0.000001)
    csv_logger = CSVLogger(log_out_path, separator=';', append=True)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=1, mode='auto',
                                        baseline=None,
                                        restore_best_weights=True)  # early stopping
    canary_interrupt_callback = misc_cnn.CanaryInterruptCallback(path=out_path)
    live_plot_callback = misc_cnn.PlotTrainingLiveCallback(out_dir=out_path, label=gpu_index_string,
                                                           epochs_target=epochs)

    callbacks_list = [model_checkpoint,
                      model_checkpoint_best,
                      csv_logger,
                      early_stop_callback,

                      canary_interrupt_callback,
                      live_plot_callback
                      ]

    if not has_lr_adjusting_optimizer:
        callbacks_list.append(lrCallBack)
        print('Including ReduceLROnPlateau Callback.')

    # TRAINING
    ##########
    if label is not None:
        print('Reminder. Training for label: ' + label)
    print('Saving model here: ' + out_path)
    print('Training started: ' + gct())

    # Checking if a data generator exists. If so, datagen mode will be used. If not, classic training.
    history_all = None
    if data_gen is None:
        print('Fitting model without a data gen!')
        history_all = model.fit(x=X, y=y,
                                validation_data=(X_val, y_val),
                                callbacks=callbacks_list,
                                epochs=epochs,
                                batch_size=batch_size,
                                # class_weight=class_weights
                                )
    else:
        print('Fitting model and using a data gen!')
        history_all = model.fit_generator(data_gen.flow(
            x=X, y=y,
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
        print(gct() + " Failed plot history! Error type:")
        print(e)
        # TODO print stacktrace

    # SAVING ON MEMORY
    del X_val
    del X
    del X_split

    del y
    del y_val
    del y_split
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

        title = 'Raw ' + label
        plt.plot(history_all.history[hist_key])
        plt.title(title)
        plt.ylabel(label)
        plt.xlabel('Epoch')

        plt.savefig(fig_path + 'raw_' + hist_key + '.png', dpi=img_dpi)
        plt.savefig(fig_path + 'raw_' + hist_key + '.svg', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'raw_' + hist_key + '.pdf', dpi=img_dpi, transparent=True)
        plt.clf()

        f = open(fig_path + 'raw_' + hist_key + '.tex', 'w')
        f.write(misc_cnn.get_plt_as_tex(data_list_y=[history_all.history[hist_key]], title=title,
                                        label_y=label, label_x='Epoch',
                                        plot_colors=['blue']))
        f.close()

        i = i + 1
        print('Saved raw data for: ' + hist_key + ' [' + label + '].')

        val_hist_key = 'val_' + hist_key
        if val_hist_key in hist_key_set:
            val_label = decode_history_key(val_hist_key)
            h_train = history_all.history[hist_key]
            h_val = history_all.history[val_hist_key]

            title = 'Model ' + label
            plt.plot(h_train)
            plt.plot(h_val)
            plt.title(title)
            plt.ylabel(label)
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='best')

            plt.savefig(fig_path + 'val_' + hist_key + '.png', dpi=img_dpi)
            plt.savefig(fig_path + 'val_' + hist_key + '.svg', dpi=img_dpi, transparent=True)
            plt.savefig(fig_path + 'val_' + hist_key + '.pdf', dpi=img_dpi, transparent=True)
            plt.clf()

            f = open(fig_path + 'model_' + hist_key + '.tex', 'w')
            f.write(misc_cnn.get_plt_as_tex(data_list_y=[h_train, h_val], title=title,
                                            label_y=label, label_x='Epoch', plot_titles=['Training', 'Validation'],
                                            plot_colors=['blue', 'orange']))
            f.close()

            print('Saved combined validation: ' + label + ' & ' + val_label)

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

    oligo_mode = False
    oligo_mode2 = True
    neuron_mode = True
    debug_mode = False
    n_jobs = 40

    print('Sleeping....')
    # time.sleep(18000)

    if debug_mode:
        train_model(
            training_path_list=debug_oligos,
            validation_path_list=debug_oligos_validation,
            test_data_path='/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test/',
            # data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_SGD' + os.sep + 'oligo-split2' + os.sep,
            gpu_index_string="1",
            optimizer='adam',
            single_thread_loading=False,
            split_proportion=0.2,
            epochs=10
        )

    if oligo_mode2:
        train_model(
            training_path_list=final_oligos_validated,
            validation_path_list=final_oligos_validated_validation_set,
            test_data_path=test_data_path_oligo,
            data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_datagen' + os.sep + 'oligo-normalize4' + os.sep,
            normalize_enum=4,
            gpu_index_string="0",
            n_jobs=n_jobs,
            optimizer='SGD',
            epochs=5000
        )


    if oligo_mode:
        train_model(
            training_path_list=final_oligos_validated,
            validation_path_list=final_oligos_validated_validation_set,
            test_data_path=test_data_path_oligo,
            data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_datagen' + os.sep + 'oligo-normalize1' + os.sep,
            normalize_enum=1,
            gpu_index_string="1",
            optimizer='SGD',
            epochs=5000
        )
        train_model(
            training_path_list=final_oligos_validated,
            validation_path_list=final_oligos_validated_validation_set,
            test_data_path=test_data_path_oligo,
            data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_datagen' + os.sep + 'oligo-normalize2' + os.sep,
            normalize_enum=2,
            gpu_index_string="1",
            optimizer='SGD',
            epochs=5000
        )
        train_model(
            training_path_list=final_oligos_validated,
            validation_path_list=final_oligos_validated_validation_set,
            test_data_path=test_data_path_oligo,
            data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_datagen' + os.sep + 'oligo-normalize3' + os.sep,
            normalize_enum=0,
            gpu_index_string="3",
            optimizer='SGD',
            epochs=5000
        )

    #paper-final_no-datagen\neuron
    if neuron_mode:
        train_model(
            training_path_list=final_neurons_validated,
            validation_path_list=final_neurons_validated_validation_set,
            test_data_path=test_data_path_neuron,
            data_gen=data_gen,
            use_SMOTE=False,
            out_path=out_path + 'paper-final_datagen' + os.sep + 'neuron-normalize4' + os.sep,
            normalize_enum=4,
            gpu_index_string="0",
            n_jobs=n_jobs,
            optimizer='SGD',
            epochs=5000
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

    print('Finished all trainings. Goodbye.')


if __name__ == "__main__":
    main()
