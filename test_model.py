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
import pandas as pd
from keras.models import load_model
import misc_omnisphero as misc

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import *

from scramblePaths import *
from misc_omnisphero import *

from keras.preprocessing.image import *
import matplotlib.pyplot as plt
import sys

from test_utils import test_cnn

# PATHS & ARGS
cuda_devices = "0"

# OLD DATA
#model_path = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/oligo_final_sigmodal/0_custom/'
#test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_trainingData_oligo/'
test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_trainingData_neuron/'

# KONTROLLIERT DATA
model_path = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-kontrolliert-weighted/neuron-n4-ep1500/0_custom/'
test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test/'
#test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'

print(model_path)
print(test_data_path)

normalize_enum = 4
img_dpi = 450
label = 'cnn-test'

# TESTING

test_cnn(model_path, test_data_path, normalize_enum, img_dpi, cuda_devices, True, label='cnn-test')

print('Testing done.')