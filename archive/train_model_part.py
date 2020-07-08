import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Custom Module
###############
import sys

sys.path.append('/bph/puredata1/bioinfdata/user/butjos/work/code/misc')

import misc_omnisphero as misc

# TRAINING DATA
###############
path_list = [
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
    '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/'
    # '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/'
    # TODO
]
X, y = misc.multiple_hdf5_loader(path_list)  # load datasets

print("Loaded data has shape: ")
print(X.shape)
print(y.shape)
# print("Correcting axes...")
# X = np.moveaxis(X,1,3)
# y = y.astype(np.int)
# print(X.shape)

np.savez('/bph/puredata4/bioinfdata/work/omnisphero/CNN/temp/temp', X, y)
