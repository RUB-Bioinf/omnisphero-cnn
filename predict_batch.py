''' Give Omnisphero-CNN predictions for a given model on a folder of csv data.

JOSHUA BUTKE, AUGUST 2019
'''
# IMPORTS
#########

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

# Custom Module
###############
import misc_omnisphero as misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gct():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

print("Imports done...")

# MODELS IN USE
# Default trained for N1 normalisation
modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/oligo_final_sigmodal/0_custom/'
#modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/neuron_final_sigmodal/0_custom/'

# MODELS TO DEBUG THAT FEATURE N4 NORMALISATION
#modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/oligo-n4/0_custom/'
#modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/neuron-n4/0_custom/'

# MODELS TO BE VALIDATED
#modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/oligo_fieldTest_WObrightness_longer/0_custom/'
print('Loading model: '+ modelSourcePath)

# LOADING MODELS
if os.path.exists(modelSourcePath + 'custom.h5'):
    model = load_model(modelSourcePath + 'custom.h5')
    model.load_weights(modelSourcePath + 'custom_weights_best.h5')
else:
    model = load_model(modelSourcePath + 'model.h5')
    model.load_weights(modelSourcePath + 'weights_best.h5')

print("Loaded model...")

# MAIN LOOP
###########
dir_list = []
source_dir = ''
source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo_40/'
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron_40/'

### To validate, use these whole well experiments:
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/unannotated/'
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/unannotated/'

# Source dir debugs
#source_dir = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/debug/2020_mar_set_oligo'
#source_dir = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/debug/2020_mar_set_neuron'

# Sorce dir with overexposure experiments
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/debug/oligo_norm/'
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/debug/neuron_norm/'

normalize_enum = 1
# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well

if len(source_dir) > 1:
    additional_dir = misc.get_immediate_subdirectories(source_dir)
    print('Discovered source dirs: ' + str(len(additional_dir)))
    for d in additional_dir:
        dir_list.append(os.path.join(source_dir,d))

gpMax = len(dir_list)
gpCurrent = 0
print('Predicting experiment count: ' + str(gpMax))

print('Waiting. Prediction will start soon! Buckle up!')
time.sleep(6)

for folder in dir_list[0:]:
    print('Considering: ' + folder)
    print('Prediction progress: ' + str(gpCurrent) + '/' + str(gpMax))
    print('Current date: '+gct())

    gpCurrent = gpCurrent + 1
    if "unannotated" in folder:
        pass
    else:
        continue

    # load data
    X_to_predict, _ = misc.hdf5_loader(str(folder),gpCurrent=gpCurrent,gpMax=gpMax,normalize_enum=normalize_enum)

    # process data
    X_to_predict = np.asarray(X_to_predict)
    print('Loaded data at: ', str(folder))
    print(X_to_predict.shape)
    X_to_predict = np.moveaxis(X_to_predict, 1, 3)
    #X_to_predict = misc.normalize_RGB_pixels(X_to_predict)
    print(X_to_predict.shape)

    # generate prediction
    print('Prediction progress: ' + str(gpCurrent) + '/' + str(gpMax))
    print('Generating predictions...')
    label = model.predict(X_to_predict, verbose=1)
    binary_label = misc.sigmoid_binary(label)
    misc.count_uniques(binary_label)

    # cleanup / free memory
    del X_to_predict

    path_to_csv = str(folder)
    os.chdir(path_to_csv)
    directory_csv = os.fsencode(path_to_csv)
    directory_csv_contents = os.listdir(directory_csv)
    directory_csv_contents.sort()
    start_point = 0

    # CSV Manipulation
    ##################
    print('Writing CSVs')
    for f in directory_csv_contents:
        filename = os.fsdecode(f)
        if filename.endswith('.csv') and not filename.endswith('_prediction.csv') and not filename.endswith(
                '_prediction_test.csv'):
            
            df_length = 0
            try:
                # reading
                print(f'Manipulating: ' + str(filename), end="\r")
                df = pd.read_csv(filename, delimiter=';')
                df_length = len(df['label'])
                df['label'] = binary_label[start_point:start_point + df_length]
    
                # save with new name
                split_name = filename.split('.')
                df.to_csv(split_name[0] + '_prediction.csv', sep=';', index=False)
            except Exception as e:
                pass

                f.close()

            # update start point
            start_point += df_length

    print('Saving raw predictions')
    np.save(path_to_csv + "all_prediction.npy", label)
    np.savetxt(path_to_csv + "all_prediction.csv", label, delimiter=';')

    if label.size < 35000:
        print('Saving positive Histogram')
        hist_pos = label[np.where(label > 0.5)]
        plt.hist(hist_pos, bins='auto')
        plt.title("Histogram: Positive")
        plt.savefig(path_to_csv + '/histogram_1.png')
        plt.clf()

        print('Saving negative Histogram')
        hist_neg = label[np.where(label <= 0.5)]
        plt.hist(hist_neg, bins='auto')
        plt.title("Histogram: Negative")
        plt.savefig(path_to_csv + '/histogram_0.png')
        plt.clf()

        print('Saving whole Histogram')
        plt.hist(label, bins='auto')
        plt.title("Histogram: All")
        plt.savefig(path_to_csv + '/histogram_all.png')
        plt.clf()

        print('Saving capped Histogram')
        plt.hist(label, bins='auto')
        plt.title("Histogram: All [Capped]")
        axes = plt.gca()
        axes.set_ylim([0, 2000])
        axes.set_xlim([0, 1])
        plt.savefig(path_to_csv + '/histogram_all2.png')
        plt.clf()
    else:
        print('Too many labels predicted. Histogram skipped.')


# END OF FILE
#############
print('Predictions done.')
