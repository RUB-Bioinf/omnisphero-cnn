''' Give Omnisphero-CNN predictions for a given model on a folder of csv data.

JOSHUA BUTKE, AUGUST 2019
'''
# IMPORTS
#########

import os
import sys
import time

import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

# Custom Module
###############
import misc_omnisphero as misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Imports done...")

# LOAD MODEL
############
model = load_model('/bph/home/nilfoe/Documents/CNN/results/neuron_final_sigmodal/0_custom/custom.h5')
model.load_weights('/bph/home/nilfoe/Documents/CNN/results/neuron_final_sigmodal/0_custom/custom_weights_best.h5')

print("Loaded model...")

# MAIN LOOP
###########

# path = '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/wholeWell/neuron/'
# construct directory walker
# dir_list = [x[0] for x in os.walk(path)]
# print("Constructed directory walker...")

dir_list = [
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/ESM31_unannotatedData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/ESM32_unannotatedData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/ESM33_unannotatedData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron/ESM34_unannotatedData_neuron/'
    
    #'/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_unannotatedData_oligo/'
    #'/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_unannotatedData_neuron/'
]

source_dir = ''
#source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo_batch3/'

if len(source_dir) > 1:
    additional_dir = misc.get_immediate_subdirectories(source_dir)
    print('Discovered source dirs: ' + str(len(additional_dir)))
    for d in additional_dir:
        dir_list.append(os.path.join(source_dir,d))

print('Predicting experiment count: ' + str(len(dir_list)))
time.sleep(3)

for folder in dir_list[0:]:
    print('Considering: ' + folder)
    if "unannotated" in folder:
        pass
    else:
        continue

    # load data
    X_to_predict, _ = misc.hdf5_loader(str(folder))

    # process data
    X_to_predict = np.asarray(X_to_predict)
    print('Loaded data at: ', str(folder))
    print(X_to_predict.shape)
    X_to_predict = np.moveaxis(X_to_predict, 1, 3)
    X_to_predict = misc.normalize_RGB_pixels(X_to_predict)
    print(X_to_predict.shape)

    # generate prediction
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
            
            # reading
            print(f'Manipulating: ' + str(filename), end="\r")
            df = pd.read_csv(filename, delimiter=';')
            df_length = len(df['label'])
            df['label'] = binary_label[start_point:start_point + df_length]

            # save with new name
            split_name = filename.split('.')
            df.to_csv(split_name[0] + '_prediction.csv', sep=';', index=False)

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
