''' Give Omnisphero-CNN predictions for a given model on a folder of csv data.

JOSHUA BUTKE, AUGUST 2019
'''
# IMPORTS
#########

import os
import sys
import misc_omnisphero as misc
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append('/bph/puredata1/bioinfdata/user/butjos/work/code/misc')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("Imports done...")

# LOAD MODEL
############
model = load_model('/bph/home/nilfoe/Documents/CNN/results/neurons_final_350/0_custom/custom.h5')
model.load_weights('/bph/home/nilfoe/Documents/CNN/results/neurons_final_350/0_custom/custom_weights_best.h5')

print("Loaded model...")

# MAIN LOOP
###########

# path = '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/wholeWell/neuron/'
# construct directory walker
# dir_list = [x[0] for x in os.walk(path)]
# print("Constructed directory walker...")

dir_list = [
    # '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_unannotatedData_neuron/',
    '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_unannotatedData_neuron/'
]

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

    # CSV Manipulation
    ##################
    path_to_csv = str(folder)
    os.chdir(path_to_csv)
    directory_csv = os.fsencode(path_to_csv)
    directory_csv_contents = os.listdir(directory_csv)
    directory_csv_contents.sort()

    print('Saving predictions and Histogram')
    np.save(path_to_csv + "all_prediction.npy", label)
    np.savetxt(path_to_csv + "all_prediction.csv", label, delimiter=';')

    hist_pos = label[np.where(label > 0.5)]
    plt.hist(hist_pos, bins='auto')
    plt.title("Histogram: Positive")
    plt.savefig(path_to_csv + '_histogram_1.png')
    plt.clf()

    hist_neg = label[np.where(label <= 0.5)]
    plt.hist(hist_neg, bins='auto')
    plt.title("Histogram: Negative")
    plt.savefig(path_to_csv + '_histogram_0.png')
    plt.clf()

    plt.hist(label, bins='auto')
    plt.title("Histogram: All")
    plt.savefig(path_to_csv + '_histogram_all.png')
    plt.clf()

    plt.hist(label, bins='auto')
    plt.title("Histogram: All [Capped]")
    axes = plt.gca()
    axes.set_ylim([0, 2000])
    axes.set_xlim([0, 1])
    plt.savefig(path_to_csv + '_histogram_all2.png')
    plt.clf()

    start_point = 0

    print('Writing CSVs')
    for f in directory_csv_contents:
        filename = os.fsdecode(f)
        if filename.endswith('.csv') and not filename.endswith('_prediction.csv') and not filename.endswith(
                '_prediction_test.csv'):
            # reading
            print('Manipulating: ', filename)
            df = pd.read_csv(filename, delimiter=';')
            df_length = len(df['label'])
            df['label'] = binary_label[start_point:start_point + df_length]

            # save with new name
            split_name = filename.split('.')
            df.to_csv(split_name[0] + '_prediction.csv', sep=';', index=False)

            # update start point
            start_point += df_length

# END OF FILE
#############
print('Predictions done.')
