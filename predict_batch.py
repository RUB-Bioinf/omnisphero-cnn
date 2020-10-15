""" Give Omnisphero-CNN predictions for a given model on a folder of csv data.

Nils Foerster
Joshua Butke
2019 - 2020
"""
# IMPORTS
#########

import pandas as pd
from keras.models import load_model

# Custom Module
###############
import misc_omnisphero as misc
from misc_omnisphero import *

gpu_index_string = "2"

# MODELS IN USE
# Default trained for N1 normalisation
model_source_path_oligo = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/oligo_final_sigmodal/0_custom/'
model_source_path_neuron = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/neuron_final_sigmodal/0_custom/'

# MODELS TO DEBUG THAT FEATURE N4 NORMALISATION
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/oligo-n4/0_custom/'
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/neuron-n4/0_custom/'

# MODELS TO BE VALIDATED
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/oligo_fieldTest_WObrightness_longer/0_custom/'

source_dir_oligo = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/final/oligo_6/'
source_dir_neuron = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/final/neuron_6/'

source_dir_whole_well_oligo = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo_wholeWell/'
source_dir_whole_well_neuron = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron_wholeWell/'

# ######### To validate, use these whole well experiments: #########
# source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/unannotated/'
# source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/unannotated/'

# Source dir debugs
# source_dir = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/debug/2020_mar_set_oligo'
# source_dir = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/debug/2020_mar_set_neuron'

# Sorce dir with overexposure experiments
# source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/debug/oligo_norm/'
# source_dir = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/debug/neuron_norm/'

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum = None


def predict_batch(model_source_path: str, source_dir: str, normalize_enum: int = normalize_enum,
                  gpu_index_string=gpu_index_string,n_jobs=1):
    print('Loading model: ' + model_source_path)
    print('Data to predict: ' + source_dir)

    gpu_indexes = list(gpu_index_string.replace(",", ""))
    gpu_index_count = len(gpu_indexes)
    print("Visible GPUs: '" + gpu_index_string + "'. Count: " + str(gpu_index_count))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_string
    if gpu_index_count > 1:
        raise Exception(
            "GPUs assigned to predicting failed: " + gpu_index_string + "! Cannot predict on more than one GPU!")

    # LOADING MODELS
    if os.path.exists(model_source_path + 'custom.h5'):
        model = load_model(model_source_path + 'custom.h5')
        model.load_weights(model_source_path + 'custom_weights_best.h5')
    else:
        model = load_model(model_source_path + 'model.h5')
        model.load_weights(model_source_path + 'weights_best.h5')
    print("Loaded model...")

    dir_list = []
    if len(source_dir) > 1:
        additional_dir = misc.get_immediate_subdirectories(source_dir)
        print('Discovered source dirs: ' + str(len(additional_dir)))
        for d in additional_dir:
            dir_list.append(os.path.join(source_dir, d))

    global_progress_max = len(dir_list)
    global_progress_current = 0
    print(gct() + ' Predicting experiment count: ' + str(global_progress_max))

    print(gct() + ' Waiting. Prediction will start soon! Buckle up!')
    time.sleep(6)

    for folder in dir_list[0:]:
        print('Considering: ' + folder)
        print('Prediction progress: ' + str(global_progress_current) + ' / ' + str(global_progress_max))
        print('Current date: ' + misc.gct())

        global_progress_current = global_progress_current + 1
        if "unannotated" not in folder:
            print('Ignoring non-data folder: "'+folder+'"')
            continue

        print('Loading prediction data.')
        # load data
        X_to_predict, _, loading_errors = misc.hdf5_loader(str(folder), gp_current=global_progress_current, gp_max=global_progress_max, normalize_enum=normalize_enum, n_jobs=n_jobs,force_verbose=True)

        # process data
        X_to_predict = np.asarray(X_to_predict)
        print("Loaded data at: ", str(folder))
        print("Loaded data to be predicted has shape: " + str(X_to_predict.shape))
        X_to_predict = np.moveaxis(X_to_predict, 1, 3)
        # X_to_predict = misc.normalize_RGB_pixels(X_to_predict)
        print("Preprocessed data to be predicted has shape: " + str(X_to_predict.shape))

        # generate prediction
        print("Prediction progress: " + str(global_progress_current) + '/' + str(global_progress_max))
        print("Generating predictions...")
        label = model.predict(X_to_predict, verbose=1)
        binary_label = misc.sigmoid_binary(label)
        misc.count_uniques(binary_label)

        # cleanup / free memory
        del X_to_predict

        # TODO what is happening here??
        path_to_csv = str(folder)
        os.chdir(path_to_csv)
        directory_csv = os.fsencode(path_to_csv)
        directory_csv_contents = os.listdir(directory_csv)
        directory_csv_contents.sort()
        start_point = 0

        # CSV Manipulation
        ##################
        print("Writing CSVs...")
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
                    # TODO display error
                    pass

                    f.close()

                # update start point
                start_point += df_length

        print("Saving raw predictions.")
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


def main():
    use_oligo = True
    use_neuron = True
    use_debug = False
    n_jobs:int = 20

    # Paper Models trained for N4
    model_source_path_oligo_paper = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize4/'
    model_source_path_neuron_paper = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/neuron-normalize4/'

    if use_neuron:
        predict_batch(model_source_path=model_source_path_neuron, source_dir=source_dir_neuron,
                      normalize_enum=1,
                      n_jobs=n_jobs,
                      gpu_index_string="0")

    if use_oligo:
        predict_batch(model_source_path=model_source_path_oligo, source_dir=source_dir_oligo,
                      normalize_enum=1,
                      n_jobs=n_jobs,
                      gpu_index_string="1")

    if use_debug:
        predict_batch(model_source_path=model_source_path_oligo_paper,
                      source_dir=source_dir_oligo,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      gpu_index_string="0")

        predict_batch(model_source_path=model_source_path_neuron_paper,
                      source_dir=source_dir_neuron,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      gpu_index_string="0")

    print(gct() + ' All Predictions done. Have a nice day. =)')


if __name__ == "__main__":
    main()
