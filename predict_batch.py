""" Give Omnisphero-CNN predictions for a given model on a folder of csv data.

Nils Foerster
Joshua Butke
2019 - 2020
"""
# IMPORTS
#########

import sys
import pandas as pd
from keras.models import load_model
import getpass
import socket

# Custom Module
###############
import misc_omnisphero as misc
from misc_omnisphero import *
from predict_batch_custom import predict_batch_custom

gpu_index_string = "2"

# MODELS IN USE
# Default trained for N1 normalisation
model_source_path_oligo = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/models/results/oligo_final_sigmodal/0_custom/'
model_source_path_neuron = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/models/results/neuron_final_sigmodal/0_custom/'

# MODELS TO DEBUG THAT FEATURE N4 NORMALISATION
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/oligo-n4/0_custom/'
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-normalizing/neuron-n4/0_custom/'

# MODELS TO BE VALIDATED
# modelSourcePath = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/oligo_fieldTest_WObrightness_longer/0_custom/'

source_dir_oligo = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/oligo_18/'
source_dir_neuron = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/neuron_18/'

source_dir_paper_redo_oligo  = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/oligo_6/'
source_dir_paper_redo_neuron = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/neuron_6/'

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
                  gpu_index_string=gpu_index_string, skip_predicted: bool = False, n_jobs=1):
    print(' == #### ===')
    print('Loading model: ' + model_source_path)
    print('Data to predict: ' + source_dir)
    print(' == #### ===')
    time.sleep(4)

    f = open(source_dir + os.sep + 'protocoll.txt', 'w')
    f.write('Host: '+str(getpass.getuser())+'\n')
    f.write('User: '+str(socket.gethostname())+'\n')
    f.write('Timestamp: '+gct()+'\n\n')
    f.write('Model path: '+model_source_path+'\n')
    f.write('GPUs: '+gpu_index_string+'\n')
    f.write('Skip predicted: '+str(skip_predicted)+'\n')
    f.write('Normalize enum: '+str(normalize_enum)+'\n')
    f.close()

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
    print("Finished loading model.")
    time.sleep(2)

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
        global_progress_current = global_progress_current + 1
        print('\n' + ' <(*.*)> ==[' + str(global_progress_current) + ' / ' + str(global_progress_max) + ']== <(*.*)>')
        print('Predicting: ' + folder)
        X_to_predict = None

        if "unannotated" not in folder:
            print('Ignoring non-data folder: "' + folder + '"')
            continue

        print('Loading prediction data.')
        # load data
        X_to_predict, _, loading_errors, skipped = misc.hdf5_loader(str(folder), gp_current=global_progress_current,
                                                                    gp_max=global_progress_max,
                                                                    normalize_enum=normalize_enum,
                                                                    skip_predicted=skip_predicted,
                                                                    n_jobs=n_jobs, force_verbose=True)

        if skipped:
            if skip_predicted:
                print('This folder was already predicted. Skipping.')
                continue
            else:
                print('This folder was already predicted. Predicting it again.')

        if len(X_to_predict) == 0:
            print(' ==[!! WARNING !!]== No wells have been loaded! Experiment skipped.')
            continue

        try:
            #print('X_to_predict len: ' + str(len(X_to_predict)))
            X_to_predict = np.asarray(X_to_predict)
            temp2 = X_to_predict.shape
            temp = np.moveaxis(X_to_predict, 1, 3)
            del temp
            del temp2
        except Exception as e:
            print(
                ' ==[!! WARNING !!]==\nFailed to convert X_to_predict to np array and determine its shape. This is a fatal error! Experiment skipped.')
            print(e)
            continue

        # process data
        print(gct()+" Loaded data to be predicted has shape: " + str(X_to_predict.shape)+'. Correcting axis.')
        X_to_predict = np.moveaxis(X_to_predict, 1, 3)
        print("Preprocessed data to be predicted has shape: " + str(X_to_predict.shape))

        # generate prediction
        print("Generating predictions...")
        label = model.predict(X_to_predict, verbose=1)
        binary_label = misc.sigmoid_binary(label)

        # Joshua function. Can this be reworked??
        # misc.count_uniques(binary_label)

        # Printing the predictions
        print('Predictions==0 count: ' + str(np.count_nonzero(binary_label == 0)))
        print('Predictions==1 count: ' + str(np.count_nonzero(binary_label == 1)))

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
                split_name = filename.split('.')
                split_name = split_name[0]

                try:
                    # reading
                    print(f'Manipulating: ' + str(filename), end="\r")
                    df = pd.read_csv(filename, delimiter=';')
                    df_length = len(df['label'])
                    df['label'] = binary_label[start_point:start_point + df_length]

                    # save with new name
                    df.to_csv(split_name + '_prediction.csv', sep=';', index=False)
                except Exception as e:
                    # TODO display error and stacktrace
                    try:
                        error_filename = path_to_csv + os.sep + split_name + '-error.txt'
                        ef = open(error_filename,'w')
                        ef.write(str(e))
                        ef.close()

                        print('Error in: ' + str(error_filename))
                        print('ERROR!!! ' + str(e))
                    except Exception as e2:
                        print('Error while saving the error: ' + str(e2))

                # update start point
                start_point += df_length

        print("Saving raw predictions.")
        np.save(path_to_csv + "-all_prediction.npy", label)
        np.savetxt(path_to_csv + "-all_prediction.csv", label, delimiter=';')

        # if label.size < 35000:
        #    print('Saving positive Histogram')
        #    hist_pos = label[np.where(label > 0.5)]
        #    plt.hist(hist_pos, bins='auto')
        #    plt.title("Histogram: Positive")
        #    plt.savefig(path_to_csv + '/histogram_1.png')
        #    plt.clf()
        #    print('Saving negative Histogram')
        #    hist_neg = label[np.where(label <= 0.5)]
        #    plt.hist(hist_neg, bins='auto')
        #    plt.title("Histogram: Negative")
        #    plt.savefig(path_to_csv + '/histogram_0.png')
        #    plt.clf()
        #    print('Saving whole Histogram')
        #    plt.hist(label, bins='auto')
        #    plt.title("Histogram: All")
        #    plt.savefig(path_to_csv + '/histogram_all.png')
        #    plt.clf()
        #    print('Saving capped Histogram')
        #    plt.hist(label, bins='auto')
        #    plt.title("Histogram: All [Capped]")
        #    axes = plt.gca()
        #    axes.set_ylim([0, 2000])
        #    axes.set_xlim([0, 1])
        #    plt.savefig(path_to_csv + '/histogram_all2.png')
        #    plt.clf()
        # else:
        #    print('Too many labels predicted. Histogram skipped.')


def main(args):
    print('Number of arguments:', len(args), 'arguments.')
    print('Argument List:', str(args))

    custom_paths = False;
    for arg in args:
        arg = str(arg).lower()
        print('Evaluating arg: "'+arg+'".')

        if arg == '-p' or arg == '-paths' or arg == '-c' or arg == '-custom':
            custom_paths = True

    if custom_paths:
        custom_paths_predict()
    else:
        prodi_gpu_predict()

def prodi_gpu_predict():
    print('Running Predictions.')
    use_oligo = True
    use_neuron = True
    use_glia = False
    use_old = False

    use_debug = False
    use_paper = True
    skip_predicted = True
    n_jobs: int = 20

    # Paper Models trained for N4
    model_source_path_oligo_paper = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize4/'
    model_source_path_neuron_paper = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/neuron-normalize4/'
    model_source_path_glia = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN-glia/models/glia/smote/'

    # .h5 dirs to be predicted for the paper
    # source_dir_redo_paper_oligo = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/oligo_paper/'
    # source_dir_redo_paper_neuron = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/final/neuron_paper/rosi/'

    # .h5 dirs to be predicted for efsa or endpoints
    source_dir_paper_oligo  = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/oligo_1/'
    source_dir_paper_neuron = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/neuron_1/'
    source_dir_glia = '/home/nilfoe/prodi/bioinfdata/work/Omnisphero/CNN/final/glia_01/'

    initial_sleep_time = 5
    print(' == Initial Sleeping: ' + str(initial_sleep_time) + ' seconds ... ===')
    time.sleep(initial_sleep_time)

    if use_glia:
        predict_batch(model_source_path=model_source_path_glia, source_dir=source_dir_glia,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      skip_predicted=skip_predicted,
                      gpu_index_string="1")


    if use_paper:
        if use_neuron:
            predict_batch(model_source_path=model_source_path_neuron_paper, source_dir=source_dir_paper_neuron,
                          normalize_enum=4,
                          n_jobs=n_jobs,
                          skip_predicted=skip_predicted,
                          gpu_index_string="1")
        if use_oligo:
            predict_batch(model_source_path=model_source_path_oligo_paper, source_dir=source_dir_paper_oligo,
                          normalize_enum=4,
                          n_jobs=n_jobs,
                          skip_predicted=skip_predicted,
                          gpu_index_string="1")
    if use_old:
        if use_neuron:
            predict_batch(model_source_path=model_source_path_neuron, source_dir=source_dir_neuron,
                          normalize_enum=1,
                          n_jobs=n_jobs,
                          skip_predicted=skip_predicted,
                          gpu_index_string="1")
        if use_oligo:
            predict_batch(model_source_path=model_source_path_oligo, source_dir=source_dir_oligo,
                          normalize_enum=1,
                          n_jobs=n_jobs,
                          skip_predicted=skip_predicted,
                          gpu_index_string="1")


    if use_debug:
        predict_batch(model_source_path=model_source_path_oligo_paper,
                      source_dir=source_dir_oligo,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      skip_predicted=False,
                      gpu_index_string="1")
        predict_batch(model_source_path=model_source_path_neuron_paper,
                      source_dir=source_dir_neuron,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      skip_predicted=False,
                      gpu_index_string="1")

    print(gct() + ' All Predictions done. Have a nice day. =)')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
