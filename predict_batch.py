""" Give Omnisphero-CNN predictions for a given model on a folder of csv data.

Nils Foerster
Joshua Butke
2019 - 2020
"""
# IMPORTS
#########

import sys
from sys import platform
import pandas as pd
from keras.models import load_model
import getpass
import socket
import math
import multiprocessing


# Custom Module
###############
import misc_omnisphero as misc
from misc_omnisphero import *
from predict_batch_custom import predict_batch_custom

gpu_index_string = "3"

# MODELS IN USE
# Default trained for N4 normalisation
default_model_source_path_oligo = '/prodi/bioinfdata/work/Omnisphero/CNN/diff/models/oligo/'
default_model_source_path_neuron = '/prodi/bioinfdata/work/Omnisphero/CNN/diff/models/neuron/'

default_source_dirs_oligo = ['/prodi/bioinfdata/work/Omnisphero/CNN/diff/data/pred/oligo10/']
default_source_dirs_neuron = ['/prodi/bioinfdata/work/Omnisphero/CNN/diff/data/pred/neuron10/']

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
    f.write('Host: ' + str(getpass.getuser()) + '\n')
    f.write('User: ' + str(socket.gethostname()) + '\n')
    f.write('Timestamp: ' + gct() + '\n\n')
    f.write('Model path: ' + model_source_path + '\n')
    f.write('GPUs: ' + gpu_index_string + '\n')
    f.write('Skip predicted: ' + str(skip_predicted) + '\n')
    f.write('Normalize enum: ' + str(normalize_enum) + '\n')
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
            print('The input folder must contain the keyword "unannotated" to be eligible for prediction!')
            continue

        print('Loading prediction data.')
        # load data
        X_to_predict, _, loading_errors, skipped = misc.hdf5_loader(str(folder), gp_current=global_progress_current,
                                                                    gp_max=global_progress_max,
                                                                    normalize_enum=normalize_enum,
                                                                    skip_predicted=skip_predicted,
                                                                    load_labels=False,
                                                                    n_jobs=n_jobs, force_verbose=True)

        if skipped:
            if skip_predicted:
                print('This folder was already predicted. Skipping.')
                continue
            else:
                print('This folder was already predicted. Predicting it again.')

        if len(loading_errors) > 0:
            print('Number of errors while loading: ' + str(len(loading_errors)))
            for i in range(len(loading_errors)):
                print('Error #' + str((i + 1)) + ': ' + str(loading_errors[i]))

        if len(X_to_predict) == 0:
            print(' ==[!! WARNING !!]== No wells have been loaded! Experiment skipped.')
            continue

        X_size = 0
        for i in range(len(X_to_predict)):
            X_size = X_size + X_to_predict[i].nbytes
        X_size = convert_size(X_size)
        print('Number of files loaded: ' + str(len(X_to_predict)) + '. Size in memory: ' + X_size)

        try:
            # print('X_to_predict len: ' + str(len(X_to_predict)))
            X_to_predict = np.asarray(X_to_predict)
            temp2 = X_to_predict.shape
            temp = np.moveaxis(X_to_predict, 1, 3)
            del temp
            del temp2
        except Exception as e:
            print(
                ' ==[!! WARNING !!]==\nFailed to convert X_to_predict to np array and determine its shape. This is a fatal error! Experiment skipped.')

            if isinstance(e, MemoryError):
                print('==[MEMORY ERROR]== Ran out of memory. This device has not enough RAM for the '+str(len(X_to_predict))+' files loaded!')

            if isinstance(X_to_predict, list):
                print('Failed to convert the data to numpy.')
            else:
                print('The dada was converted to numpy, but the shape could not be determined.')

            print('Exception: ' + str(e))
            del X_to_predict
            continue

        # process data
        print(gct() + " Loaded data to be predicted has shape: " + str(X_to_predict.shape) + '. Correcting axis.')
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

        if platform == "linux" or platform == "linux2":
            os.system('ls ' + str(folder) + ' > /dev/null')

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
                    print(f'Writing CSV: ' + str(filename), end="\r")
                    df = pd.read_csv(filename, delimiter=';')
                    df_length = len(df['label'])
                    df['label'] = binary_label[start_point:start_point + df_length]

                    # save with new name
                    df.to_csv(split_name + '_prediction.csv', sep=';', index=False)
                except Exception as e:
                    # TODO display error and stacktrace
                    try:
                        error_filename = path_to_csv + os.sep + split_name + '-error.txt'
                        ef = open(error_filename, 'w')
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


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def main(args):
    print('Number of arguments:', len(args), 'arguments.')
    print('Argument List:', str(args))

    custom_paths = False
    for arg in args:
        arg = str(arg).lower()
        print('Evaluating arg: "' + arg + '".')

        if arg == '-p' or arg == '-paths' or arg == '-c' or arg == '-custom':
            custom_paths = True

    if custom_paths:
        predict_batch_custom()
    else:
        prodi_gpu_predict()


def prodi_gpu_predict():
    print('\n_____________________________')
    print('Running Predictions.')
    use_oligo = True
    use_neuron = True
    use_glia = False
    use_old = False

    use_debug = False
    use_paper = True
    skip_predicted = True
    n_jobs: int = math.floor(int(multiprocessing.cpu_count()*1.15)+1)

    if sys.platform == 'win32':
        use_debug = True

    initial_sleep_time = 5
    print(' ## Predicting Neurons: ' + str(use_neuron))
    print(' ## Predicting Oligos: ' + str(use_oligo))
    print(' ## Multiprocessing on '+str(n_jobs)+' cores!')
    print(' == Initial Sleeping: ' + str(initial_sleep_time) + ' seconds ... ===')
    time.sleep(initial_sleep_time)

    if use_glia:
        predict_batch(model_source_path=model_source_path_glia, source_dir=source_dir_glia,
                      normalize_enum=4,
                      n_jobs=n_jobs,
                      skip_predicted=skip_predicted,
                      gpu_index_string="0")

    if use_paper:
        if use_neuron:
            for current_path in default_source_dirs_neuron:
                predict_batch(model_source_path=default_model_source_path_neuron, source_dir=current_path,
                              normalize_enum=4,
                              n_jobs=n_jobs,
                              skip_predicted=skip_predicted,
                              gpu_index_string="0")
        if use_oligo:
            for current_path in default_source_dirs_oligo:
                predict_batch(model_source_path=default_model_source_path_oligo, source_dir=current_path,
                              normalize_enum=4,
                              n_jobs=n_jobs,
                              skip_predicted=skip_predicted,
                              gpu_index_string="0")

    # if use_old:
    #     if use_neuron:
    #         predict_batch(model_source_path=model_source_path_neuron, source_dir=source_dir_neuron,
    #                       normalize_enum=1,
    #                       n_jobs=n_jobs,
    #                       skip_predicted=skip_predicted,
    #                       gpu_index_string="0")
    #     if use_oligo:
    #         predict_batch(model_source_path=model_source_path_oligo, source_dir=source_dir_oligo,
    #                       normalize_enum=1,
    #                       n_jobs=n_jobs,
    #                       skip_predicted=skip_predicted,
    #                       gpu_index_string="0")

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
