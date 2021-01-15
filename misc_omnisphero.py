# Nils Foerster
# Joshua Butke
# 2019 - 2020
##############


"""This script contains miscellaneous helper functions
to be used. Some might work, others might not...

PROJECT: Omnisphero CNN

"""

# IMPORTS
########

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling.smote import SMOTE

from misc_cnn import gct, get_time_diff

# conda install -c glemaitre imbalanced-learn
# Since conda is absolute bs, use this command to download imblearn from an external source
# Stackoverflow answer: https://stackoverflow.com/questions/40008015/problems-importing-imblearn-python-package-on-ipython-notebook
# There is a special place in hell for conda devs, alongside MATLAB devs

hdf5_loeader_default_param_pattern: str = '_[A-Z][0-9]{2}_'
hdf5_loeader_default_param_suffix_data: str = '.h5'
hdf5_loeader_default_param_suffix_label: str = '_label.h5'
hdf5_loeader_default_param_gp_current: int = 0
hdf5_loeader_default_param_gp_max: int = 0
hdf5_loeader_default_param_normalize_enum: int = 1
hdf5_loeader_default_param_verbose: bool = True


# FUNCTION DEFINITIONS
#####################

def hdf5_loader(path: str, pattern: str = '_[A-Z][0-9]{2}_', suffix_data: str = '.h5', suffix_label: str = '_label.h5',
                gp_current: int = 0, gp_max: int = 0, normalize_enum: int = 1, n_jobs: int = 1,
                skip_predicted: bool = False, force_verbose: bool = False):
    '''Helper function which loads all datasets from a hdf5 file in
    a specified file at a specified path.

    # Arguments
        The split argument is used to split the key-string and sort alphanumerically
        instead of sorting the Python-standard way of 1,10,2,9,...
        The two suffix arguments define the way the datasets are looked up:
        (Training) data should always end in .h5 and corresponding labels should
        carry the same name and end in _label.h5
        normalize_enum is an enum to determine normalisation as follows:
         0 = no normalisation
         1 = normalize between 0 and 255
         2 = normalize every cell individually with every color channel independent
         3 = normalize every cell individually with every color channel using the min / max of all three
         4 = normalize every cell but with bounds determined by the brightest cell in the same well

    # Returns
        X and y lists containing all of the data.

    # Usage
        path = 'path/to/folder'
        X, y = hdf5_loader(path, split=3)
        X = np.asarray(X)
        y = np.asarray(y)
        print(X.shape)
        print(y.shape)
    '''

    X = []
    y = []

    os.chdir(path)
    directory = os.fsencode(path)
    directory_contents = os.listdir(directory)
    directory_contents.sort()
    # well_regex = "(\w\d+)_\d+$"
    # well_regex = re.compile(well_regex)

    try:
        terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()
        terminal_columns = int(terminal_columns)
    except Exception as e:
        terminal_columns = None

    n_jobs = max(n_jobs, 1)
    verbose: bool = n_jobs == 1
    if force_verbose:
        verbose = force_verbose

    file_count = len(directory_contents)
    if verbose:
        print('Reading ' + str(file_count) + ' files with normalization strategy index: ' + str(
            normalize_enum) + '. Skipping already predicted: ' + str(skip_predicted))

    executor = ThreadPoolExecutor(max_workers=n_jobs)
    future_list = []
    worker_verbose: bool = n_jobs == 1

    if verbose:
        print('Setting up futures for path: ' + path)
    for i in range(file_count):
        filename = os.fsdecode(directory_contents[i])
        if verbose:
            line_print(str(i) + '/' + str(file_count) + ': Preparing future to load: ' + filename)

        future = executor.submit(hdf5_loader_worker,  # the actual function i want to run. Args, see below
                                 filename,  # filename
                                 path,  # path
                                 pattern,  # pattern
                                 suffix_data,  # suffix_data
                                 suffix_label,  # suffix_label
                                 gp_current,  # gp_current
                                 gp_max,  # gp_max
                                 normalize_enum,  # normalize_enum
                                 worker_verbose,  # verbose
                                 terminal_columns  # terminal_columns#
                                 )
        future_list.append(future)

        # Running through the future list again to check if preivously added futures have already finished.
        # On ProDi Server "Ehrlich", adding tasks is so slow, it might be more efficient to check even while adding futures. Wow.
        for ft in future_list:
            if ft.done():
                e = ft.exception()
                if e is None:
                    _, _, prediction_future = ft.result()
                    if prediction_future and skip_predicted:
                        if verbose:
                            print(
                                '\n' + 'While adding tasks: Predicted well(s) already found!! Skipping this experiment.')
                        executor.shutdown(wait=False)
                        return None, None, None, True

    if verbose:
        print(gct() + ' Starting to read ' + str(file_count) + ' label / data .h5 files on ' + str(
            n_jobs) + ' thread(s).')
    start_time = gct(raw=True)
    all_finished: bool = False
    executor.shutdown(wait=False)

    while not all_finished:
        finished_count = 0
        predicted_count = 0
        error_count = 0

        for future in future_list:
            if future.done():
                finished_count = finished_count + 1

                e = future.exception()
                if e is None:
                    _, _, prediction_future = future.result()
                    if prediction_future:
                        predicted_count = predicted_count + 1
                else:
                    error_count = error_count + 1

        if verbose:
            line_print('[' + str(gp_current) + ' / ' + str(gp_max) + '] ' + str(
                n_jobs) + ' Threads running. Finished: ' + str(finished_count) + '/' + str(
                len(future_list)) + '. Already predicted: ' + str(predicted_count) + '. Errors: ' + str(
                error_count) + '. Running: ' + get_time_diff(
                start_time) + '. ' + gct(), max_width=terminal_columns)
        all_finished = finished_count == len(future_list)
        time.sleep(1)

        if skip_predicted and predicted_count > 0:
            if verbose:
                print('\n' + str(predicted_count) + ' predicted well(s) found. Skipping this experiment.\n')
            executor.shutdown(wait=False)
            return None, None, None, True

    if verbose:
        print('\n' + gct() + ' Finished concurrent execution. Fetching results.')

    error_list = []
    for i in range(len(future_list)):
        future = future_list[i]
        if verbose:
            line_print('Extracting future: '+str(i)+'/'+str(len(future_list)))

        e = future.exception()
        if e is None:
            X_w, y_w, prediction_file = future.result()
            if X_w is not None:
                X.extend(X_w)
            if y_w is not None:
                y.extend(y_w)
        else:
            print('\n' + gct() + 'Error extracting future results: ' + str(e)+'\n')
            error_list.append(e)

    if verbose:
        print(gct() + ' Fully Finished Loading Path.')

    # Deleting the futures and the future list to immediately releasing the memory.
    del future_list[:]
    del future_list

    return X, y, error_list, False


###

def hdf5_loader_worker(filename: str, path: str, pattern: str, suffix_data: str, suffix_label: str,
                       gp_current: int, gp_max: int, normalize_enum: int, verbose: bool, terminal_columns: int):
    worker_x = None
    worker_y = None
    prediction_file = False
    best_well_min = [255, 255, 255]
    best_well_max = [0, 0, 0]
    pattern = re.compile(pattern)

    well_regex = "(\w\d+)_\d+$"
    well_regex = re.compile(well_regex)

    if filename.endswith(suffix_label):
        worker_y = []
        with h5py.File(path + os.sep + filename, 'r') as f:
            key_list = list(f.keys())
            key_list.sort(key=lambda a: int(re.split(pattern, a)[1].split('_')[0]))

            for key in key_list:
                if verbose:
                    line_print("Reading label file: " + filename + " - Current dataset key: " + str(key) + " [" + str(
                        gp_current) + "/" + str(gp_max) + "]", max_width=terminal_columns)
                worker_y.append(np.array(f[str(key)]))
            f.close()
        # done evaluating y file
    elif filename.endswith(suffix_data) and not filename.endswith(suffix_label):
        worker_x = []
        with h5py.File(path + os.sep + filename, 'r') as f:
            key_list = list(f.keys())
            key_list.sort(key=lambda a: int(re.split(pattern, a)[1]))

            for k in range(len(key_list)):
                key = key_list[k]
                # print("Loading dataset associated with key ", str(key))
                current_well = re.split(well_regex, key)[1]

                if verbose:
                    line_print("Reading data file: " + filename + " - Current dataset key: " + str(
                        key) + " Well: " + current_well + " [" + str(gp_current) + "/" + str(gp_max) + "]",
                               max_width=terminal_columns)

                current_x = np.array(f[str(key)])
                if normalize_enum == 0:
                    pass
                elif normalize_enum == 1:
                    current_x = normalize_np(current_x, 0, 255)
                elif normalize_enum == 2:
                    current_x[0] = normalize_np(current_x[0], current_x[0].min(), current_x[0].max())
                    current_x[1] = normalize_np(current_x[1], current_x[1].min(), current_x[1].max())
                    current_x[2] = normalize_np(current_x[2], current_x[2].min(), current_x[2].max())
                elif normalize_enum == 3:
                    current_x[0] = normalize_np(current_x[0], current_x.min(), current_x.max())
                    current_x[1] = normalize_np(current_x[1], current_x.min(), current_x.max())
                    current_x[2] = normalize_np(current_x[2], current_x.min(), current_x.max())
                elif normalize_enum == 4:
                    best_well_max[0] = max(best_well_max[0], current_x[0].max())
                    best_well_max[1] = max(best_well_max[1], current_x[1].max())
                    best_well_max[2] = max(best_well_max[2], current_x[2].max())

                    best_well_min[0] = min(best_well_min[0], current_x[0].min())
                    best_well_min[1] = min(best_well_min[1], current_x[1].min())
                    best_well_min[2] = min(best_well_min[2], current_x[2].min())
                    # TODO: Implement
                else:
                    raise Exception('Undefined state of normalize_enum')

                worker_x.append(np.array(current_x))
            f.close()

        if normalize_enum == 4:
            for j in range(len(worker_x)):
                if verbose:
                    line_print('Normalizing well entry ' + str(j) + ' / ' + str(
                        len(worker_x)), max_width=terminal_columns)
                worker_x[j][0] = normalize_np(worker_x[j][0], best_well_min[0], best_well_max[0])
                worker_x[j][1] = normalize_np(worker_x[j][1], best_well_min[1], best_well_max[1])
                worker_x[j][2] = normalize_np(worker_x[j][2], best_well_min[2], best_well_max[2])

        # Done evaluating X file
    elif filename.endswith('_prediction.csv'):
        prediction_file = True
        worker_x = None
        worker_y = None
    else:
        worker_x = None
        worker_y = None
        # print("Unknown file type. Skipping: " + filename)

    return worker_x, worker_y, prediction_file


###

def multiple_hdf5_loader(path_list: [str], pattern: str = '_[A-Z][0-9]{2}_', suffix_data: str = '.h5',
                         suffix_label: str = '_label.h5', n_jobs: int = 1, single_thread_loading: bool = False,
                         gp_current: int = 0, gp_max: int = 0, normalize_enum: int = 1, force_verbose: bool = False):
    '''Helper function which loads all datasets from targeted hdf5 files in
    a specified folder. Returns X and y arrays containing all of them.
    This function uses hdf5_loader.

    # Usage
        path_list = ['path/to/folder/file_1',
                     'path/to/folder/file_2,
                      ...
                    ]
        split_list = [int_1,int_2,...]

        X, y = multiple_hdf5_loader(path_list, split_list)
    
        print(X.shape)
        print(y.shape)
    '''

    try:
        terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()
        terminal_columns = int(terminal_columns)
    except Exception as e:
        terminal_columns = None

    n_jobs = max(n_jobs, 1)
    if single_thread_loading:
        n_jobs = 1

    X_full = np.empty((0, 3, 64, 64))
    y_full = np.empty((0, 1))

    l = len(path_list)
    i = 1
    executor = ThreadPoolExecutor(max_workers=n_jobs)
    future_list = []
    verbose: bool = n_jobs == 1
    if force_verbose:
        verbose = True

    worker_threads = int((n_jobs - (l / 2)) / l)
    worker_threads = max(worker_threads, 1)
    print('Distributing ' + str(l) + ' managers with ' + str(worker_threads) + ' workers each from a pool of ' + str(
        n_jobs))

    for path in path_list:
        print("Preparing to load dataset at: ", path)
        # X, y = hdf5_loader(path, pattern, suffix_data, suffix_label, normalize_enum=normalize_enum, gp_current=i, gp_max=l)
        future = executor.submit(hdf5_loader,  # the actual function i want to run. Args, see below
                                 path,
                                 pattern,  # hdf5_loeader_default_param_pattern
                                 suffix_data,  # hdf5_loeader_default_param_suffix_data
                                 suffix_label,  # hdf5_loeader_default_param_suffix_label
                                 i,  # hdf5_loeader_default_param_gp_current
                                 l,  # hdf5_loeader_default_param_gp_max
                                 normalize_enum,  # hdf5_loeader_default_param_normalize_enum
                                 worker_threads,  # n_jobs: int = 1
                                 False,  # skip_predicted
                                 force_verbose  # hdf5_loeader_default_param_verbose
                                 )
        future_list.append(future)
        i = i + 1

    print(gct() + ' Loading ' + str(l) + ' data-set(s) on ' + str(n_jobs) + ' thread(s)! Progress is indeterminable.\n')
    start_time = gct(raw=True)
    all_finished: bool = False
    executor.shutdown(wait=False)

    while not all_finished:
        finished_count = 0
        error_count = 0

        for future in future_list:
            if future.done():
                finished_count = finished_count + 1
                e = future.exception()
                if e is None:
                    _, _, errors, _ = future.result()
                    error_count = error_count + len(errors)
                else:
                    error_count = error_count + 1

        line_print('Loader-Manager Threads running. Finished: ' + str(finished_count) + '/' + str(
            len(future_list)) + '. Errors: ' + str(error_count) + '. Running: ' + get_time_diff(
            start_time) + '. ' + gct(), max_width=terminal_columns)

        all_finished = finished_count == len(future_list)
        time.sleep(1)

        # if predicted_count > 0 and skip_predicted:
        #     print('Found an already predicted well. Skipping this experiment. Shutting down threads.')
        #     all_finished = True

        #     executor.shutdown(wait=False, cancel_futures=True)
        #     return X_full, y_full, error_list, True

    print(
        '\n' + gct() + ' Finished concurrent execution in ' + get_time_diff(start_time) + ' minutes. Fetching results.')

    error_list = []
    for future in future_list:
        e = future.exception()
        if e is None:
            X, y, errors, _ = future.result()
            X = np.asarray(X)
            y = np.asarray(y)
            X_full = np.concatenate((X_full, X), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)

            if len(errors) > 0:
                error_list.extend(errors)
        else:
            print('Error extracting future results: ' + str(e))
            error_list.append(e)

    print(gct() + ' Finished Loading.')
    return X_full, y_full, error_list, False


def line_print(text: str, max_width: int = None, cutoff_too_large_text: bool = True):
    text = str(text)

    if max_width is None:
        try:
            terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()
            terminal_rows = int(terminal_rows)
            terminal_columns = int(terminal_columns)
            # print('Did you know your terminal is ' + str(terminal_columns) + 'x' + str(terminal_rows) + ' characters big.')
        except Exception as e:
            print(text)
            return
    else:
        terminal_columns = max_width

    out_s = ''
    if cutoff_too_large_text:  # and len(text) < terminal_columns:
        for i in range(terminal_columns - 1):
            if i < len(text):
                out_s = out_s + text[i]
            else:
                out_s = out_s + ' '
        # out_s = out_s + ' ' + str(len(out_s))
    else:
        out_s = str(text)

    print(out_s, end="\r")


def save_random_samples(X, y, count: int, path: str):
    os.makedirs(path, exist_ok=True)
    print('y==0 count: ' + str(np.count_nonzero(y == 0)))

    y0_all = list(np.where(y == 0)[0])
    y1_all = list(np.where(y == 1)[0])

    for i in range(count):
        y0_current = random.choice(y0_all)
        y1_current = random.choice(y1_all)

        x0_current = X[y0_current]
        x1_current = X[y1_current]

        plt.imsave(path + os.sep + '0_' + str(i + 1) + '.png', x0_current)
        plt.imsave(path + os.sep + '1_' + str(i + 1) + '.png', x1_current)


###

def normalize_np(nparr: np.ndarray, lower=0, upper=255):
    nnv = np.vectorize(normalize_np_worker)
    return nnv(nparr, lower, upper)


def normalize_np_worker(x: float, lower: float, upper: float):
    if lower == upper:
        return 0

    lower = float(lower)
    upper = float(upper)
    return (x - lower) / (upper - lower)


def sigmoid_binary(ndarr):
    '''Transform ndarray entries into 0 if they are <= 0.5
    or 1 if they are > 0.5
    Returns the transformed array.
    '''
    result = np.where(ndarr <= 0.5, 0, 1)
    return result


###

def count_uniques(ndarr):
    '''Counts the occurence of items in an ndarray
    Outputs {item:count,item2:count2,...}
    '''
    unique, counts = np.unique(ndarr, return_counts=True)
    result = dict(zip(unique, counts))
    print('Counting uinques: ' + str(result))

    return result


###

def normalize_rgb_pixels(ndarr):
    '''normalize RGB pixel values ranging
    from 0-255 into a range of [0,1]
    '''

    raise Exception('Depreciated')
    # return (ndarr.astype(float) / 255.0)


###

def check_predicted_classes(labels, predictions):
    '''
    '''


###

# SMOTE default params:
# ratio: Any = 'auto',
# random_state: Any = None,
# k: Any = None,
# k_neighbors: Any = 5,
# m: Any = None,
# m_neighbors: Any = 10,
# out_step: Any = 0.5,
# kind: Any = 'regular',
# svm_estimator: Any = None,
# n_jobs: Any = 1
def create_smote_handler(random_state: int = None, n_jobs: int = 1, k_neighbors: int = 5) -> SMOTE:
    if random_state is None:
        random_state = int(time.time())
    sm = SMOTE(random_state=random_state, n_jobs=n_jobs, k_neighbors=k_neighbors)
    return sm


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def main():
    print("Thanks for running this function, but it actually does nothing. Have a nice day. =)")

    # import time
    # line_print("test")
    # time.sleep(1)
    # line_print("test2")
    # time.sleep(1)
    # line_print("a")
    # time.sleep(1)
    # print('\n')


if __name__ == "__main__":
    main()
