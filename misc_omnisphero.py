# Joshua Butke
# July 2019
##############

"""This script contains miscellaneous helper functions
to be used. Some might work, others might not...

PROJECT: Omnisphero CNN

"""

# IMPORTS
########

import os
import numpy as np
import h5py
import re
import matplotlib.pyplot as plt


# FUNCTION DEFINITONS
#####################

def hdf5_loader(path, pattern='_[A-Z][0-9]{2}_', suffix_data='.h5', suffix_label='_label.h5',
                gpCurrent=0, gpMax=0, normalize_enum=1):
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

    pattern = re.compile(pattern)

    well_regex = "(\w\d+)_\d+$"
    well_regex = re.compile(well_regex)

    file_count = len(directory_contents)
    for i in range(file_count):
        filename = os.fsdecode(directory_contents[i])

        last_known_well_index = len(X)
        best_well_min = [255, 255, 255]
        best_well_max = [0, 0, 0]

        if filename.endswith(suffix_label):
            # print("Opening: ", filename, "\n")

            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key=lambda a: int(re.split(pattern, a)[1].split('_')[0]))

                for key in key_list:
                    # print("Loading dataset associated with key ", str(key))
                    print(f"Reading label file: " + str(i) + " / " + str(
                        file_count) + ": " + filename + " - Current dataset key: " + str(key) + " [" + str(
                        gpCurrent) + "/" + str(gpMax) + "]   ", end="\r")
                    y.append(np.array(f[str(key)]))
                f.close()
                # print("\nClosed ", filename, "\n")
                continue
        elif filename.endswith(suffix_data) and not filename.endswith(suffix_label):
            # print("Opening: ", filename, "\n")

            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key=lambda a: int(re.split(pattern, a)[1]))

                for k in range(len(key_list)):
                    key = key_list[k]
                    # print("Loading dataset associated with key ", str(key))

                    current_well = re.split(well_regex, key)[1]
                    print(f"Reading data file: " + str(i) + " / " + str(
                        file_count) + ": " + filename + "                         - Current dataset key: " + str(
                        key) + " Well: " + current_well + " [" + str(gpCurrent) + "/" + str(gpMax) + "]   ", end="\r")

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

                    X.append(np.array(current_x))
                f.close()
                # print("\nClosed ", filename, "\n")

            if normalize_enum == 4:
                for j in range(last_known_well_index, len(X)):
                    print(f'Normalizing well entry ' + str(j - last_known_well_index) + ' / ' + str(
                        len(X) - last_known_well_index) + '     <', end="\r")
                    X[j][0] = normalize_np(X[j][0], best_well_min[0], best_well_max[0])
                    X[j][1] = normalize_np(X[j][1], best_well_min[1], best_well_max[1])
                    X[j][2] = normalize_np(X[j][2], best_well_min[2], best_well_max[2])

            # Done evaluating X file
        else:
            pass
            # print("Unknown file type. Skipping: " + filename)

    # Dummy prints to make space for the next prints
    print('')
    return X, y


###

def multiple_hdf5_loader(path_list, pattern='_[A-Z][0-9]{2}_', suffix_data='.h5', suffix_label='_label.h5', gpCurrent=0,
                         gpMax=0, normalize_enum=1):
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

    X_full = np.empty((0, 3, 64, 64))
    y_full = np.empty((0, 1))

    for path in path_list:
        print("\nIterating over dataset at: ", path)
        X, y = hdf5_loader(path, pattern, suffix_data, suffix_label, normalize_enum=normalize_enum)
        X = np.asarray(X)
        y = np.asarray(y)
        X_full = np.concatenate((X_full, X), axis=0)
        y_full = np.concatenate((y_full, y), axis=0)
        print("Finished with loading dataset located at: ", path + " [" + str(gpCurrent) + "/" + str(gpMax) + "]")

    return X_full, y_full


###

def normalize_np(nparr, lower=0, upper=255):
    nnv = np.vectorize(normalize_np_worker)
    return nnv(nparr, lower, upper)


def normalize_np_worker(x, lower, upper):
    if lower == upper:
        return 0
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
    print(result)

    return result


###

def normalize_RGB_pixels(ndarr):
    '''normalize RGB pixel values ranging
    from 0-255 into a range of [0,1]
    '''
    
    raise Exception('Depreciated')
    return (ndarr.astype(float) / 255.0)


###

def check_predicted_classes(labels, predictions):
    '''
    '''


###

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
