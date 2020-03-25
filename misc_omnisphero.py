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


# FUNCTION DEFINITONS
#####################

def hdf5_loader(path, pattern='_[A-Z][0-9]{2}_', suffix_data='.h5', suffix_label='_label.h5', gpCurrent = 0, gpMax = 0):
    '''Helper function which loads all datasets from a hdf5 file in
    a specified file at a specified path.

    # Arguments
        The split argument is used to split the key-string and sort alphanumerically
        instead of sorting the Python-standard way of 1,10,2,9,...
        The two suffix arguments define the way the datasets are looked up:
        (Training) data should always end in .h5 and corresponding labels should
        carry the same name and end in _label.h5

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

    file_count = len(directory_contents)
    for i in range(file_count):
        filename = os.fsdecode(directory_contents[i])

        if filename.endswith(suffix_label):
            # print("Opening: ", filename, "\n")

            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key=lambda a: int(re.split(pattern, a)[1].split('_')[0]))

                for key in key_list:
                    # print("Loading dataset associated with key ", str(key))
                    print(f"Reading label file: " + str(i) + " / " + str(file_count) + ": " + filename + " - Current dataset key: " + str(key) + " [" + str(gpCurrent) + "/" + str(gpMax)+"]   ", end="\r")
                    y.append(np.array(f[str(key)]))
                f.close()
                # print("\nClosed ", filename, "\n")
                continue

        elif filename.endswith(suffix_data) and not filename.endswith(suffix_label):
            # print("Opening: ", filename, "\n")

            with h5py.File(filename, 'r') as f:
                key_list = list(f.keys())
                key_list.sort(key=lambda a: int(re.split(pattern, a)[1]))

                for key in key_list:
                    # print("Loading dataset associated with key ", str(key))
                    print(f"Reading data file: " + str(i) + " / " + str(file_count) + ": " + filename + "                         - Current dataset key: " + str(key) + " [" + str(gpCurrent) + "/" + str(gpMax)+"]   ", end="\r")
                    X.append(np.array(f[str(key)]))
                f.close()
                # print("\nClosed ", filename, "\n")

    # Dummy prints to make space for the next prints
    print('')
    return X, y


###

def multiple_hdf5_loader(path_list, pattern='_[A-Z][0-9]{2}_', suffix_data='.h5', suffix_label='_label.h5',gpCurrent=0,gpMax=0):
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
        X, y = hdf5_loader(path, pattern, suffix_data, suffix_label)
        X = np.asarray(X)
        y = np.asarray(y)
        X_full = np.concatenate((X_full, X), axis=0)
        y_full = np.concatenate((y_full, y), axis=0)
        print("Finished with loading dataset located at: ", path + " [" + str(gpCurrent) + "/" + str(gpMax)+"]")

    return X_full, y_full


###

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
    return (ndarr.astype(float) / 255.0)


###

def check_predicted_classes(labels, predictions):
    '''
    '''

###

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]