''' Give Omnisphero-CNN predictions for a given model on a folder of csv data.

JOSHUA BUTKE, AUGUST 2019
'''
# IMPORTS
#########
import os
import sys
sys.path.append('/bph/puredata1/bioinfdata/user/butjos/work/code/misc')

import misc_omnisphero as misc
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Imports done...")

# LOAD MODEL
############
model = load_model(#TODO)

print("Loaded model...")

# MAIN LOOP
###########

path = #TODO

#construct directory walker
dir_list = [x[0] for x in os.walk(path)]

print("Constructed directory walker...")

for folder in dir_list[1:]:
    if "unannotated" in folder:
        pass
    else:
        continue
    #load data
    X_to_predict, _ = misc.hdf5_loader(str(folder))
#process data
    X_to_predict = np.asarray(X_to_predict)
    print('Loaded data at: ',str(folder))
    print(X_to_predict.shape)
    X_to_predict = np.moveaxis(X_to_predict,1,3)
    X_to_predict = misc.normalize_RGB_pixels(X_to_predict)
    print(X_to_predict.shape)

    #generate prediction
    print('Generating predictions...')
    label = model.predict(X_to_predict, verbose=1)
    binary_label = misc.sigmoid_binary(label)
    misc.count_uniques(binary_label)
    
    #cleanup / free memory
    del X_to_predict

    # CSV Manipulation
    ##################
    path_to_csv = str(folder)
    os.chdir(path_to_csv)
    directory_csv = os.fsencode(path_to_csv)
    directory_csv_contents = os.listdir(directory_csv)
    directory_csv_contents.sort()
    
    start_point = 0
    
    for f in directory_csv_contents:
        filename = os.fsdecode(f)
        if filename.endswith('.csv') and not filename.endswith('_prediction.csv'):
            #reading
            print('Manipulating: ', filename)
            df = pd.read_csv(filename, delimiter=';')
            df_length = len(df['label'])
            df['label'] = binary_label[start_point:start_point + df_length]
    
            #save with new name
            split_name = filename.split('.')
            df.to_csv(split_name[0] + '_prediction.csv', sep=';', index=False)
    
            #update start point
            start_point += df_length

# END OF FILE
#############
