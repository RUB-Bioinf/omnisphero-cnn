'''Train a CNN to predict binary decisions on Omnisphero data.
Can be used for either neuron or oligo detection.

JOSHUA BUTKE, SEPTEMBER 2019
'''

# IMPORTS
#########
import numpy as np
import h5py
import os
import time

from tensorflow import keras

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras import optimizers, regularizers
from sklearn.metrics import roc_auc_score, auc
from keras.callbacks import Callback

from sklearn.metrics import roc_curve, roc_auc_score

from scramblePaths import *

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Custom Module
###############
import sys
sys.path.append('/bph/puredata1/bioinfdata/user/butjos/work/code/misc')

import misc_omnisphero as misc

# =========== List of all available neuron experiments on SAS15 ============================================================================
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'

# =========== List of all available oligo experiments on SAS15 ============================================================================
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK95_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK153_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK155_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK156_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM10_trainingData_oligo/'
# '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/MP70_trainingData_oligo/'

allNeurons = [
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/EKB5_trainingData_neuron/',
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ESM9_trainingData_neuron/'
            ]

allOligos = [
        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ELS81_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ELS79_BIS-I_NPC2-5_062_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK122_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK95_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK153_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK155_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/JK156_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/EKB5_trainingData_oligo/',
#'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ESM9_trainingData_oligo/',
#'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/ESM10_trainingData_oligo/',
'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/MP70_trainingData_oligo/'
            ]

# CNN MODELS
############
def omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format):
    """prototype model for single class decision
    """
    #Input
    img_input = Input(shape=(input_height, input_width, input_depth), name='input_layer')
    
    #Convolutional Blocks (FEATURE EXTRACTION)
    
    #Conv Block 1
    c1 = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv1', data_format=data_format)(img_input)
    bn1 = BatchNormalization(name='batch_norm_1')(c1)
    c2 = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(bn1)
    bn2 = BatchNormalization(name='batch_norm_2')(c2)
    p1 = MaxPooling2D((2,2), name='block1_pooling', data_format=data_format)(bn2)
    block1 = p1

    # Dave's Idee:
    # #Conv Block 1
    # c1 = Conv2D(32, (3,3), padding='same', name='block1_conv1', data_format=data_format)(img_input)
    # bn1 = BatchNormalization(name='batch_norm_1')(c1)
    # act1 = Activation('relu', alpha=0.0, max_value=None, threshold=0.0)(bn1)
    # 
    # c2 = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(act1)
    # bn2 = BatchNormalization(name='batch_norm_2')(c2)
    # p1 = MaxPooling2D((2,2), name='block1_pooling', data_format=data_format)(bn2)
    # block1 = p1

    #Conv Block 2
    c3 = Conv2D(64, (3,3), activation='relu', padding='same', name='block2_conv1', data_format=data_format)(block1)
    bn3 = BatchNormalization(name='batch_norm_3')(c3)
    c4 = Conv2D(64, (3,3), activation='relu', padding='same', name='block2_conv2', data_format=data_format)(bn3)
    bn4 = BatchNormalization(name='batch_norm_4')(c4)
    p2 = MaxPooling2D((2,2), name='block2_pooling', data_format='channels_last')(bn4)
    block2 = p2
    
    #Conv Block 3
    c5 = Conv2D(128, (3,3), activation='relu', padding='same', name='block3_conv1', data_format=data_format)(block2)
    bn5 = BatchNormalization(name='batch_norm_5')(c5)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same', name='block3_conv2', data_format=data_format)(bn5)
    bn6 = BatchNormalization(name='batch_norm_6')(c6)
    p3 = MaxPooling2D((2,2), name='block3_pooling', data_format='channels_last')(bn6)
    block3 = p3
    
    #Conv Block 4
    c7 = Conv2D(256, (3,3), activation='relu', padding='same', name='block4_conv1', data_format=data_format)(block3)
    bn7 = BatchNormalization(name='batch_norm_7')(c7)
    c8 = Conv2D(256, (3,3), activation='relu', padding='same', name='block4_conv2', data_format=data_format)(bn7)
    bn8 = BatchNormalization(name='batch_norm_8')(c8)
    p4 = MaxPooling2D((2,2), name='block4_pooling', data_format='channels_last')(bn8)
    block4 = p4
    
    #Fully-Connected Block (CLASSIFICATION)
    flat = Flatten(name='flatten')(block3)
    fc1 = Dense(256, activation='relu', name='fully_connected1')(flat)
    drop_fc_1 = Dropout(0.5)(fc1)
    
    prediction = Dense(n_classes, activation='sigmoid', name='output_layer')(drop_fc_1)
    
    #Construction
    model = Model(inputs=img_input, outputs=prediction)
    
    return model


# ROC stuff
# Source: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
class roc_callback(Callback):
    def __init__(self,training_data,validation_data,file_handle):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.f = file_handle


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #print('Calculating roc values')
        #y_pred = self.model.predict(self.x)
        #roc = roc_auc_score(self.y, y_pred)
        #y_pred_val = self.model.predict(self.x_val)
        #roc_val = roc_auc_score(self.y_val, y_pred_val)
        #self.f.write('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        #self.f.write('roc-auc: ' + str(round(roc,4))+ ' - roc-auc_val: ' + str(round(roc_val,4)) +'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# HARDCODED PARAMETERS
###############
#training_path_list = [
#        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS81_trainingData_neuron/',
#        #'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK125_trainingData_neuron/',
#        #'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/FJK130_trainingData_neuron/',
#        #'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK96_trainingData_neuron/',
#        #'/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/ELS79_BIS-I_NPC2-5_062_trainingData_neuron/'
#        #TODO
#            ]
#
#validation_path_list = [
#        '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/neuron/JK122_trainingData_neuron/'
#       ]
#
#
#label = 'ayy';
###############

# Initiating dummy variables
X = 0
y = 0
model = 0

scrambleResults = scramblePaths(allOligos,2)
outPath = '/bph/puredata4/bioinfdata/work/omnisphero/CNN/64x_unbalanced_histAdjusted_discard0/oligo/results/roc_results/';
print('Saving results here: ' + outPath)
os.makedirs(outPath,exist_ok=True);

print('sleeep [roc]...')
time.sleep(10)

for n in range(len(scrambleResults)):
    # Remove previous iteration
    del X
    del y
    del model

    scrambles = scrambleResults[n]
    label = scrambles['label']
    training_path_list = scrambles['train']
    validation_path_list = scrambles['val']
    outPathCurrent = outPath + str(n)+'_'+ label + os.sep
    os.makedirs(outPathCurrent,exist_ok=True);

    f = open(outPathCurrent + label + '_training_info.txt','w+')
    f.write('Label: '+label+'\nTraining paths:\n')
    for i in range(len(training_path_list)):
        f.write(training_path_list[i]+'\n')
    f.write('\nValidation paths:\n')
    for i in range(len(validation_path_list)):
        f.write(validation_path_list[i]+'\n')
    f.close()

    print('Round: ' + str(n+1) + '/' + str(len(scrambleResults)) + ' -> ' + label)
    print('Writing results here: ' + outPathCurrent)
    time.sleep(5)

    # TRAINING DATA
    ###############
    print("Traing data size: "+str(len(training_path_list)))
    X, y = misc.multiple_hdf5_loader(training_path_list) #load datasets
    print("Loaded data has shape: ")
    print(X.shape)
    print(y.shape)
    
    # # Loading temp data
    # data = np.load('/bph/puredata4/bioinfdata/work/omnisphero/CNN/temp/temp.npz')
    # X_pre = data.f.arr_0
    # y_pre = data.f.arr_1
    # data.close()
    # 
    # X = np.concatenate((X_pre, X), axis=0)
    # y = np.concatenate((y_pre, y), axis=0)
    # del X_pre, y_pre
    # # ==============================
    
    print("Correcting axes...")
    X = np.moveaxis(X,1,3)
    y = y.astype(np.int)
    print(X.shape)
    
    #np.savez('/bph/puredata4/bioinfdata/work/omnisphero/CNN/temp/temp2', X, y) 
    
    X = misc.normalize_RGB_pixels(X) #preprocess data
    
    # VALIDATION DATA
    #################
    print("Validation data size: " + str(len(validation_path_list)))
    X_val, y_val = misc.multiple_hdf5_loader(validation_path_list)
    #################
    
    print("Loaded validation data has shape: ")
    print(X_val.shape)
    print(y_val.shape)
    print("Correcting axes...")
    X_val = np.moveaxis(X_val,1,3)
    X_val = misc.normalize_RGB_pixels(X_val)
    y_val = y_val.astype(np.int)
    print(X_val.shape)
    print(y_val.shape)
    
    #####################################################################
    
    # HYPERPARAMETERS
    #################
    batch_size = 1000
    n_classes = 1
    input_height = 64
    input_width = 64
    input_depth = 3
    data_format = 'channels_last'
    optimizer_name = 'adadelta'
    learn_rate = 0.0001
    epochs = 40
    
    # CONSTRUCTION
    ##############
    
    print("Building model...")
    model = omnisphero_CNN(n_classes, input_height, input_width, input_depth, data_format)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=learn_rate), metrics=['accuracy'])
    model.summary()
    print("Model output shape: ", model.output_shape)
    
    #class weighting
    from sklearn.utils.class_weight import compute_class_weight
    
    y_order = y.reshape(y.shape[0])
    class_weights = compute_class_weight('balanced', np.unique(y), y_order)
    print("Class weights: ", class_weights)
    
    # TRAINING
    ##########
    
    f = open(outPathCurrent + label + '_roc.txt','w+')
    history_all = model.fit(X, y, 
                        validation_data=(X_val, y_val), 
                        callbacks=[roc_callback(training_data=(X, y),validation_data=(X_val, y_val),file_handle=f)],
                        epochs=epochs, batch_size=batch_size, 
                        class_weight=class_weights
                       )

    history=[np.zeros((epochs,4), dtype=np.float32)]
    history[0][:,0] = history_all.history['loss']
    history[0][:,1] = history_all.history['acc']
    history[0][:,2]= history_all.history['val_loss']
    history[0][:,3] = history_all.history['val_acc']
    f.close()
    
    # SAVING
    ########
    
    model.save(outPathCurrent + label + '.h5')
    model.save_weights(outPathCurrent + label + '_weights.h5')
    print('Saved model: ' + outPathCurrent + label)
    
    # Validate the trained model.
    scores = model.evaluate(X_val, y_val, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    #np.save(np.stack([history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']]),outPathCurrent + label + '_history.npy')
    
    # Plot training & validation accuracy values
    plt.plot(history_all.history['acc'])
    plt.plot(history_all.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(outPathCurrent + label + '_accuracy.png')
    plt.clf()
    print('Saved accuracy.')
    
    # Plot training & validation loss values
    plt.plot(history_all.history['loss'])
    plt.plot(history_all.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(outPathCurrent + label + '_loss.png')
    plt.clf()
    print('Saved loss.')

    np.save(outPathCurrent + label + "_history.npy", history)
    print('Saved history.')
    
    #print('Calculating roc curve.')
    y_pred_roc = model.predict(X_val).ravel()
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_val, y_pred_roc)
    auc_roc = auc(fpr_roc, tpr_roc)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_roc, tpr_roc, label='ROC (area = {:.3f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(outPathCurrent + label + '_roc.png')
    plt.clf()

    #print('Calculating roc curve.')
    #y_val_cat_prob = model.predict(X_val)
    #fpr, tpr, thresholds = roc_curve(y_val,y_val_cat_prob)

    #print('Saved roc curve.')
    #plt.plot(fpr,tpr)
    #plt.axis([0,1,0,1])
    #plt.title('Model loss')
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')

    #plt.savefig(outPathCurrent + label + 'roc.png')
    #plt.clf()
    
# END OF FILE
#############

print('Training done.')
