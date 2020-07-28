# Omnisphero-CNN
This repo contains a few scripts to train, validate and save CNN models for the prediction of Neurons or/and Oligos on Omnisphero data.  
It can predict in batches on unannotated data to construct new '.csv' files.

#### train_model.py
- Give a comma seperated path list for training data which needs to be completly annotated
- Give a comma seperated path list for validation data which ALSO needs to be completly annotated
- Define the CNN model as you wish
- Set saving paths for the trained model, its weights and plots

#### predict_batch.py
- Give the location of a saved model which should be used to predict on new data
- Give the directory which contains unannotated data that should be labeled

## Required libraries
Please aquire the following libraries on your own, as they are not included within this repository:

 - numpy
 - pandas
 - keras
 - matplotlib
 - h5py
 - sklearn
 - imblearn

#### Conda Environment
Please look at the `/envs/` directory within this repository for the <i>Anaconda Environment</i> requirements file to recreate the environment used within this codebase.