# Omnisphero-CNN
This repo contains a few scripts to train, validate and save CNN models for the prediction of Neurons or/and Oligos on Omnisphero data.  
It can predict in batches on unannotated data to construct new '.csv' files.

### train_model.py
- Give a comma seperated path list for training data which needs to be completly annotated
- Give a comma seperated path list for validation data which ALSO needs to be completly annotated
- Define the CNN model as you wish
- Set saving paths for the trained model, its weights and plots

### predict_batch.py
- Give the location of a saved model which should be used to predict on new data
- Give the directory which contains unannotated data that should be labeled

