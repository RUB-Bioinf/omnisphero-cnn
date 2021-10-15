import os
import numpy as np
import sys
import platform


############
# This function is very project specific and contains project paths.
# Running this is not necessary

def main():
    prediction_path = None
    bilderordner_path = None
    if platform.system() == 'Windows':
        prediction_path = 'Z:\\nilfoe\\Temp\\temp_predictions'
        bilderordner_path = 'U:\\bioinfdata\\work\\OmniSphero\\Bilderordner'
    else:
        prediction_path = ''
        bilderordner_path = ''

    print('Starting to move predictions.')
    move_predictions(prediction_path=prediction_path, bilderordner_path=bilderordner_path, copy_files=True)


def move_predictions(prediction_path: str, bilderordner_path: str, copy_files: bool = True):
    directories = os.listdir(prediction_path)
    for directory in directories:
        path = prediction_path + os.sep + directory
        print(path)


if __name__ == '__main__':
    main()
    print('All finished.')
