import sys
import time
import math
import multiprocessing

import os


def predict_batch_custom():
    print('\n === === === NOTE: TO CANCEL, PRESS CTRL+C AT ANY TIME === === ===\n')
    time.sleep(2)

    print('Confirm your inputs by pressing enter.')
    time.sleep(1)
    print('Enter the path your model to use:')
    input_model_source_path = input()
    # print('Your input: '+input_model_source_path)

    input_model_source_path = str(input_model_source_path)
    if not os.path.exists(input_model_source_path):
        print('That file does not exist.')
        return

    print('\nEnter path your batches are located in:')
    input_data_source_path = input()
    input_data_source_path=str(input_data_source_path)
    # print('Your input: '+input_data_source_path)
    if not os.path.exists(input_data_source_path):
        print('That path does not exist.')
        return

    print('\nEnter the indexes of the gpu you would like to use (numbers separated by commas, if multiple):')
    input_gpus = input()

    n_jobs_reccomended: int = math.floor(int(multiprocessing.cpu_count()*1.15)+1)
    print('\nEnter how many CPU cores you would like to use (type a number greater that zero, reccomended for this device: '+str(n_jobs_reccomended)+'):')
    input_n_jobs = input()
    input_n_jobs = int(input_n_jobs)

    print('\n\n')

    from predict_batch import predict_batch
    predict_batch(model_source_path=input_model_source_path,
                  source_dir=input_data_source_path,
                  normalize_enum=4,
                  n_jobs=input_n_jobs,
                  skip_predicted=True,
                  gpu_index_string=input_gpus)


def main(args):
    print('Predicting experiments with custom input data')
    predict_batch_custom()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
