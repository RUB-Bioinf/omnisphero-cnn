import sys
import time

import os


def predict_batch_custom():
    print('\n === === === NOTE: TO CANCEL, PRESS CTRL+C AT ANY TIME === === ===\n')
    time.sleep(2)

    print('Confirm your inputs by pressing enter.')
    print('Type the path your model to use:')
    input_model_source_path = input()
    # print('Your input: '+input_model_source_path)

    if not os.path.exists(input_model_source_path):
        print('That file does not exist.')
        return

    print('\nType path your batches are located in:')
    input_data_source_path = input()
    # print('Your input: '+input_data_source_path)
    if not os.path.exists(input_data_source_path):
        print('That path does not exist.')
        return

    print('\nType the indexes of the gpu you would like to use (numbers separated by commas, if multiple):')
    input_gpus = input()

    print('\nType how many CPU cores you would like to use (type a number greater that zero):')
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
