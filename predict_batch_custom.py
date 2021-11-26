import math
import multiprocessing
import os
import sys
import time


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
    input_data_source_path = str(input_data_source_path)
    # print('Your input: '+input_data_source_path)
    if not os.path.exists(input_data_source_path):
        print('That path does not exist.')
        return

    print('\nEnter the indexes of the gpu you would like to use (numbers separated by commas, if multiple):')
    input_gpus = input()

    n_jobs_reccomended: int = math.floor(int(multiprocessing.cpu_count() * 1.15) + 1)
    print(
        '\nEnter how many CPU cores you would like to use (type a number greater that zero, reccomended for this device: ' + str(
            n_jobs_reccomended) + '):')
    input_n_jobs = input()
    input_n_jobs = int(input_n_jobs)

    print('\n0 | no normalisation')
    print('1 | normalize every cell between 0 and 255 (8 bit)')
    print('2 | normalize every cell individually with every color channel independent')
    print('3 | normalize every cell individually with every color channel using the min / max of all three')
    print('4 | normalize every cell but with bounds determined by the brightest cell in the bag')
    print(
        'Enter the index for the normalization strategy to use (should match the strategy the model was trained with):')
    normalize_enum = input()
    normalize_enum = int(str(normalize_enum))

    print('\n\n')
    print('Starting cnn predictions.')
    time.sleep(1)

    from predict_batch import predict_batch
    predict_batch(model_source_path=input_model_source_path,
                  source_dir=input_data_source_path,
                  normalize_enum=normalize_enum,
                  n_jobs=input_n_jobs,
                  skip_predicted=True,
                  gpu_index_string=input_gpus)


def main(args):
    print('Predicting experiments with custom input data')
    print('Args (will be ignored): ' + str(args))
    predict_batch_custom()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
