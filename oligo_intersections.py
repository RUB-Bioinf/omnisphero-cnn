import os

import numpy as np
from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

import misc_cnn
import misc_omnisphero as misc
from predict_batch import default_model_source_path_neuron
from predict_batch import default_model_source_path_oligo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    img_dpi = 500
    out_path = '/bph/puredata4/bioinfdata/work/OmniSphero/Sciebo/HCA/07_CNN_TrainingData/debug/'

    test_oligo = '/prodi/bioinfdata/work/Omnisphero/CNN/diff/data/train/final/oligo/test/'
    test = '/prodi/bioinfdata/work/Omnisphero/CNN/diff/data/train/final/neuron/test/'

    model_source_path_neuron = default_model_source_path_neuron
    model_source_path_oligo = default_model_source_path_oligo
    oligo_intersections(model_source_path_neuron=model_source_path_neuron,
                        model_source_path_oligo=model_source_path_oligo, test_oligo=test_oligo, test=test,
                        img_dpi=img_dpi)


def oligo_intersections(model_source_path_neuron: str, model_source_path_oligo: str, test_oligo: str, test: str,
                        img_dpi: int):
    print('Loading Models.')
    model = load_model(model_source_path_neuron + 'model.h5')
    model.load_weights(model_source_path_neuron + 'weights_best.h5')

    model_oligo = load_model(model_source_path_oligo + 'model.h5')
    model_oligo.load_weights(model_source_path_oligo + 'weights_best.h5')

    print('Loading Data.')
    X_test_oligo, y_test_oligo, loading_errors, skipped = misc.hdf5_loader(test_oligo, gp_current=1,
                                                                           gp_max=1,
                                                                           normalize_enum=4,
                                                                           # load_samples=False,
                                                                           n_jobs=15, force_verbose=False)

    X_test, y_test, test_loading_errors, _ = misc.hdf5_loader(test, gp_current=1, gp_max=1,
                                                              normalize_enum=4, n_jobs=15,
                                                              force_verbose=False)
    print('Finished loading all data.')

    # Sanity check on the data. Data must be loaded and array size of y_test and y_test_oligo must match.
    assert len(X_test) > 0
    assert len(y_test) > 0
    assert len(y_test_oligo) > 0
    assert len(y_test) == len(y_test_oligo)

    print('Done. Preprocessing test data.')
    y_test = np.asarray(y_test)
    y_test_oligo = np.asarray(y_test_oligo)
    y_test = y_test.astype(np.int)
    y_test_oligo = y_test_oligo.astype(np.int)

    X_test = np.asarray(X_test)
    print(X_test.shape)
    X_test = np.moveaxis(X_test, 1, 3)

    X_test_oligo = np.asarray(X_test_oligo)
    print(X_test_oligo.shape)
    X_test_oligo = np.moveaxis(X_test_oligo, 1, 3)

    print("Loaded test data has shape: ")
    print(X_test.shape)
    print(X_test_oligo.shape)
    print(y_test.shape)

    # Intersections
    print('Predicting...')
    y_pred = model.predict(X_test)
    y_pred_olgio = model_oligo.predict(X_test_oligo)

    print('Predictions done.')
    y_pred_binary = misc.sigmoid_binary(y_pred)
    y_pred_binary_oligo = misc.sigmoid_binary(y_pred_olgio)

    assert y_pred_binary_oligo.shape == y_pred_binary.shape
    assert y_pred_binary.shape == y_test.shape
    assert y_pred_binary.shape == y_test_oligo.shape
    del X_test

    intersection_count = 0
    intersection_count_training = 0
    y_test_unrestricted = []
    zero_index = 0

    for i in range(len(y_test)):
        y = int(y_test[i])
        y_o = int(y_test_oligo[i])
        pred = int(y_pred_binary[i])
        pred_o = int(y_pred_binary_oligo[i])

        annotation_error = False
        if int(y_test[i]) == int(y_test_oligo[i]) and int(y_test[i]) == 1:
            intersection_count_training = intersection_count_training + 1
            annotation_error = True
        # annotation_error=False

        if y == 0:
            y_test_unrestricted.append(0)
            zero_index = y
        elif y == 1:
            if pred == pred_o and pred_o == 1:
                intersection_count = intersection_count + 1

                if annotation_error:
                    y_test_unrestricted.append(1)
                else:
                    y_test_unrestricted.append(0)
            else:
                y_test_unrestricted.append(1)

    y_test_unrestricted = np.asarray(y_test_unrestricted)
    y_test_unrestricted = y_test_unrestricted.astype(np.int)

    # Checking if all the predictions still exist
    assert len(y_test_unrestricted) == len(y_test)

    # PRECISION RECALL CURVE
    for t in [(y_test_unrestricted, 'oligo-first'), (y_test, 'intersected')]:
        y_current = t[0]
        label = t[1]
        fig_path = model_source_path_neuron + os.sep + 'intersections' + os.sep
        os.makedirs(fig_path, exist_ok=True)
        print('Writing for ' + label + ': Nuclei: ' + str(len(y_current)))

        ###########
        # PR CURVE
        ###########
        lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_current, y_pred)
        lr_auc = auc(lr_recall, lr_precision)
        lr_no_skill = len(y_current[y_current == 1]) / len(y_current)

        plt.plot([0, 1], [lr_no_skill, lr_no_skill], linestyle='--')
        plt.plot(lr_recall, lr_precision, label='PR (Area = {:.3f})'.format(lr_auc))
        plt.xlabel('Recall (TPR)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve: ' + label)
        plt.legend(loc='best')
        plt.savefig(fig_path + label + '-pr.png', dpi=img_dpi)
        plt.savefig(fig_path + label + '-pr.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + label + '-pr.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        #############
        # ROC CURVE
        #############
        print('Calculating ROC.')
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_current, y_pred)
        auc_roc = auc(fpr_roc, tpr_roc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_roc, tpr_roc, label='ROC (Area = {:.3f})'.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(fig_path + label + '-roc.png', dpi=img_dpi)
        plt.savefig(fig_path + label + '-roc.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + label + '-roc.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        ###########
        # TEX
        ###########
        print('Writing .tex')
        f = open(fig_path + label + '-roc.tex', 'w')
        f.write(misc_cnn.get_plt_as_tex(data_list_x=[fpr_roc], data_list_y=[tpr_roc], title='ROC Curve',
                                        label_y='True positive rate', label_x='False Positive Rate',
                                        plot_colors=['blue']))
        f.close()

        f = open(fig_path + label + '-pr.tex', 'w')
        f.write(misc_cnn.get_plt_as_tex(data_list_x=[lr_recall], data_list_y=[lr_precision],
                                        title='Precision Recall Curve', label_y='True positive rate',
                                        label_x='False Positive Rate',
                                        plot_titles=['PR (Area = {:.3f})'.format(lr_auc)],
                                        plot_colors=['blue'], legend_pos='south west'))
        f.close()

        # Raw PR data
        f = open(fig_path + label + "-pr_raw.csv", 'w+')
        f.write('Baseline: ' + str(lr_no_skill) + '\n')
        f.write('i;Recall;Precision;Thresholds\n')
        for i in range(len(lr_precision)):
            text_thresholds = 'NaN'
            if i < len(lr_thresholds):
                text_thresholds = str(lr_thresholds[i])
            f.write(
                str(i + 1) + ';' + str(lr_recall[i]) + ';' + str(lr_precision[i]) + ';' + text_thresholds + ';\n')
        f.close()

    print('Saved figures to: ' + fig_path)
    print('Intersection count (Annotated): ' + str(intersection_count))
    print('Intersection count (Predicted): ' + str(intersection_count_training))

    f = open(model_source_path_neuron + os.sep + 'intersections' + os.sep + 'intersections.txt', 'w')
    f.write('Saved figures to: ' + fig_path + '\n')
    f.write('Intersection count (Annotated): ' + str(intersection_count) + '\n')
    f.write('Intersection count (Predicted): ' + str(intersection_count_training) + '\n')
    f.write('Nuclei predicted: ' + str(len(y_test)) + '\n')
    f.write('Predicted 0 count: ' + str(np.count_nonzero(y_pred_binary == 0)) + '\n')
    f.write('Predicted 1 count: ' + str(np.count_nonzero(y_pred_binary == 1)) + '\n')
    f.close()


if __name__ == '__main__':
    main()

    print('Done.')
