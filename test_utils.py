# IMPORTS
#########

import sys
from datetime import datetime

from keras.models import load_model
from sklearn.metrics import *

import misc_omnisphero as misc
from misc_omnisphero import *


def test_cnn(model_path, test_data_path, normalize_enum, img_dpi, cuda_devices, include_date=True, label='cnn-test'):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # TESTING

    fig_path = model_path + os.sep + label
    if include_date:
        fig_path = fig_path + datetime.now().strftime("%Y_%m_%d")
    fig_path = fig_path + os.sep

    os.makedirs(fig_path, exist_ok=True)

    print('Loading model & weights')
    if os.path.exists(model_path + 'custom.h5'):
        model = load_model(model_path + 'custom.h5')
        model.load_weights(model_path + 'custom_weights_best.h5')
    else:
        model = load_model(model_path + 'model.h5')
        model.load_weights(model_path + 'weights_best.h5')

    y_test = np.empty((0, 1))
    X_test, y_test = misc.hdf5_loader(test_data_path, gpCurrent=1, gpMax=1, normalize_enum=normalize_enum)
    print('Done. Preprocessing test data.')
    y_test = np.asarray(y_test)
    y_test = y_test.astype(np.int)

    X_test = np.asarray(X_test)
    print(X_test.shape)
    X_test = np.moveaxis(X_test, 1, 3)
    # X_test = misc.normalize_RGB_pixels(X_test)

    print("Loaded test data has shape: ")
    print(X_test.shape)
    print(y_test.shape)

    try:
        # Preditcing Test Data
        print('Trying to predict test data')
        y_pred_roc = model.predict(X_test)  # .ravel()

        # PRECISION RECALL CURVE
        lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_test, y_pred_roc)
        lr_auc = auc(lr_recall, lr_precision)
        lr_no_skill = len(y_test[y_test == 1]) / len(y_test)

        plt.plot([0, 1], [lr_no_skill, lr_no_skill], linestyle='--')
        plt.plot(lr_recall, lr_precision, label='PR (Area = {:.3f})'.format(lr_auc))
        plt.xlabel('Recall (TPR)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.savefig(fig_path + 'pr.png', dpi=img_dpi)
        plt.savefig(fig_path + 'pr.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'pr.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        # Raw PR data
        print('Saving raw PR data')
        f = open(fig_path + "pr_data_raw.csv", 'w+')
        f.write('Baseline: ' + str(lr_no_skill) + '\n')
        f.write('i;Recall;Precision;Thresholds\n')
        for i in range(len(lr_precision)):
            f.write(
                str(i + 1) + ';' + str(lr_recall[i]) + ';' + str(lr_precision[i]) + ';' + str(lr_precision[i]) + ';\n')
        f.close()

        # ROC CURVE
        print('Calculating roc curve.')
        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_pred_roc)

        print('Calculating AUC.')
        auc_roc = auc(fpr_roc, tpr_roc)

        print('Plotting roc curve.')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_roc, tpr_roc, label='ROC (Area = {:.3f})'.format(auc_roc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(fig_path + 'roc.png', dpi=img_dpi)
        plt.savefig(fig_path + 'roc.pdf', dpi=img_dpi, transparent=True)
        plt.savefig(fig_path + 'roc.svg', dpi=img_dpi, transparent=True)
        plt.clf()

        # Raw ROC data
        print('Saving raw ROC data')
        f = open(fig_path + "roc_data_raw.csv", 'w+')
        f.write('i;FPR;TPR;Thresholds\n')
        for i in range(len(thresholds_roc)):
            f.write(
                str(i + 1) + ';' + str(fpr_roc[i]) + ';' + str(tpr_roc[i]) + ';' + str(thresholds_roc[i]) + ';\n')
        f.close()

        # HISTOGRAM

        hist_pos = y_pred_roc[np.where(y_pred_roc > 0.5)]
        plt.hist(hist_pos, bins='auto')
        plt.title("Histogram: Positive")
        plt.savefig(fig_path + 'histogram_1.png', dpi=img_dpi)
        plt.clf()

        hist_neg = y_pred_roc[np.where(y_pred_roc <= 0.5)]
        plt.hist(hist_neg, bins='auto')
        plt.title("Histogram: Negative")
        plt.savefig(fig_path + 'histogram_0.png', dpi=img_dpi)
        plt.clf()

        plt.hist(y_pred_roc, bins='auto')
        plt.title("Histogram: All")
        plt.savefig(fig_path + 'histogram_all.png', dpi=img_dpi)
        plt.clf()

        # plt.hist(y_pred_roc, bins='auto')
        # plt.title("Histogram: All [Capped]")
        # axes = plt.gca()
        # plt.ylim(0, 2000)
        # plt.xlim(0, 1)
        # plt.savefig(fig_path + 'histogram_all2.png', dpi=img_dpi)
        # plt.clf()

        # TPR / FNR
        print("Calculating TPR / TNR, etc. for: " + label + ".")

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        bp = 0
        bn = 0
        pp = 0
        pn = 0

        y_baseline_values = sigmoid_binary(y_test)
        y_prediction_values = sigmoid_binary(y_pred_roc)
        for i in range(len(y_baseline_values)):
            current_baseline = y_baseline_values[i][0]
            current_prediction = y_prediction_values[i][0]

            if current_baseline == 1:
                bp = bp + 1
                if current_prediction == 1:
                    tp = tp + 1
                    pp = pp + 1
                else:
                    fn = fn + 1
                    pn = pn + 1
            else:
                bn = bn + 1
                if current_prediction == 1:
                    fp = fp + 1
                    pp = pp + 1
                else:
                    tn = tn + 1
                    pn = pn + 1

        f = open(fig_path + "test_data_statistics.csv", 'w+')
        f.write('Count;Baseline;Predicted\n')
        f.write('All;' + str(len(y_baseline_values)) + ';' + str(len(y_prediction_values)) + '\n')
        f.write('Positive;' + str(bp) + ';' + str(pp) + '\n')
        f.write('Negative;' + str(bn) + ';' + str(pn) + '\n\n')

        f.write('TPR;' + str(tp / bp) + '\n')
        f.write('TNR;' + str(tn / bn) + '\n')
        f.write('FPR;' + str(fp / bn) + '\n')
        f.write('FNR;' + str(fn / bp) + '\n')

        f.write('ACC;' + str((tp + tn) / (bp + bn)) + '\n')
        f.write('BACC;' + str(((tp / bp) + (tn / bn)) / 2) + '\n')
        f.write('F1;' + str((2 * tp) / (2 * tp + fp + fn)) + '\n')

        f.close()

    except Exception as e:
        # Printing the exception message to file.
        print("Failed to calculate roc curve for: " + label + ".")
        f = open(fig_path + "rocError.txt", 'w+')
        f.write(str(e))

        try:
            # Printing the stack trace to the file
            exc_info = sys.exc_info()
            f.write('\n')
            f.write(str(exc_info))
        except Exception as e2:
            print('Failed to write the whole stack trace into the error file. Reason:')
            print(str(e2))
            pass

        f.close()

    print('Testing done.')
