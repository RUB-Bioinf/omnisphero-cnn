# IMPORTS
#########
import sys
from datetime import datetime

from keras.models import load_model
from sklearn.metrics import *

import misc_cnn
import misc_omnisphero as misc
from misc_omnisphero import *

# PATHS & ARGS
cuda_devices = "0"

# OLD DATA
# model_path = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/results/oligo_final_sigmodal/0_custom/'
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/oligo/EKB25_trainingData_oligo/'
# test_data_path = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/wholeWell/neuron/EKB25_trainingData_neuron/'

# KONTROLLIERT TEST DATA
# model_path = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/models/debug-kontrolliert-weighted/neuron-n4-ep1500/0_custom/'
test_data_path_oligo = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test/'
test_data_path_neuron = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test/'

test_data_path_oligo_filter = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test_filters/'
test_data_path_neuron_filter = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_filters/'

test_data_path_oligo_filter_erneut = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_erneut_kontrolliert_test_filters/'
test_data_path_neuron_filter_erneut = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_erneut_kontrolliert_test_filters/'

test_data_path_oligo_erneut = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_erneut_kontrolliert_test/'
test_data_path_neuron_erneut = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_erneut_kontrolliert_test/'

test_data_path_oligo_withoutKB25 = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test_woBK25/'
test_data_path_neuron_withoutKB25 = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_woBK25/'

test_data_path_oligo_debug = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/oligo_kontrolliert_test_debug/'
test_data_path_neuron_debug = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_debug/'

# Models in use:
model_path_paper_neuron = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/neuron-normalize4/'
model_path_paper_oligo = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize4/'

normalize_enum = 4
img_dpi_default = 450
label = 'cnn-test'
cuda_devices_default = "0"

def test_cnn(model_path: str, test_data_path: str, normalize_enum: int, img_dpi: int=img_dpi_default, cuda_devices: str=cuda_devices_default,
             include_date: bool = True, label: str = 'cnn-test',n_jobs:int = 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(' ### Testing CNN! ###')
    print('Model path: '+model_path)
    print('Test Data path: '+test_data_path)

    # TESTING
    fig_path = model_path + os.sep + label
    if include_date:
        fig_path = fig_path + '-' + datetime.now().strftime("%Y_%m_%d")
    fig_path = fig_path + os.sep

    os.makedirs(fig_path, exist_ok=True)

    print('Loading model & weights')
    print('Model path: ' + model_path)
    if os.path.exists(model_path + 'custom.h5'):
        model = load_model(model_path + 'custom.h5')
        model.load_weights(model_path + 'custom_weights_best.h5')
    else:
        model = load_model(model_path + 'model.h5')
        model.load_weights(model_path + 'weights_best.h5')
    print('Finished loading model.')

    print('Loading test data: ' + test_data_path)
    y_test = np.empty((0, 1))
    X_test, y_test, test_loading_errors,_ = misc.hdf5_loader(test_data_path, gp_current=1, gp_max=1, normalize_enum=normalize_enum, n_jobs=n_jobs, force_verbose = True)
    print('Finished loading test data.')

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

    # Printing test data:
    test_out_file = fig_path + 'test_data.txt'
    try:
        f = open(test_out_file, 'w')
        f.write('Data Source Path: '+test_data_path+'\n\n')

        f.write('X_test shape: ' + str(X_test.shape) + '\n')
        f.write('y_test shape: ' + str(y_test.shape) + '\n')
        f.write('Read class 0 count: ' + str(np.count_nonzero(y_test == 0)) + '\n')
        f.write('Read class 1 count: ' + str(np.count_nonzero(y_test == 1)) + '\n')
        f.close()
    except Exception as e:
        pass

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

        f = open(fig_path + 'pr.tex', 'w')
        f.write(misc_cnn.get_plt_as_tex(data_list_x=[lr_recall], data_list_y=[lr_precision],
                                        title='Precision Recall Curve', label_y='True positive rate',
                                        label_x='False Positive Rate',
                                        plot_titles=['PR (Area = {:.3f})'.format(lr_auc)],
                                        plot_colors=['blue'], legend_pos='south west'))
        f.close()

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

        # ROC stuff info:
        # Source: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

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

        f = open(fig_path + 'roc.tex', 'w')
        f.write(misc_cnn.get_plt_as_tex(data_list_x=[fpr_roc], data_list_y=[tpr_roc], title='ROC Curve',
                                        label_y='True positive rate', label_x='False Positive Rate',
                                        plot_colors=['blue']))
        f.close()

        # Raw ROC data
        print('Saving raw ROC data: ' + fig_path)
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

        try:
            f.write(str(e)+'\n')

            # Printing the stack trace to the file
            exc_info = sys.exc_info()
            f.write('\n')
            f.write(str(exc_info))
        except Exception as e2:
            print('Failed to write the whole stack trace into the error file. Reason:')
            print(str(e2))
            pass

        f.close()


def main():
    oligo_mode = False
    neuron_mode = False

    debug_mode = True
    paper_mode = True

    o1 = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/oligo-old/'
    n1 = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/neuron/'
    db = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-debug-smote/oligo-debug/'

    # Paper Individuum Test
    # '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_val_PaperIndividuum1/'

    pi1 = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_PaperIndividuum1/'
    pi2 = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_PaperIndividuum2/'
    pi3 = '/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/neuron_kontrolliert_test_PaperIndividuum3/'
    paper_individuum_model = '/prodi/bioinf/bioinfdata/work/Omnisphero/CNN/training/debug/paper-final_datagen/neuron_kontrolliert_PaperIndividuum2/'

    print("Running CNN test.")

    if paper_mode:
        test_cnn('/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize0/', test_data_path_oligo_filter_erneut, normalize_enum=0, cuda_devices="0", label='cnn-normalize-redo',n_jobs=22)
        test_cnn('/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize1/', test_data_path_oligo_filter_erneut, normalize_enum=1, cuda_devices="0", label='cnn-normalize-redo',n_jobs=22)
        test_cnn('/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize2/', test_data_path_oligo_filter_erneut, normalize_enum=2, cuda_devices="0", label='cnn-normalize-redo',n_jobs=22)
        test_cnn('/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize3/', test_data_path_oligo_filter_erneut, normalize_enum=3, cuda_devices="0", label='cnn-normalize-redo',n_jobs=22)
        test_cnn('/prodi/bioinf/bioinfdata/work/omnisphero/CNN/training/debug/paper-final_datagen/oligo-normalize4/', test_data_path_oligo_filter_erneut, normalize_enum=4, cuda_devices="0", label='cnn-normalize-redo',n_jobs=22)

        return

    if oligo_mode:
        # test_cnn(o1, test_data_path_oligo, normalize_enum, img_dpi, cuda_devices, True, label='cnn-test')
        test_cnn(o1, test_data_path_oligo, normalize_enum, cuda_devices="0", label='cnn-debug-test')
    if neuron_mode:
        test_cnn(n1, test_data_path_neuron, normalize_enum, cuda_devices="0", label='cnn-debug-test')
    if debug_mode:
        test_cnn(paper_individuum_model, pi1, n_jobs=15, normalize_enum=4, cuda_devices="3", label='cnn-individuum1-test')
        test_cnn(paper_individuum_model, pi2, n_jobs=15, normalize_enum=4, cuda_devices="3", label='cnn-individuum2-test')
        test_cnn(paper_individuum_model, pi3, n_jobs=15, normalize_enum=4, cuda_devices="3", label='cnn-individuum3-test')

    print('Testing done.')


if __name__ == "__main__":
    main()
