from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import scipy.io as scio
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import random
#from classifiers import resnet
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.io import savemat
import h5py
import pylab
import gc

time_start = time.time()

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

classifier_name = 'inception_v9_v2'
data_train_date = '20251109'
data_test_date_arr = ['20240609', '20240611', '20240617', '20251109']
rx_train_arr_total = np.array([0, 1, 2, 3, 4])
rx_test_arr_vld_arr = np.array([0, 1, 2, 3, 4])
seed_num = 5
nb_epochs = 50
data_name1 = 'wifi_feature_1_normal'
data_name2 = 'wifi_feature_1_normal'

class_acc_diff_scenario_arr = np.zeros((len(data_test_date_arr)*len(rx_test_arr_vld_arr), len(rx_train_arr_total)))
rx_test_idx = 0

for data_test_date in data_test_date_arr:
    for rx_test_arr_vld in rx_test_arr_vld_arr:
        rx_train_idx = 0
        for rx_train_arr in rx_train_arr_total:
            class_acc_same_rx_seed = np.zeros(seed_num)
            class_acc_diff_rx_seed = np.zeros(seed_num)
            for exp_seed in range(seed_num):
                mat_v7_3 = False

                mul_loc = False
                dev_partial = False

                soft_label = True
                soft_factor = 0.1
                batch_size = 64
                learning_rate = 0.001
                l2 = 0.1
                depth = 6
                verbose = 0

                # According to input data
                channel_num = 1
                ori_len = 52 * channel_num
                complex_flag = False

                local_vld = True

                if mul_loc == True:
                    rx_train_arr = np.array([0, 1, 2, 3])
                    rx_test_arr_vld = np.array([1])
                if local_vld == True:
                    depth = 6

                    train_ratio = 0.6
                    test_ratio = 0.2
                    vld_ratio = 0.2
                    train_ratio_vld = 0.1
                    test_ratio_vld  = 0.9

                    data_train = np.zeros([ori_len, 320000], dtype=float)
                    data_test  = np.zeros([ori_len, 40000], dtype=float)
                    data_vld   = np.zeros([ori_len, 40000], dtype=float)
                    data_train_vld = np.zeros([ori_len, 40000], dtype=float)
                    data_test_vld  = np.zeros([ori_len, 80000], dtype=float)

                    if complex_flag == True:
                        data_train = complex(data_train)
                        data_test = complex(data_test)
                        data_vld = complex(data_vld)
                        data_train_vld = complex(data_train_vld)
                        data_test_vld = complex(data_test_vld)

                    root_dir = r'*Results output folder*'
                    data_train_dir = r'*Data folder*' + '\\' + data_train_date
                    data_test_dir = r'*Data folder*' + '\\' + data_test_date

                dev_train_arr = np.array([1, 6, 7, 8, 9, 12, 13, 14])

                dev_test_arr_vld = dev_train_arr

                labels = ["dev1", "dev6", "dev7", "dev8", "dev9", "dev12", "dev13", "dev14"]

                loc_arr = np.arange(0, 5)

                len_train_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_test_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_vld_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

                begin_flag = 1

                sample_train_num = 16000
                sample_vld_num   = 6000
                sample_train_p = 0
                sample_test_p  = 0
                sample_vld_p = 0
                for dev in dev_train_arr:
                    for rx in [rx_train_arr]:
                        if mat_v7_3 == False:
                            data_dict = scio.loadmat(data_train_dir + '\\rx' + str(rx+1) + '\\dev' + str(dev) + \
                                                 '\\' + data_name1 + '.mat')
                            data = data_dict[data_name1]
                        else:
                            a = 1

                        index = [i for i in range(data.shape[1])]
                        random.seed(exp_seed)
                        random.shuffle(index)
                        data = data[:, index]
                        data_train_par = data[:, 0:round(data.shape[1] * train_ratio)]
                        data_vld_par = data[:, round(data.shape[1] * train_ratio):round(data.shape[1] * train_ratio) + round(
                            data.shape[1] * vld_ratio)]
                        data_test_par = data[:, round(data.shape[1] * train_ratio + data.shape[1] * vld_ratio): round(
                            data.shape[1] * train_ratio + data.shape[1] * vld_ratio + data.shape[1] * test_ratio)]

                        len_train_arr[rx][dev] = data_train_par.shape[1]
                        len_test_arr[rx][dev] = data_test_par.shape[1]
                        len_vld_arr[rx][dev] = data_vld_par.shape[1]

                        data_train[:, sample_train_p:sample_train_p + data_train_par.shape[1]] = data_train_par
                        data_test[:, sample_test_p :sample_test_p + data_test_par.shape[1]] = data_test_par
                        data_vld[:, sample_vld_p : sample_vld_p + data_vld_par.shape[1]] = data_vld_par
                        sample_train_p = sample_train_p + data_train_par.shape[1]
                        sample_test_p  = sample_test_p + data_test_par.shape[1]
                        sample_vld_p = sample_vld_p + data_vld_par.shape[1]

                len_train_arr = len_train_arr[:, :]
                len_test_arr  = len_test_arr[:, :]
                len_vld_arr = len_vld_arr[:, :]
                data_train = data_train[:, 0:sample_train_p]
                data_test  = data_test[:, 0:sample_test_p]
                data_vld = data_vld[:, 0:sample_vld_p]

                time_end = time.time()
                print('time cost', (time_end - time_start)/60, 'min')

                len_train_arr_vld = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_test_arr_vld  = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

                sample_train_vld_p = 0
                sample_test_vld_p  = 0

                begin_flag = 1
                for dev in dev_test_arr_vld:
                    for rx in [rx_test_arr_vld]:
                        if mat_v7_3 == False:
                            data_dict = scio.loadmat(data_test_dir + '\\rx' + str(rx+1) +'\\dev' + str(dev) + \
                                                     '\\' + data_name2 + '.mat')
                            data = data_dict[data_name2]
                        else:
                            a = 1

                        index = [i for i in range(data.shape[1])]
                        random.seed(exp_seed)
                        random.shuffle(index)
                        data = data[:,index]
                        data_train_par_vld = data[:, 0:round(data.shape[1] * train_ratio_vld)]
                        data_test_par_vld  = data[:,round(data.shape[1] * train_ratio_vld):round(data.shape[1] * train_ratio_vld) + round(data.shape[1] * test_ratio_vld)]

                        len_train_arr_vld[rx][dev] = data_train_par_vld.shape[1]
                        len_test_arr_vld[rx][dev] = data_test_par_vld.shape[1]
                        data_train_vld[:, sample_train_vld_p:sample_train_vld_p + data_train_par_vld.shape[1]] = data_train_par_vld
                        data_test_vld[:, sample_test_vld_p:sample_test_vld_p + data_test_par_vld.shape[1]] = data_test_par_vld
                        sample_train_vld_p = sample_train_vld_p + data_train_par_vld.shape[1]
                        sample_test_vld_p  = sample_test_vld_p + data_test_par_vld.shape[1]

                len_train_arr_vld = len_train_arr_vld[:, :]
                len_test_arr_vld = len_test_arr_vld[:, :]
                data_train_vld = data_train_vld[:, 0:sample_train_vld_p]
                data_test_vld = data_test_vld[:, 0:sample_test_vld_p]

                data_train = data_train.T
                data_vld = data_vld.T
                data_test  = data_test.T
                data_train_vld = data_train_vld.T
                data_test_vld  = data_test_vld.T
                del data, data_train_par, data_vld_par, data_test_par, data_train_par_vld, data_test_par_vld
                gc.collect()

                if complex_flag == False:
                    framelen = int(data_train.shape[1]/channel_num)
                else:
                    framelen = int(data_train.shape[1])

                len_train = np.sum(len_train_arr, axis=0)
                len_vld = np.sum(len_vld_arr, axis=0)
                len_test  = np.sum(len_test_arr, axis=0)
                len_train_vld = np.sum(len_train_arr_vld, axis=0)
                len_test_vld = np.sum(len_test_arr_vld, axis=0)

                len_train = len_train[len_train != 0]
                len_vld = len_vld[len_vld != 0]
                len_test = len_test[len_test != 0]
                len_train_vld = len_train_vld[len_train_vld != 0]
                len_test_vld = len_test_vld[len_test_vld != 0]

                label = np.arange(0, len(dev_train_arr))
                len_train = len_train.astype(int)
                len_vld = len_vld.astype(int)
                len_test  = len_test.astype(int)
                len_train_vld = len_train_vld.astype(int)
                len_test_vld  = len_test_vld.astype(int)

                y_train = np.repeat(label, len_train)
                y_train.shape = (len(y_train),1)
                y_vld = np.repeat(label, len_vld)
                y_vld.shape = (len(y_vld), 1)
                y_test = np.repeat(label, len_test)
                y_test.shape = (len(y_test), 1)
                y_train_vld = np.repeat(label, len_train_vld)
                y_train_vld.shape = (len(y_train_vld),1)
                y_test_vld = np.repeat(label, len_test_vld)
                y_test_vld.shape = (len(y_test_vld), 1)

                if len(data_train.shape) == 2:  # if univariate
                    # add a dimension to make it multivariate with one dimension
                    data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], 1))
                    data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], 1))
                    data_vld = data_vld.reshape((data_vld.shape[0], data_vld.shape[1], 1))
                    data_train_vld = data_train_vld.reshape((data_train_vld.shape[0], data_train_vld.shape[1], 1))
                    data_test_vld = data_test_vld.reshape((data_test_vld.shape[0], data_test_vld.shape[1], 1))

                output_directory = root_dir + '\\results\\' + classifier_name + '\\'

                test_dir_df_metrics = output_directory + 'df_metrics.csv'

                create_directory(output_directory)
                if classifier_name == 'inception_v9_v2':
                    from classifiers import inception_v9_v2

                input_shape = data_train.shape[1:]
                nb_classes = len(dev_train_arr)
                # verbose = False
                # transform the labels from integers to one hot vectors
                enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                y_train_arr = y_train
                y_vld_arr = y_vld
                y_test_arr = y_test
                y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                y_vld = enc.transform(y_vld.reshape(-1, 1)).toarray()
                y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

                if soft_label == True:
                    y_train = label_smoothing(y_train, soft_factor)

                enc.fit(np.concatenate((y_train_vld, y_test_vld), axis=0).reshape(-1, 1))
                y_train_vld_arr = y_train_vld
                y_test_vld_arr  = y_test_vld
                y_train_vld = enc.transform(y_train_vld.reshape(-1, 1)).toarray()
                y_test_vld = enc.transform(y_test_vld.reshape(-1, 1)).toarray()

                # save orignal y because later we will use binary
                y_vld_true = np.argmax(y_vld, axis=1)
                y_test_true = np.argmax(y_test, axis=1)
                y_train_vld_true = np.argmax(y_train_vld, axis=1)
                y_test_vld_true = np.argmax(y_test_vld, axis=1)

                model = inception_v9_v2.Classifier_INCEPTION_V9(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=nb_epochs, batch_size = batch_size, depth=depth, lr = learning_rate, l2=l2)

                gc.collect()
                print('#############Start the training process#############')
                history = model.fit(data_train, y_train, data_vld, y_vld, y_vld_true)

                if data_test_date == data_train_date:
                    y_predict_test_acc = model.predict(data_test, y_test_true, data_train, y_train, y_test)
                    print(f'Ours, Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Test Scenario = {data_test_date}, Seed = {exp_seed + 1}')
                    print('Diff Rx, Acc = %.4f' % y_predict_test_acc.accuracy)

                    class_acc_diff_rx_seed[exp_seed] = y_predict_test_acc.accuracy
                else:
                    y_predict_test_vld_acc = model.predict(data_test_vld, y_test_vld_true, data_train, y_train,
                                                           y_test_vld)
                    print(f'Ours, Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Test Scenario = {data_test_date}, Seed = {exp_seed + 1}')
                    print('Diff Rx, Acc = %.4f' % y_predict_test_vld_acc.accuracy)

                    class_acc_diff_rx_seed[exp_seed] = y_predict_test_vld_acc.accuracy

            class_acc_diff_scenario_arr[rx_test_idx, rx_train_idx] = class_acc_diff_rx_seed.mean()
            mdict = {'acc_251109_HL_diff_scenario_Rx_train_more_arr': class_acc_diff_scenario_arr}
            savemat('acc_251109_HL_diff_scenario_Rx_train_more_arr.mat', mdict)
            rx_train_idx = rx_train_idx + 1

        rx_test_idx = rx_test_idx + 1

mdict = {'acc_251109_HL_diff_scenario_Rx_train_more_arr': class_acc_diff_scenario_arr}
savemat('acc_251109_HL_diff_scenario_Rx_train_more_arr.mat', mdict)


##########################################################################
from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import scipy.io as scio
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import random
#from classifiers import resnet
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.io import savemat
import h5py
import pylab
import gc

time_start = time.time()

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

classifier_name = 'inception_v9_v2'
data_train_date = '20251109'
data_test_date_arr = ['20240609', '20240611', '20240617', '251109']
rx_train_arr_total = np.array([0, 1, 2, 3, 4])
rx_test_arr_vld_arr = np.array([0, 1, 2, 3, 4])
seed_num = 5
nb_epochs = 50
data_name1 = 'chen_feature_normal'
data_name2 = 'chen_feature_normal'

class_acc_diff_scenario_arr = np.zeros((len(data_test_date_arr)*len(rx_test_arr_vld_arr), len(rx_train_arr_total)))
rx_test_idx = 0

for data_test_date in data_test_date_arr:
    for rx_test_arr_vld in rx_test_arr_vld_arr:
        rx_train_idx = 0
        for rx_train_arr in rx_train_arr_total:
            class_acc_same_rx_seed = np.zeros(seed_num)
            class_acc_diff_rx_seed = np.zeros(seed_num)
            for exp_seed in range(seed_num):
                mat_v7_3 = False

                mul_loc = False
                dev_partial = False

                soft_label = True
                soft_factor = 0.1
                batch_size = 64
                learning_rate = 0.001
                l2 = 0.1
                depth = 6
                verbose = 0

                # According to input data
                channel_num = 1
                ori_len = 12 * channel_num
                complex_flag = False

                # For local vld ##################################################
                local_vld = True
                # local_vld = False

                if mul_loc == True:
                    rx_train_arr = np.array([0, 1, 2, 3])
                    rx_test_arr_vld = np.array([1])
                if local_vld == True:
                    depth = 6

                    train_ratio = 0.6
                    test_ratio = 0.2
                    vld_ratio = 0.2
                    train_ratio_vld = 0.1
                    test_ratio_vld  = 0.9

                    data_train = np.zeros([ori_len, 320000], dtype=float)
                    data_test  = np.zeros([ori_len, 40000], dtype=float)
                    data_vld   = np.zeros([ori_len, 40000], dtype=float)
                    data_train_vld = np.zeros([ori_len, 40000], dtype=float)
                    data_test_vld  = np.zeros([ori_len, 80000], dtype=float)

                    if complex_flag == True:
                        data_train = complex(data_train)
                        data_test = complex(data_test)
                        data_vld = complex(data_vld)
                        data_train_vld = complex(data_train_vld)
                        data_test_vld = complex(data_test_vld)

                    root_dir = r'*Results output folder*'
                    data_train_dir = r'*Data folder*' + '\\' + data_train_date
                    data_test_dir = r'*Data folder*' + '\\' + data_test_date

                dev_train_arr = np.array([1, 6, 7, 8, 9, 12, 13, 14])

                dev_test_arr_vld = dev_train_arr

                labels = ["dev1", "dev6", "dev7", "dev8", "dev9", "dev12", "dev13", "dev14"]

                loc_arr = np.arange(0, 5)

                len_train_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_test_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_vld_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

                begin_flag = 1

                sample_train_num = 16000
                sample_vld_num   = 6000
                sample_train_p = 0
                sample_test_p  = 0
                sample_vld_p = 0
                for dev in dev_train_arr:
                    for rx in [rx_train_arr]:
                        if mat_v7_3 == False:
                            data_dict = scio.loadmat(data_train_dir + '\\rx' + str(rx+1) + '\\dev' + str(dev) + \
                                                 '\\' + data_name1 + '.mat')
                            data = data_dict[data_name1]
                        else:
                            a = 1

                        index = [i for i in range(data.shape[1])]
                        random.seed(exp_seed)
                        random.shuffle(index)
                        data = data[:, index]
                        data_train_par = data[:, 0:round(data.shape[1] * train_ratio)]
                        data_vld_par = data[:, round(data.shape[1] * train_ratio):round(data.shape[1] * train_ratio) + round(
                            data.shape[1] * vld_ratio)]
                        data_test_par = data[:, round(data.shape[1] * train_ratio + data.shape[1] * vld_ratio): round(
                            data.shape[1] * train_ratio + data.shape[1] * vld_ratio + data.shape[1] * test_ratio)]

                        len_train_arr[rx][dev] = data_train_par.shape[1]
                        len_test_arr[rx][dev] = data_test_par.shape[1]
                        len_vld_arr[rx][dev] = data_vld_par.shape[1]

                        data_train[:, sample_train_p:sample_train_p + data_train_par.shape[1]] = data_train_par
                        data_test[:, sample_test_p :sample_test_p + data_test_par.shape[1]] = data_test_par
                        data_vld[:, sample_vld_p : sample_vld_p + data_vld_par.shape[1]] = data_vld_par
                        sample_train_p = sample_train_p + data_train_par.shape[1]
                        sample_test_p  = sample_test_p + data_test_par.shape[1]
                        sample_vld_p = sample_vld_p + data_vld_par.shape[1]

                len_train_arr = len_train_arr[:, :]
                len_test_arr  = len_test_arr[:, :]
                len_vld_arr = len_vld_arr[:, :]
                data_train = data_train[:, 0:sample_train_p]
                data_test  = data_test[:, 0:sample_test_p]
                data_vld = data_vld[:, 0:sample_vld_p]

                time_end = time.time()
                print('time cost', (time_end - time_start)/60, 'min')

                len_train_arr_vld = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
                len_test_arr_vld  = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

                sample_train_vld_p = 0
                sample_test_vld_p  = 0

                begin_flag = 1
                for dev in dev_test_arr_vld:
                    for rx in [rx_test_arr_vld]:
                        if mat_v7_3 == False:
                            data_dict = scio.loadmat(data_test_dir + '\\rx' + str(rx+1) +'\\dev' + str(dev) + \
                                                     '\\' + data_name2 + '.mat')
                            data = data_dict[data_name2]
                        else:
                            a = 1

                        index = [i for i in range(data.shape[1])]
                        random.seed(exp_seed)
                        random.shuffle(index)
                        data = data[:,index]
                        data_train_par_vld = data[:, 0:round(data.shape[1] * train_ratio_vld)]
                        data_test_par_vld  = data[:,round(data.shape[1] * train_ratio_vld):round(data.shape[1] * train_ratio_vld) + round(data.shape[1] * test_ratio_vld)]

                        len_train_arr_vld[rx][dev] = data_train_par_vld.shape[1]
                        len_test_arr_vld[rx][dev] = data_test_par_vld.shape[1]
                        data_train_vld[:, sample_train_vld_p:sample_train_vld_p + data_train_par_vld.shape[1]] = data_train_par_vld
                        data_test_vld[:, sample_test_vld_p:sample_test_vld_p + data_test_par_vld.shape[1]] = data_test_par_vld
                        sample_train_vld_p = sample_train_vld_p + data_train_par_vld.shape[1]
                        sample_test_vld_p  = sample_test_vld_p + data_test_par_vld.shape[1]

                len_train_arr_vld = len_train_arr_vld[:, :]
                len_test_arr_vld = len_test_arr_vld[:, :]
                data_train_vld = data_train_vld[:, 0:sample_train_vld_p]
                data_test_vld = data_test_vld[:, 0:sample_test_vld_p]

                data_train = data_train.T
                data_vld = data_vld.T
                data_test  = data_test.T
                data_train_vld = data_train_vld.T
                data_test_vld  = data_test_vld.T
                del data, data_train_par, data_vld_par, data_test_par, data_train_par_vld, data_test_par_vld
                gc.collect()

                if complex_flag == False:
                    framelen = int(data_train.shape[1]/channel_num)
                else:
                    framelen = int(data_train.shape[1])

                len_train = np.sum(len_train_arr, axis=0)
                len_vld = np.sum(len_vld_arr, axis=0)
                len_test  = np.sum(len_test_arr, axis=0)
                len_train_vld = np.sum(len_train_arr_vld, axis=0)
                len_test_vld = np.sum(len_test_arr_vld, axis=0)

                len_train = len_train[len_train != 0]
                len_vld = len_vld[len_vld != 0]
                len_test = len_test[len_test != 0]
                len_train_vld = len_train_vld[len_train_vld != 0]
                len_test_vld = len_test_vld[len_test_vld != 0]

                label = np.arange(0, len(dev_train_arr))
                len_train = len_train.astype(int)
                len_vld = len_vld.astype(int)
                len_test  = len_test.astype(int)
                len_train_vld = len_train_vld.astype(int)
                len_test_vld  = len_test_vld.astype(int)

                y_train = np.repeat(label, len_train)
                y_train.shape = (len(y_train),1)
                y_vld = np.repeat(label, len_vld)
                y_vld.shape = (len(y_vld), 1)
                y_test = np.repeat(label, len_test)
                y_test.shape = (len(y_test), 1)
                y_train_vld = np.repeat(label, len_train_vld)
                y_train_vld.shape = (len(y_train_vld),1)
                y_test_vld = np.repeat(label, len_test_vld)
                y_test_vld.shape = (len(y_test_vld), 1)

                if len(data_train.shape) == 2:  # if univariate
                    # add a dimension to make it multivariate with one dimension
                    data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], 1))
                    data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], 1))
                    data_vld = data_vld.reshape((data_vld.shape[0], data_vld.shape[1], 1))
                    data_train_vld = data_train_vld.reshape((data_train_vld.shape[0], data_train_vld.shape[1], 1))
                    data_test_vld = data_test_vld.reshape((data_test_vld.shape[0], data_test_vld.shape[1], 1))

                output_directory = root_dir + '\\results\\' + classifier_name + '\\'

                test_dir_df_metrics = output_directory + 'df_metrics.csv'

                create_directory(output_directory)
                if classifier_name == 'inception_v9_v2':
                    from classifiers import inception_v9_v2

                input_shape = data_train.shape[1:]
                nb_classes = len(dev_train_arr)
                # verbose = False
                # transform the labels from integers to one hot vectors
                enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                y_train_arr = y_train
                y_vld_arr = y_vld
                y_test_arr = y_test
                y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                y_vld = enc.transform(y_vld.reshape(-1, 1)).toarray()
                y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

                if soft_label == True:
                    y_train = label_smoothing(y_train, soft_factor)

                enc.fit(np.concatenate((y_train_vld, y_test_vld), axis=0).reshape(-1, 1))
                y_train_vld_arr = y_train_vld
                y_test_vld_arr  = y_test_vld
                y_train_vld = enc.transform(y_train_vld.reshape(-1, 1)).toarray()
                y_test_vld = enc.transform(y_test_vld.reshape(-1, 1)).toarray()

                # save orignal y because later we will use binary
                y_vld_true = np.argmax(y_vld, axis=1)
                y_test_true = np.argmax(y_test, axis=1)
                y_train_vld_true = np.argmax(y_train_vld, axis=1)
                y_test_vld_true = np.argmax(y_test_vld, axis=1)

                model = inception_v9_v2.Classifier_INCEPTION_V9(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=nb_epochs, batch_size = batch_size, depth=depth, lr = learning_rate, l2=l2)

                gc.collect()
                print('#############Start the training process#############')
                history = model.fit(data_train, y_train, data_vld, y_vld, y_vld_true)

                if data_test_date == data_train_date:
                    y_predict_test_acc = model.predict(data_test, y_test_true, data_train, y_train, y_test)
                    print(f'Chen, Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Test Scenario = {data_test_date}, Seed = {exp_seed + 1}')
                    print('Diff Rx, Acc = %.4f' % y_predict_test_acc.accuracy)

                    class_acc_diff_rx_seed[exp_seed] = y_predict_test_acc.accuracy
                else:
                    y_predict_test_vld_acc = model.predict(data_test_vld, y_test_vld_true, data_train, y_train, y_test_vld)
                    print(f'Chen, Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Test Scenario = {data_test_date}, Seed = {exp_seed + 1}')
                    print('Diff Rx, Acc = %.4f' % y_predict_test_vld_acc.accuracy)

                    class_acc_diff_rx_seed[exp_seed] = y_predict_test_vld_acc.accuracy

            class_acc_diff_scenario_arr[rx_test_idx, rx_train_idx] = class_acc_diff_rx_seed.mean()
            mdict = {'acc_251109_Chen_diff_scenario_Rx_train_more_arr': class_acc_diff_scenario_arr}
            savemat('acc_251109_Chen_diff_scenario_Rx_train_more_arr.mat', mdict)
            rx_train_idx = rx_train_idx + 1

        rx_test_idx = rx_test_idx + 1

mdict = {'acc_251109_Chen_diff_scenario_Rx_train_more_arr': class_acc_diff_scenario_arr}
savemat('acc_251109_Chen_diff_scenario_Rx_train_more_arr.mat', mdict)