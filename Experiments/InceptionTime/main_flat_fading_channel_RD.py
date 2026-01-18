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
from utils.utils_yx import calculate_metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# matplotlib.use('TkAgg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import savemat
import h5py
# import joblib
import pylab
import gc

time_start=time.time()

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

rx_train_arr_total = np.array([0, 1, 2, 3, 4])
test_arr = np.array([2])
nb_epochs = 50
data_name1 = 'my_lstf_feature_normal'
data_name2 = 'my_lltf_feature_normal'
for rx_test_arr_vld in test_arr:
    for rx_train_arr in rx_train_arr_total:

        for exp_seed in range(5):
            classifier_name = 'inception_v9_v2'
            data_train_date = '20240517'
            data_test_date  = '20240517'

            mat_v7_3 = False

            # For Training
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
            if data_name1 == 'chen_feature_normal':
                ori_len1 = 12 * channel_num
            elif data_name1 == 'flat_combine_feature':
                ori_len1 = 64 * channel_num
            elif data_name1 == 'my_lstf_feature_normal':
                ori_len1 = 12 * channel_num
            else:
                raise RuntimeError('testError')

            if data_name2 == 'my_lltf_feature_normal':
                ori_len2 = 52 * channel_num
            else:
                raise  RuntimeError('testError')
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

                data_train1 = np.zeros([ori_len1, 320000], dtype=float)
                data_test1  = np.zeros([ori_len1, 40000], dtype=float)
                data_vld1   = np.zeros([ori_len1, 40000], dtype=float)
                data_train_vld1 = np.zeros([ori_len1, 400000], dtype=float)
                data_test_vld1  = np.zeros([ori_len1, 400000], dtype=float)

                data_train2 = np.zeros([ori_len2, 320000], dtype=float)
                data_test2  = np.zeros([ori_len2, 40000], dtype=float)
                data_vld2   = np.zeros([ori_len2, 40000], dtype=float)
                data_train_vld2 = np.zeros([ori_len2, 400000], dtype=float)
                data_test_vld2  = np.zeros([ori_len2, 400000], dtype=float)

                if complex_flag == True:
                    data_train = complex(data_train)
                    data_test = complex(data_test)
                    data_vld = complex(data_vld)
                    data_train_vld = complex(data_train_vld)
                    data_test_vld = complex(data_test_vld)

                root_dir = r'*Results output folder*'
                data_train_dir = r'*Data folder*' + '\\' + data_train_date
                data_test_dir = r'*Data folder*' + '\\' + data_test_date

            dev_train_arr = np.array([4, 5, 6, 7, 8, 10, 11, 12, 13, 14])

            dev_test_arr_vld = dev_train_arr

            labels = ["dev4", "dev5", "dev6", "dev7", "dev8", "dev10", "dev11", "dev12", "dev13", "dev14"]

            loc_arr1 = np.arange(0, 5)
            loc_arr2 = loc_arr1

            len_train_arr1 = np.zeros([len(loc_arr1), dev_train_arr[-1]+1])
            len_test_arr1 = np.zeros([len(loc_arr1), dev_train_arr[-1]+1])
            len_vld_arr1 = np.zeros([len(loc_arr1), dev_train_arr[-1]+1])
            len_train_arr2 = len_train_arr1
            len_test_arr2 = len_test_arr1
            len_vld_arr2 = len_vld_arr1

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
                        data1 = data_dict[data_name1]
                        data_dict = scio.loadmat(data_train_dir + '\\rx' + str(rx + 1) + '\\dev' + str(dev) + \
                                                 '\\' + data_name2 + '.mat')
                        data2 = data_dict[data_name2]

                    index = [i for i in range(data1.shape[1])]
                    random.seed(exp_seed)
                    random.shuffle(index)
                    data1 = data1[:, index]
                    data2 = data2[:, index]
                    data_train_par1 = data1[:, 0:round(data1.shape[1] * train_ratio)]
                    data_vld_par1   = data1[:, round(data1.shape[1] * train_ratio):round(data1.shape[1] * train_ratio) + round(
                        data1.shape[1] * vld_ratio)]
                    data_test_par1 = data1[:, round(data1.shape[1] * train_ratio + data1.shape[1] * vld_ratio): round(
                        data1.shape[1] * train_ratio + data1.shape[1] * vld_ratio + data1.shape[1] * test_ratio)]

                    data_train_par2 = data2[:, 0:round(data2.shape[1] * train_ratio)]
                    data_vld_par2 = data2[:,
                                    round(data2.shape[1] * train_ratio):round(data2.shape[1] * train_ratio) + round(
                                        data2.shape[1] * vld_ratio)]
                    data_test_par2 = data2[:, round(data2.shape[1] * train_ratio + data2.shape[1] * vld_ratio): round(
                        data2.shape[1] * train_ratio + data2.shape[1] * vld_ratio + data2.shape[1] * test_ratio)]

                    len_train_arr1[rx][dev] = data_train_par1.shape[1]
                    len_test_arr1[rx][dev] = data_test_par1.shape[1]
                    len_vld_arr1[rx][dev] = data_vld_par1.shape[1]

                    len_train_arr2[rx][dev] = data_train_par2.shape[1]
                    len_test_arr2[rx][dev] = data_test_par2.shape[1]
                    len_vld_arr2[rx][dev] = data_vld_par2.shape[1]

                    data_train1[:, sample_train_p:sample_train_p + data_train_par1.shape[1]] = data_train_par1
                    data_test1[:, sample_test_p :sample_test_p + data_test_par1.shape[1]] = data_test_par1
                    data_vld1[:, sample_vld_p : sample_vld_p + data_vld_par1.shape[1]] = data_vld_par1

                    data_train2[:, sample_train_p:sample_train_p + data_train_par2.shape[1]] = data_train_par2
                    data_test2[:, sample_test_p:sample_test_p + data_test_par2.shape[1]] = data_test_par2
                    data_vld2[:, sample_vld_p: sample_vld_p + data_vld_par2.shape[1]] = data_vld_par2

                    sample_train_p = sample_train_p + data_train_par1.shape[1]
                    sample_test_p = sample_test_p + data_test_par1.shape[1]
                    sample_vld_p = sample_vld_p + data_vld_par1.shape[1]

            len_train_arr1 = len_train_arr1[:, :]
            len_test_arr1  = len_test_arr1[:, :]
            len_vld_arr1 = len_vld_arr1[:, :]
            data_train1 = data_train1[:, 0:sample_train_p]
            data_test1  = data_test1[:, 0:sample_test_p]
            data_vld1 = data_vld1[:, 0:sample_vld_p]

            len_train_arr2 = len_train_arr2[:, :]
            len_test_arr2 = len_test_arr2[:, :]
            len_vld_arr2 = len_vld_arr2[:, :]
            data_train2 = data_train2[:, 0:sample_train_p]
            data_test2 = data_test2[:, 0:sample_test_p]
            data_vld2 = data_vld2[:, 0:sample_vld_p]

            time_end = time.time()
            print('time cost', (time_end - time_start)/60, 'min')

            len_train_arr_vld1 = np.zeros([len(loc_arr1), dev_train_arr[-1]+1])
            len_test_arr_vld1  = np.zeros([len(loc_arr1), dev_train_arr[-1]+1])
            len_train_arr_vld2 = np.zeros([len(loc_arr2), dev_train_arr[-1] + 1])
            len_test_arr_vld2  = np.zeros([len(loc_arr2), dev_train_arr[-1] + 1])

            sample_train_vld_p = 0
            sample_test_vld_p  = 0

            begin_flag = 1
            for dev in dev_test_arr_vld:
                for rx in [rx_test_arr_vld]:
                    if mat_v7_3 == False:
                        data_dict = scio.loadmat(data_test_dir + '\\rx' + str(rx+1) +'\\dev' + str(dev) + \
                                                 '\\' + data_name1 + '.mat')
                        data1 = data_dict[data_name1]
                        data_dict = scio.loadmat(data_test_dir + '\\rx' + str(rx + 1) + '\\dev' + str(dev) + \
                                                 '\\' + data_name2 + '.mat')
                        data2 = data_dict[data_name2]


                    index = [i for i in range(data1.shape[1])]
                    random.seed(exp_seed)
                    random.shuffle(index)
                    data1 = data1[:, index]
                    data2 = data2[:, index]
                    # data = data[:,index]
                    data_train_par_vld1 = data1[:, 0:round(data1.shape[1] * train_ratio_vld)]
                    data_test_par_vld1  = data1[:,round(data1.shape[1] * train_ratio_vld):round(data1.shape[1] * train_ratio_vld) + round(data1.shape[1] * test_ratio_vld)]
                    data_train_par_vld2 = data2[:, 0:round(data2.shape[1] * train_ratio_vld)]
                    data_test_par_vld2  = data2[:,round(data2.shape[1] * train_ratio_vld):round(data2.shape[1] * train_ratio_vld) + round(data2.shape[1] * test_ratio_vld)]

                    len_train_arr_vld1[rx][dev] = data_train_par_vld1.shape[1]
                    len_test_arr_vld1[rx][dev]  = data_test_par_vld1.shape[1]
                    len_train_arr_vld2[rx][dev] = data_train_par_vld2.shape[1]
                    len_test_arr_vld2[rx][dev]  = data_test_par_vld2.shape[1]

                    data_train_vld1[:, sample_train_vld_p:sample_train_vld_p + data_train_par_vld1.shape[1]] = data_train_par_vld1
                    data_test_vld1[:, sample_test_vld_p:sample_test_vld_p + data_test_par_vld1.shape[1]] = data_test_par_vld1
                    data_train_vld2[:, sample_train_vld_p:sample_train_vld_p + data_train_par_vld2.shape[1]] = data_train_par_vld2
                    data_test_vld2[:, sample_test_vld_p:sample_test_vld_p + data_test_par_vld2.shape[1]] = data_test_par_vld2

                    sample_train_vld_p = sample_train_vld_p + data_train_par_vld1.shape[1]
                    sample_test_vld_p  = sample_test_vld_p + data_test_par_vld1.shape[1]

            len_train_arr_vld1 = len_train_arr_vld1[:, :]
            len_test_arr_vld1 = len_test_arr_vld1[:, :]
            data_train_vld1 = data_train_vld1[:, 0:sample_train_vld_p]
            data_test_vld1  = data_test_vld1[:, 0:sample_test_vld_p]

            len_train_arr_vld2 = len_train_arr_vld2[:, :]
            len_test_arr_vld2 = len_test_arr_vld2[:, :]
            data_train_vld2 = data_train_vld2[:, 0:sample_train_vld_p]
            data_test_vld2 = data_test_vld2[:, 0:sample_test_vld_p]

            data_train1 = data_train1.T
            data_vld1 = data_vld1.T
            data_test1  = data_test1.T
            data_train_vld1 = data_train_vld1.T
            data_test_vld1  = data_test_vld1.T

            data_train2 = data_train2.T
            data_vld2 = data_vld2.T
            data_test2 = data_test2.T
            data_train_vld2 = data_train_vld2.T
            data_test_vld2 = data_test_vld2.T

            del data1, data_train_par1, data_vld_par1, data_test_par1, data_train_par_vld1, data_test_par_vld1
            del data2, data_train_par2, data_vld_par2, data_test_par2, data_train_par_vld2, data_test_par_vld2
            gc.collect()

            if complex_flag == False:
                framelen = int(data_train1.shape[1]/channel_num)
            else:
                framelen = int(data_train1.shape[1])

            len_train1 = np.sum(len_train_arr1, axis=0)
            len_vld1 = np.sum(len_vld_arr1, axis=0)
            len_test1  = np.sum(len_test_arr1, axis=0)
            len_train_vld1 = np.sum(len_train_arr_vld1, axis=0)
            len_test_vld1 = np.sum(len_test_arr_vld1, axis=0)

            len_train2 = np.sum(len_train_arr2, axis=0)
            len_vld2 = np.sum(len_vld_arr2, axis=0)
            len_test2 = np.sum(len_test_arr2, axis=0)
            len_train_vld2 = np.sum(len_train_arr_vld2, axis=0)
            len_test_vld2 = np.sum(len_test_arr_vld2, axis=0)

            len_train1 = len_train1[len_train1 != 0]
            len_vld1 = len_vld1[len_vld1 != 0]
            len_test1 = len_test1[len_test1 != 0]
            len_train_vld1 = len_train_vld1[len_train_vld1 != 0]
            len_test_vld1 = len_test_vld1[len_test_vld1 != 0]

            len_train2 = len_train2[len_train2 != 0]
            len_vld2 = len_vld2[len_vld2 != 0]
            len_test2 = len_test2[len_test2 != 0]
            len_train_vld2 = len_train_vld2[len_train_vld2 != 0]
            len_test_vld2 = len_test_vld2[len_test_vld2 != 0]

            label = np.arange(0, len(dev_train_arr))
            len_train1 = len_train1.astype(int)
            len_vld1 = len_vld1.astype(int)
            len_test1  = len_test1.astype(int)
            len_train_vld1 = len_train_vld1.astype(int)
            len_test_vld1  = len_test_vld1.astype(int)

            len_train2 = len_train2.astype(int)
            len_vld2 = len_vld2.astype(int)
            len_test2 = len_test2.astype(int)
            len_train_vld2 = len_train_vld2.astype(int)
            len_test_vld2 = len_test_vld2.astype(int)

            y_train1 = np.repeat(label, len_train1)
            y_train1.shape = (len(y_train1),1)
            y_vld1 = np.repeat(label, len_vld1)
            y_vld1.shape = (len(y_vld1), 1)
            y_test1 = np.repeat(label, len_test1)
            y_test1.shape = (len(y_test1), 1)
            y_train_vld1 = np.repeat(label, len_train_vld1)
            y_train_vld1.shape = (len(y_train_vld1),1)
            y_test_vld1 = np.repeat(label, len_test_vld1)
            y_test_vld1.shape = (len(y_test_vld1), 1)

            y_train2 = np.repeat(label, len_train2)
            y_train2.shape = (len(y_train2),1)
            y_vld2 = np.repeat(label, len_vld2)
            y_vld2.shape = (len(y_vld2), 1)
            y_test2 = np.repeat(label, len_test2)
            y_test2.shape = (len(y_test2), 1)
            y_train_vld2 = np.repeat(label, len_train_vld2)
            y_train_vld2.shape = (len(y_train_vld2),1)
            y_test_vld2 = np.repeat(label, len_test_vld2)
            y_test_vld2.shape = (len(y_test_vld2), 1)

            if len(data_train1.shape) == 2:  # if univariate
                # add a dimension to make it multivariate with one dimension
                data_train1 = data_train1.reshape((data_train1.shape[0], data_train1.shape[1], 1))
                data_test1 = data_test1.reshape((data_test1.shape[0], data_test1.shape[1], 1))
                data_vld1 = data_vld1.reshape((data_vld1.shape[0], data_vld1.shape[1], 1))
                data_train_vld1 = data_train_vld1.reshape((data_train_vld1.shape[0], data_train_vld1.shape[1], 1))
                data_test_vld1 = data_test_vld1.reshape((data_test_vld1.shape[0], data_test_vld1.shape[1], 1))

            if len(data_train2.shape) == 2:  # if univariate
                # add a dimension to make it multivariate with one dimension
                data_train2 = data_train2.reshape((data_train2.shape[0], data_train2.shape[1], 1))
                data_test2 = data_test2.reshape((data_test2.shape[0], data_test2.shape[1], 1))
                data_vld2 = data_vld2.reshape((data_vld2.shape[0], data_vld2.shape[1], 1))
                data_train_vld2 = data_train_vld2.reshape((data_train_vld2.shape[0], data_train_vld2.shape[1], 1))
                data_test_vld2 = data_test_vld2.reshape((data_test_vld2.shape[0], data_test_vld2.shape[1], 1))

            output_directory = root_dir + '\\results3\\' + classifier_name
            output_directory1 = root_dir + '\\results1\\' + classifier_name
            output_directory2 = root_dir + '\\results2\\' + classifier_name

            test_dir_df_metrics = output_directory1 + 'df_metrics.csv'

            create_directory(output_directory1)
            create_directory(output_directory2)
            if classifier_name == 'inception_v9_v2':
                from classifiers import inception_v9_v2

            input_shape1 = data_train1.shape[1:]
            input_shape2 = data_train2.shape[1:]
            nb_classes = len(dev_train_arr)
            # verbose = False
            # transform the labels from integers to one hot vectors
            enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
            enc.fit(np.concatenate((y_train1, y_test1), axis=0).reshape(-1, 1))
            y_train_arr1 = y_train1
            y_vld_arr1 = y_vld1
            y_test_arr1 = y_test1
            y_train1 = enc.transform(y_train1.reshape(-1, 1)).toarray()
            y_vld1 = enc.transform(y_vld1.reshape(-1, 1)).toarray()
            y_test1 = enc.transform(y_test1.reshape(-1, 1)).toarray()

            enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
            enc.fit(np.concatenate((y_train2, y_test2), axis=0).reshape(-1, 1))
            y_train_arr2 = y_train2
            y_vld_arr2 = y_vld2
            y_test_arr2 = y_test2
            y_train2 = enc.transform(y_train2.reshape(-1, 1)).toarray()
            y_vld2 = enc.transform(y_vld2.reshape(-1, 1)).toarray()
            y_test2 = enc.transform(y_test2.reshape(-1, 1)).toarray()

            if soft_label == True:
                y_train1 = label_smoothing(y_train1, soft_factor)
                y_train2 = label_smoothing(y_train2, soft_factor)

            enc.fit(np.concatenate((y_train_vld1, y_test_vld1), axis=0).reshape(-1, 1))
            y_train_vld_arr1 = y_train_vld1
            y_test_vld_arr1  = y_test_vld1
            y_train_vld1 = enc.transform(y_train_vld1.reshape(-1, 1)).toarray()
            y_test_vld1 = enc.transform(y_test_vld1.reshape(-1, 1)).toarray()

            enc.fit(np.concatenate((y_train_vld2, y_test_vld2), axis=0).reshape(-1, 1))
            y_train_vld_arr2 = y_train_vld2
            y_test_vld_arr2 = y_test_vld2
            y_train_vld2 = enc.transform(y_train_vld2.reshape(-1, 1)).toarray()
            y_test_vld2 = enc.transform(y_test_vld2.reshape(-1, 1)).toarray()

            # save orignal y because later we will use binary
            y_vld_true1 = np.argmax(y_vld1, axis=1)
            y_test_true1 = np.argmax(y_test1, axis=1)
            y_train_vld_true1 = np.argmax(y_train_vld1, axis=1)
            y_test_vld_true1 = np.argmax(y_test_vld1, axis=1)

            y_vld_true2 = np.argmax(y_vld2, axis=1)
            y_test_true2 = np.argmax(y_test2, axis=1)
            y_train_vld_true2 = np.argmax(y_train_vld2, axis=1)
            y_test_vld_true2 = np.argmax(y_test_vld2, axis=1)

            model1 = inception_v9_v2.Classifier_INCEPTION_V9(output_directory1, input_shape1, nb_classes, verbose=verbose, nb_epochs=nb_epochs, batch_size = batch_size, depth=depth, lr = learning_rate, l2=l2)

            gc.collect()
            print('#############Start the training process 1#############')
            history1 = model1.fit(data_train1, y_train1, data_vld1, y_vld1, y_vld_true1)

            model2 = inception_v9_v2.Classifier_INCEPTION_V9(output_directory2, input_shape2,
                                                                         nb_classes, verbose=verbose,
                                                                         nb_epochs=nb_epochs,
                                                                         batch_size=batch_size, depth=depth,
                                                                         lr=learning_rate, l2=l2)

            gc.collect()
            print('#############Start the training process 2#############')
            history2 = model2.fit(data_train2, y_train2, data_vld2, y_vld2, y_vld_true2)

            y_predict_test_acc1 = model1.predict(data_test1, y_test_true1, data_train1, y_train1, y_test1,return_df_metrics=False)
            y_predict_vld_acc1 = model1.predict(data_vld1, y_vld_true1, data_train1, y_train1, y_vld1,return_df_metrics=False)
            y_predict_test_vld_acc1 = model1.predict(data_test_vld1, y_test_vld_true1, data_train1, y_train1, y_test_vld1,return_df_metrics=False)

            y_predict_test_acc2 = model2.predict(data_test2, y_test_true2, data_train2, y_train2, y_test2,return_df_metrics=False)
            y_predict_vld_acc2 = model2.predict(data_vld2, y_vld_true2, data_train2, y_train2, y_vld2,return_df_metrics=False)
            y_predict_test_vld_acc2 = model2.predict(data_test_vld2, y_test_vld_true2, data_train2,y_train2, y_test_vld2,return_df_metrics=False)

            y_pred = np.argmax(y_predict_test_acc1+y_predict_test_acc2, axis=1)
            y_predict_test_acc = calculate_metrics(y_test_true1, y_pred, 0.0)
            y_pred = np.argmax(y_predict_test_vld_acc1 + y_predict_test_vld_acc2, axis=1)
            y_predict_test_vld_acc = calculate_metrics(y_test_vld_true1, y_pred, 0.0)

            print('#############', 'feature = ',  data_name1, ', test = ', rx_test_arr_vld+1, ', train = ', rx_train_arr+1,  ', exp_idx = ', exp_seed + 1,
                  '#############')
            print('y_predict_test:\n', y_predict_test_acc)
            print('y_predict_test_vld:\n', y_predict_test_vld_acc)

            cm = confusion_matrix(y_test_vld_true1, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('y_test_arr')
            plt.savefig(output_directory + str(exp_seed) + 'y_test_arr_cm.jpg')
            mdict = {'y_test_arr': cm}
            savemat(output_directory + str(exp_seed) + 'y_test_arr.mat', mdict)

            cm = confusion_matrix(y_test_vld_true1, y_pred)
            cm = cm/cm.astype(np.float64).sum(axis=1,keepdims=True)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('y_test_arr_per')
            plt.savefig(output_directory + 'y_test_arr_cm_per.jpg')
            mdict = {'y_test_arr_per': cm}
            savemat(output_directory + 'y_test_arr_per.mat', mdict)