from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import tensorflow as tf
import scipy.io as scio
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import random
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.io import savemat
import h5py
import pylab
import gc
from deep_learning_models import transformer_dense_layer, lstm_model_dense_layer, FCN_model, gru_model
from deep_learning_models import TransformerEncoder, PositionEmbedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.utils import to_categorical
from LengthVersatileUtils import load_single_file, shuffle, awgn, ch_ind_spectrogram, data_generator

time_start = time.time()

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

rx_train_arr_total = np.array([0, 1, 2, 3, 4])
# rx_train_arr_total = np.array([1])
# test_arr = np.array([0, 2])
test_arr = np.array([0, 2])
nb_epochs = 500

data_name1 = 'my_lstf_feature_normal'
data_name2 = 'my_lltf_feature_normal'
model_type = 'transformer'
# model_type = 'lstm'

seed_num = 5
class_acc_same_rx_arr = np.zeros((len(test_arr), len(rx_train_arr_total)))
class_acc_diff_rx_arr = np.zeros((len(test_arr), len(rx_train_arr_total)))
rx_test_idx = 0
for rx_test_arr_vld in test_arr:
    rx_train_idx = 0
    for rx_train_arr in rx_train_arr_total:
        class_acc_same_rx_seed = np.zeros(seed_num)
        class_acc_diff_rx_seed = np.zeros(seed_num)
        for exp_seed in range(seed_num):
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

            # For local vld ##################################################
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

                data_train_dir = r'*Data folder*' + '\\' + data_train_date
                data_test_dir  = r'*Data folder*' + '\\' + data_test_date

            dev_train_arr = np.array([4, 5, 6, 7, 8, 10, 11, 12, 13, 14])

            dev_test_arr_vld = dev_train_arr
            if dev_partial == True:
                dev_train_arr = np.array([0, 1, 2, 3, 5, 9])
                dev_test_arr_vld = np.array([0, 1, 2, 3, 5, 9])

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

            input_shape1 = data_train1.shape[1:]
            input_shape2 = data_train2.shape[1:]
            nb_classes = len(dev_train_arr)
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

            train_generator1 = data_generator(data_train1, y_train1, batch_size)
            valid_generator1 = data_generator(data_vld1, y_vld1, batch_size)

            train_generator2 = data_generator(data_train2, y_train2, batch_size)
            valid_generator2 = data_generator(data_vld2, y_vld2, batch_size)

            '''Define the neural network'''
            if model_type == 'transformer':
                model1 = transformer_dense_layer(maximum_position_encoding=254,
                                    embed_dim=64, num_heads=8, num_classes=len(dev_train_arr), l2=l2)
                model2 = transformer_dense_layer(maximum_position_encoding=254,
                                    embed_dim=64, num_heads=8, num_classes=len(dev_train_arr), l2=l2)
            elif model_type == 'lstm':
                model1 = lstm_model_dense_layer(num_classes=len(dev_train_arr), embed_dim=64, l2=l2)
                model2 = lstm_model_dense_layer(num_classes=len(dev_train_arr), embed_dim=64, l2=l2)

            '''Training configurations'''
            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=60,
                                                          min_lr=0.0001)
            file_path = 'results\\flat_fading_best_model.hdf5'
            model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                               save_best_only=True)
            callbacks = [reduce_lr, model_checkpoint]

            opt = Adam(learning_rate=1e-3)

            model1.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
            model2.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

            history1 = model1.fit(train_generator1,
                                steps_per_epoch=sample_train_p // batch_size,
                                validation_data=valid_generator1,
                                validation_steps=sample_vld_p // batch_size,
                                epochs=nb_epochs,
                                verbose=verbose,
                                callbacks=callbacks)
            history2 = model2.fit(train_generator2,
                                steps_per_epoch=sample_train_p // batch_size,
                                validation_data=valid_generator2,
                                validation_steps=sample_vld_p // batch_size,
                                epochs=nb_epochs,
                                verbose=verbose,
                                callbacks=callbacks)

            tf.keras.models.save_model(model1, 'results\\transformer_offline1.h5')
            tf.keras.models.save_model(model2, 'results\\transformer_offline2.h5')

            plt.figure()
            plt.plot(range(1,nb_epochs+1), history2.history['loss'], 'b', label='Training loss')
            plt.plot(range(1,nb_epochs+1), history2.history['val_loss'], 'r', label='Validation val_loss')
            plt.title('Traing and Validation loss')
            plt.legend()
            save_path = f'results\Direct_Trans_TrainRx_{rx_train_arr}_TestRx_{rx_test_arr_vld}_Seed_{exp_seed}_model_loss.jpg'
            plt.savefig(save_path)
            plt.ioff()
            plt.show(block=True)

            pred_prob1 = model1.predict(data_test1)
            pred_prob2 = model2.predict(data_test2)

            pred_prob = pred_prob1 + pred_prob2
            pred_label = pred_prob.argmax(axis=-1)
            conf_mat = confusion_matrix(y_test_true1, pred_label)
            acc = accuracy_score(y_test_true1, pred_label)
            print(f'Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Seed = {exp_seed + 1}')
            print('Same Rx, Acc = %.4f' % acc)

            class_acc_same_rx_seed[exp_seed] = acc

            pred_prob1 = model1.predict(data_test_vld1)
            pred_prob2 = model2.predict(data_test_vld2)

            pred_prob = pred_prob1 + pred_prob2
            pred_label = pred_prob.argmax(axis=-1)
            conf_mat = confusion_matrix(y_test_vld_true1, pred_label)
            acc = accuracy_score(y_test_vld_true1, pred_label)
            print('Diff Rx, Acc = %.4f' % acc)

            class_acc_diff_rx_seed[exp_seed] = acc

        class_acc_same_rx_arr[rx_test_idx, rx_train_idx] = class_acc_same_rx_seed.mean()
        class_acc_diff_rx_arr[rx_test_idx, rx_train_idx] = class_acc_diff_rx_seed.mean()
        rx_train_idx = rx_train_idx + 1

    rx_test_idx = rx_test_idx + 1

mdict = {'acc_240517_transformer_dense_layer_same_rx_arr': class_acc_same_rx_arr}
savemat('acc_240517_transformer_dense_layer_same_rx_arr.mat', mdict)
mdict = {'acc_240517_transformer_dense_layer_diff_rx_arr': class_acc_diff_rx_arr}
savemat('acc_240517_transformer_dense_layer_diff_rx_arr.mat', mdict)