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
# rx_train_arr_total = np.array([0])
rx_test_arr_vld_total = np.array([0, 2])
# rx_test_arr_vld_total = np.array([2])
nb_epochs = 500

data_name1 = 'wifi_feature_1_normal'
data_name2 = 'wifi_feature_1_normal'
model_type = 'transformer'
# model_type = 'lstm'

seed_num = 5
class_acc_same_rx_arr = np.zeros((len(rx_test_arr_vld_total), len(rx_train_arr_total)))
class_acc_diff_rx_arr = np.zeros((len(rx_test_arr_vld_total), len(rx_train_arr_total)))
rx_test_idx = 0
for rx_test_arr_vld in rx_test_arr_vld_total:
    rx_train_idx = 0
    for rx_train_arr in rx_train_arr_total:
        class_acc_same_rx_seed = np.zeros(seed_num)
        class_acc_diff_rx_seed = np.zeros(seed_num)
        for exp_seed in range(seed_num):
            data_train_date = '20240609'
            data_test_date = '20240609'

            mat_v7_3 = False

            mul_loc = False
            dev_partial = False

            soft_label = True
            soft_factor = 0.1
            batch_size = 64
            learning_rate = 0.001
            l2 = 0.1
            depth = 6
            verbose = 2

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

                data_train_dir = r'*Data folder*' + '\\' + data_train_date
                data_test_dir  = r'*Data folder*' + '\\' + data_test_date

            dev_train_arr = np.array([1, 2, 3, 6, 7, 8, 9, 12, 13, 14])

            dev_test_arr_vld = dev_train_arr

            labels = ["dev1", "dev2", "dev3", "dev6", "dev7", "dev8", "dev9", "dev12", "dev13", "dev14"]

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

            train_generator = data_generator(data_train, y_train, batch_size)
            valid_generator = data_generator(data_vld, y_vld, batch_size)

            '''Define the neural network'''
            if model_type == 'transformer':
                model = transformer_dense_layer(maximum_position_encoding=254,
                                         embed_dim=64, num_heads=8, num_classes=len(dev_train_arr), l2=l2)
            elif model_type == 'lstm':
                model = lstm_model_dense_layer(num_classes=len(dev_train_arr), embed_dim=64, l2=l2)

            '''Training configurations'''
            reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=60,
                                                      min_lr=0.0001)
            file_path = 'results\\frequency_selective_fading_transformer_best_model.hdf5'
            model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)
            callbacks = [reduce_lr, model_checkpoint]
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),metrics=['accuracy'])

            history = model.fit(train_generator,
                                  steps_per_epoch=sample_train_p // batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=sample_vld_p // batch_size,
                                  epochs=nb_epochs,
                                  verbose=0,
                                  callbacks=callbacks)

            tf.keras.models.save_model(model, 'results\\frequency_selective_fading_transformer_offline.h5')

            pred_prob = model.predict(data_test)
            pred_label = pred_prob.argmax(axis=-1)
            conf_mat = confusion_matrix(y_test_true, pred_label)
            acc = accuracy_score(y_test_true, pred_label)
            print(f'Train Rx = {rx_train_arr + 1}, Test Rx = {rx_test_arr_vld + 1}, Seed = {exp_seed + 1}')
            print('Same Rx, Acc = %.4f' % acc)

            class_acc_same_rx_seed[exp_seed] = acc

            pred_prob = model.predict(data_test_vld)
            pred_label = pred_prob.argmax(axis=-1)
            conf_mat = confusion_matrix(y_test_vld_true, pred_label)
            acc = accuracy_score(y_test_vld_true, pred_label)
            print('Diff Rx, Acc = %.4f' % acc)

            class_acc_diff_rx_seed[exp_seed] = acc

        class_acc_same_rx_arr[rx_test_idx, rx_train_idx] = class_acc_same_rx_seed.mean()
        class_acc_diff_rx_arr[rx_test_idx, rx_train_idx] = class_acc_diff_rx_seed.mean()
        rx_train_idx = rx_train_idx + 1

    rx_test_idx = rx_test_idx + 1

mdict = {'acc_240609_transformer_dense_layer_same_rx_arr': class_acc_same_rx_arr}
savemat('acc_240609_transformer_dense_layer_same_rx_arr.mat', mdict)
mdict = {'acc_240609_transformer_dense_layer_diff_rx_arr': class_acc_diff_rx_arr}
savemat('acc_240609_transformer_dense_layer_diff_rx_arr.mat', mdict)