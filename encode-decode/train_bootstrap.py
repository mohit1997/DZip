import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, Input, add, concatenate, Lambda, TimeDistributed
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import argparse
from keras.callbacks import CSVLogger
from keras import backend as K
from utils import *
from models import *
np.random.seed(0)
tf.set_random_seed(0)


def fit_model(X, Y, bs, nb_epoch, preprocessor, num_classes):
    y = Y
    mul = len(Y)/5e7
    decayrate = mul/(len(Y) // bs)
    # lr = 1e-3 for synthetic datasets and lr = 5e-3 for real datasets
    optim = keras.optimizers.Adam(lr=5e-3, beta_1=0.9, beta_2=0.999, clipnorm=0.1)
    preprocessor.compile(loss={'1': loss_fn}, optimizer=optim, metrics=['acc'])
    checkpoint = ModelCheckpoint(FLAGS.file_name + "_" + FLAGS.model, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log_{}_{}_bootstrap".format(FLAGS.file_name, FLAGS.model), append=True, separator=',')
    early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # lr_manager = OneCycleLR(10**(-1.2), bs, len(y), nb_epoch, end_percentage=0.3)

    callbacks_list = [checkpoint, csv_logger, early_stopping]

    train_gen = DataGenerator(X, y, bs, num_classes, shuffle=True, use_model=False)
    # val_gen = DataGenerator(X, y, teacher, bs, n_classes, True, use_model=False)

    return preprocessor.fit_generator(train_gen, epochs=nb_epoch, verbose=1, callbacks=callbacks_list, use_multiprocessing=True, workers=0)

batch_size=2048
sequence_length=64

import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10',
                        help='The name of the input file')
    # parser.add_argument('--log_file', type=str, default='logs.txt',
    #                     help='Name for the log file')
    parser.add_argument('--model', type=str, default='biGRU_jump',
                        help='Name for the log file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Name for the log file')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Name for the log file')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()
num_epochs=FLAGS.epochs
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


sequence = np.load(FLAGS.file_name + ".npy")
n_classes = len(np.unique(sequence))
sequence = sequence

X, Y = generate_single_output_data(sequence, batch_size, sequence_length)


toprint = [FLAGS.file_name, len(sequence), FLAGS.model]

PRNN = eval(FLAGS.model)(batch_size, sequence_length, n_classes)

out = fit_model(X, Y, batch_size, num_epochs, PRNN, n_classes)

