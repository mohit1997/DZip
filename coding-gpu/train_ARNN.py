from __future__ import print_function
import numpy as np
import keras
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from math import sqrt
import keras.regularizers as regularizers
from keras.layers.embeddings import Embedding
import tensorflow as tf
import argparse
from keras import backend as K
from utils import *
from models import *
import sys

def iterate_minibatches(inputs, targets, batchsize, n_classes, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        # if(start_idx + batchsize >= inputs.shape[0]):
        #   break;

        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], keras.utils.to_categorical(targets[excerpt], num_classes=n_classes)
        
def fit_model(X, Y, bs, ARNN):
    y = Y
    optim = tf.train.AdamOptimizer(learning_rate=5e-4)
    ARNN.compile(loss={'1': loss_fn, '2': loss_fn}, loss_weights=[1.0, 0.1], optimizer=optim, metrics=['acc'])
    
    i = 0
    loss_list = []
    for batch_x, batch_y in iterate_minibatches(X, y, bs, n_classes, shuffle=False):
	i = i+1
	out = ARNN.test_on_batch(batch_x, [batch_y, batch_y])
        loss_list.append(out[1])
	out = ARNN.train_on_batch(batch_x, [batch_y, batch_y])
	if i%10==0:
	    sys.stdout.flush()
	    print('Batch {}/{} Loss = {:4f}'.format(i, len(y)//bs, np.mean(loss_list)), end='\r')
        if i%100000==0:
            np.save('ARNN_{}_losslist'.format(FLAGS.file_name), np.array(loss_list))
	
    print('Training Complete Batch {}/{} Loss = {:4f}'.format(i, len(y)//bs, np.mean(loss_list)))
    np.save('ARNN_{}_losslist'.format(FLAGS.file_name), np.array(loss_list))


batch_size=64
sequence_length=64

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10',
                        help='The name of the input file')
    parser.add_argument('--ARNN', type=str, default='biGRU_big',
                        help='Name for the ARNN architecture')
    parser.add_argument('--PRNN', type=str, default='biGRU_jump',
                        help='Name for the PRNN architecture')
    parser.add_argument('--gpu', type=str, default='1',
                        help='Name for the log file')
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

sequence = np.load(FLAGS.file_name + ".npy")
n_classes = len(np.unique(sequence))
sequence = sequence

X, Y = generate_single_output_data(sequence, batch_size, sequence_length)

ARNN, PRNN = eval(FLAGS.ARNN)(batch_size, sequence_length, n_classes)

optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
PRNN.compile(loss=loss_fn, optimizer=optim, metrics=['acc'])
PRNN.load_weights("{}_{}".format(FLAGS.file_name, FLAGS.PRNN))

for l in PRNN.layers:
    l.trainable = True

ARNN.summary()

fit_model(X, Y, batch_size, ARNN)
