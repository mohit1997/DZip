# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#
 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.layers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
from models import *
from utils import *
import tempfile
import shutil
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['PYTHONHASHSEED']=str(0)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(0)

np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-batch_size', action='store', dest='batch_size', type=int,
                    help='model file')
parser.add_argument('-data', action='store', dest='sequence_npy_file',
                    help='data file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-output', action='store',dest='output_file_prefix',
                    help='compressed file name')
parser.add_argument('-gpu', action='store', dest='gpu_id', default="1",
                    help='params file')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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
        yield inputs[excerpt], targets[excerpt]

def predict_lstm(X, y_original, timesteps, bs, alphabet_size, model_name):
	ARNN, PRNN = eval(model_name)(bs, timesteps, alphabet_size)
	PRNN.load_weights(args.model_weights_file)

	decayrate = 32*2.0/(len(X) // bs)
	optim = keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decayrate, amsgrad=False)
	ARNN.compile(loss={'1': loss_fn, '2': loss_fn}, loss_weights=[1.0, 0.1], optimizer=optim, metrics=['acc'])
	l = int(len(X)/bs)*bs

	f = open(args.file_prefix, 'wb')
	bitout = arithmeticcoding_fast.BitOutputStream(f)
	enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
	prob = np.ones(alphabet_size)/alphabet_size
	cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
	cumul[1:] = np.cumsum(prob*10000000 + 1)
	for j in range(timesteps):
		enc.write(cumul, X[0, j])
	cumul = np.zeros((1, alphabet_size+1), dtype = np.uint64)
	for bx, by in iterate_minibatches(X[:l], y_original[:l], bs):
		for j in range(bs):
			prob, _ = ARNN.predict(bx[j:j+1], batch_size=1)
			cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
			enc.write(cumul[0, :], int(by[j]))
		onehot = keras.utils.to_categorical(by, num_classes=alphabet_size)
		ARNN.train_on_batch(bx, [onehot, onehot])
	if len(X[l:]) > 0:
		# prob, _ = ARNN.predict(X[l:], batch_size=len(X[l:]))
		cumul = np.zeros((1, alphabet_size+1), dtype = np.uint64)
		for i in range(len(y_original[l:])):
			prob, _ = ARNN.predict(X[l:][i:i+1], batch_size=1)
			cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
			enc.write(cumul[0, :], int(y_original[l:][i]))

	enc.finish()
	bitout.close()
	f.close()

def main():
    np.random.seed(0)
    tf.set_random_seed(0)
    args.file_prefix = args.output_file_prefix + ".dzip"
    sequence = np.load(args.sequence_npy_file + ".npy")[:500]
    n_classes = len(np.unique(sequence))
    batch_size = args.batch_size
    timesteps = 64
        
    with open(args.params_file, 'r') as f:
        params = json.load(f)

    params['len_series'] = len(sequence)
    params['bs'] = batch_size
    params['timesteps'] = timesteps

    with open(args.output_file_prefix+'.params','w') as f:
        json.dump(params, f, indent=4)

    sequence = sequence.reshape(-1)
    series = sequence.copy()
    data = strided_app(series, timesteps+1, 1)
    X = data[:, :-1]
    Y_original = data[:, -1:]
    predict_lstm(X, Y_original, timesteps, batch_size, n_classes, args.model_name) 


if __name__ == "__main__":
        main()

