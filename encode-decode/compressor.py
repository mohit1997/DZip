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

args = parser.parse_args()


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
	model = eval(model_name)(bs, timesteps, alphabet_size)
	model.load_weights(args.model_weights_file)

	l = int(len(X)/bs)*bs

	f = open(args.file_prefix, 'wb')
	bitout = arithmeticcoding_fast.BitOutputStream(f)
	enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
	prob = np.ones(alphabet_size)/alphabet_size
	cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
	cumul[1:] = np.cumsum(prob*10000000 + 1)
	for j in range(timesteps):
		enc.write(cumul, X[0, j])
	cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
	for bx, by in iterate_minibatches(X[:l], y_original[:l], bs):
		prob = model.predict(bx, batch_size=bs)
		cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
		for j in range(bs):
			enc.write(cumul[j, :], by[j])
	if len(X[l:]) > 0:
		prob = model.predict(X[l:], batch_size=len(X[l:]))
		cumul = np.zeros((len(X[l:]), alphabet_size+1), dtype = np.uint64)
		cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
		for x, y in zip(cumul, y_original[l:]):
			enc.write(x, y)

	enc.finish()
	bitout.close()
	f.close()

def main():
    args.file_prefix = args.output_file_prefix
    sequence = np.load(args.sequence_npy_file + ".npy")
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

    X, Y_original = generate_single_output_data(sequence, batch_size, timesteps)
    predict_lstm(X, Y_original, timesteps, batch_size, n_classes, args.model_name) 


if __name__ == "__main__":
        main()

