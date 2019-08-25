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
import sys

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-batch_size', action='store', dest='batch_size', type=int,
                    help='model file')
parser.add_argument('-input', action='store', dest='file_prefix',
                    help='compressed file')
parser.add_argument('-output', action='store',dest='output_file',
                    help='decompressed_file')

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

def predict_lstm(length, timesteps, bs, alphabet_size, model_name):
	model = eval(model_name)(bs, timesteps, alphabet_size)
	model.load_weights(args.model_weights_file)

        series = np.zeros((length), dtype=np.uint8)
        data = strided_app(series, timesteps+1, 1)

	X = data[:, :-1]
	y_original = data[:, -1:]
	l = int(len(X)/bs)*bs

	f = open(args.file_prefix + ".dzip", 'rb')
	bitin = arithmeticcoding_fast.BitInputStream(f)
	dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
	prob = np.ones(alphabet_size)/alphabet_size
	cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
	cumul[1:] = np.cumsum(prob*10000000 + 1)
	for j in range(timesteps):
		series[j] = dec.read(cumul, alphabet_size)

	cumul = np.zeros((1, alphabet_size+1), dtype = np.uint64)
	index = timesteps
	for bx, by in iterate_minibatches(X[:l], y_original[:l], 1):
		prob = model.predict(bx, batch_size=1)
		cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
		series[index] = dec.read(cumul[0, :], alphabet_size)
		index = index+1
	if len(X[l:]) > 0:
		for bx, by in iterate_minibatches(X[l:], y_original[l:], 1):
			prob = model.predict(bx, batch_size=1)
			cumul = np.zeros((1, alphabet_size+1), dtype = np.uint64)
			cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
			series[index] = dec.read(cumul[0, :], alphabet_size)
			index = index+1
	bitin.close()
	f.close()
	return series

def main():
    with open(args.file_prefix +".params", 'r') as f:
        param_dict = json.load(f)
    
    len_series = param_dict['len_series']
    batch_size = param_dict['bs']
    timesteps = param_dict['timesteps']
    id2char_dict = param_dict['id2char_dict'] 
    n_classes = len(id2char_dict)
    print(n_classes, len_series)
    
    series = np.zeros(len_series,dtype=np.uint8)
    series = predict_lstm(len_series, timesteps, batch_size, n_classes, args.model_name)

    f = open(args.output_file,'w')
    print(id2char_dict)
    print(series[:10])
    f.write(''.join([id2char_dict[str(s)] for s in series]))
    f.close()


if __name__ == "__main__":
        main()

