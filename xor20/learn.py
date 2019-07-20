import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, inputs, labels, batch_size=32, n_classes=10, shuffle=True):
		# 'Initialization'
		self.batch_size = batch_size
		self.labels = labels
		self.inputs = inputs
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.labels) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)
		

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.labels))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):

		X = self.inputs[list_IDs_temp]
		y = self.labels[list_IDs_temp]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, Input, add, concatenate, Lambda 
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from math import sqrt
import keras.regularizers as regularizers
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
from keras import backend as K
from clr import LRFinder, OneCycleLR

def res_block(inp, units=512, activation='relu'):
	res = inp

	x = BatchNormalization()(res)
	x = Activation(activation)(x)
	x = Dense(units)(x)

	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = Dense(units)(x)

	out = add([x, res])

	return out

def resnet(bs, time_steps, alphabet_size):
	inputs_bits = Input(shape=(time_steps,))
	temp = Input(tensor=tf.constant([1.0]))
	emb = Embedding(alphabet_size, 64)(inputs_bits)

	flat = Flatten()(emb)
	x = Dense(2048, kernel_regularizer=None)(flat)

	x = res_block(x, 2048, 'relu')
	x = res_block(x, 2048, 'relu')

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(2048)(x)

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(alphabet_size)(x)
	x = Lambda(lambda array: array[0]/array[1])([x,temp])
	x = Activation('softmax')(x)
	model = Model([inputs_bits,temp], x)
	return model


def loss_fn(y_true, y_pred):
	return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred+1e-8)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
	nrows = ((a.size - L) // S) + 1
	n = a.strides[0]
	return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate_single_output_data(series,batch_size,time_steps):
	series = series.reshape(-1, 1)

	series = series.reshape(-1)
	series = series.copy()
	data = strided_app(series, time_steps+1, 1)
	l = int(len(data)/batch_size) * batch_size

	data = data[:l] 
	X = data[:, :-1]
	Y = data[:, -1:]

	return X,Y

		
def fit_model(X, Y, bs, nb_epoch, model):
  y = Y
  optim = optim = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
  model.compile(loss=loss_fn, optimizer=optim, metrics=['acc'])
  checkpoint = ModelCheckpoint("model", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
  csv_logger = CSVLogger("log_teacher.csv", append=True, separator=';')
  early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

  lr_manager = OneCycleLR(10**(-2.0), bs, len(y), nb_epoch, end_percentage=0.3)

  callbacks_list = [checkpoint, csv_logger, early_stopping, lr_manager]
  
  indices = np.arange(X.shape[-1]).reshape(1, -1)
  train_gen = DataGenerator(X, y, bs, n_classes, True)

  model.fit_generator(train_gen, epochs=nb_epoch, verbose=1, callbacks=callbacks_list, use_multiprocessing=True, workers=0)

batch_size=2048
sequence_length=64
num_epochs=10
noise = 0.0

def biGRU_big(bs,time_steps,alphabet_size):
  model = Sequential()
  model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
  model.add(Bidirectional(CuDNNGRU(128, stateful=False, return_sequences=True)))
  model.add(Flatten())
  model.add(Dense(128*time_steps, activation='relu'))
  model.add(Dense(128*time_steps))
  model.add(Reshape((time_steps, 128,)))
  model.add(Bidirectional(CuDNNGRU(128, stateful=False, return_sequences=False)))
  # model.add(Dense(64, activation='relu'))
  model.add(Dense(alphabet_size, activation='softmax'))
  return model

print("Calling")
sequence = np.load('output.npy')
n_classes = len(np.unique(sequence))
sequence = sequence

X, Y = generate_single_output_data(sequence, batch_size, sequence_length)
# valX, valY = generate_single_output_data(sequence[10000000:12000000], batch_size, sequence_length)
print(Y.shape[1])

model = resnet(batch_size, sequence_length, n_classes)
fit_model(X, Y, batch_size, num_epochs, model)
