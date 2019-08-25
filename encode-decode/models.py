import numpy as np
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

np.random.seed(0)
tf.set_random_seed(0)


def biGRU_jump(bs,time_steps,alphabet_size):
  jump = 16
  def my_shape(input_shape):
     print(np.ceil(float(input_shape[1])/jump))
     return tuple((input_shape[0],int(np.ceil(float(input_shape[1])/jump)),input_shape[2]))

  inputs_bits = Input(shape=(time_steps,))
  x   = Embedding(alphabet_size, 8,)(inputs_bits)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  x = Lambda(lambda tensor: tensor[:,::-jump,:][:,::-1,:], output_shape=my_shape)(x)
  # x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  flat = Flatten()(x)
  x = Dense(16, activation='relu')(flat)
  x = Add()([Dense(alphabet_size)(x),  Dense(alphabet_size)(flat)])
  
  s1 = Activation('softmax', name="1")(x)

  model = Model(inputs_bits, s1)
  
  return model

def biGRU(bs,time_steps,alphabet_size):
  inputs_bits = Input(shape=(time_steps,))
  x   = Embedding(alphabet_size, 8,)(inputs_bits)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=False))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  # x = Lambda(lambda tensor: tensor[:,::-jump,:][:,::-1,:], output_shape=my_shape)(x)
  # x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  x = Dense(8, activation='relu')(x)
  x = Dense(alphabet_size)(x)

  s1 = Activation('softmax', name="1")(x)

  model = Model(inputs_bits, s1)

  return model

def biGRU_stack(bs,time_steps,alphabet_size):
  inputs_bits = Input(shape=(time_steps,))
  x   = Embedding(alphabet_size, 8,)(inputs_bits)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  # x = TimeDistributed(Dense(8, activation='relu'))(x)
  # x = Bidirectional(CuDNNGRU(8, stateful=False, return_sequences=True))(x)
  flat = Flatten()(x)
  x = Dense(16, activation='relu')(flat)
  x = Add()([Dense(alphabet_size)(x),  Dense(alphabet_size)(flat)])

  s1 = Activation('softmax', name="1")(x)

  model = Model(inputs_bits, s1)

  return model

def FC(bs,time_steps,alphabet_size):
  inputs_bits = Input(shape=(time_steps,))
  x  = Embedding(alphabet_size, 8,)(inputs_bits)
  flat = Flatten()(x)
  x = Dense(128, activation='relu')(flat)
  x = Dense(128, activation='relu')(x)
  x = Dense(alphabet_size)(x)

  s1 = Activation('softmax', name="1")(x)

  model = Model(inputs_bits, s1)

  return model

