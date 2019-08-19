from __future__ import print_function
import numpy as np
import keras
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, Input, add, concatenate
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
from utils import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, labels, model, batch_size=32, n_classes=10, shuffle=True, use_model=False):
        # 'Initialization'
        self.batch_size = batch_size
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_model = use_model

        # layer_name = 'probs'
        # self.intermediate_layer_model = Model(inputs=model.input,
        #                          outputs=model.get_layer(layer_name).output)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        X = self.inputs[list_IDs_temp]
        y = self.labels[list_IDs_temp]
        true_lab = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, true_lab

def res_block(inp, units=512, activation='relu'):
    x = res = inp

    # x = BatchNormalization()(res)
    x = Activation(activation)(x)
    x = Dense(units)(x)

    # x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dense(units)(x)

    out = add([x, res])

    return out

def resnet(bs, time_steps, alphabet_size):
    inputs_bits = Input(shape=(time_steps,))

    emb = Embedding(alphabet_size, 64)(inputs_bits)

    flat = Flatten()(emb)
    x = Dense(2048, kernel_regularizer=None)(flat)

    x = res_block(x, 2048, 'relu')
    x = res_block(x, 2048, 'relu')

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(2048, name='feats')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(alphabet_size, activation='softmax', name='probs')(x)

    model = Model(inputs_bits, x)
    return model


def loss_fn(y_true, y_pred):
    return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred[:, 0:n_classes]+1e-8)

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

        
def fit_model(X, Y, bs, nb_epoch, student, teacher):
    y = Y
    decayrate = 16*2.0/(len(Y) // bs)
    optim = keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decayrate, amsgrad=False)
    student.compile(loss={'1': loss_fn}, optimizer=optim, metrics=['acc'])
    checkpoint = ModelCheckpoint("modeltransferexp", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log_transferexp.csv", append=True, separator=';')
    early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # lr_manager = OneCycleLR(10**(-1.2), bs, len(y), nb_epoch, end_percentage=0.3)

    callbacks_list = [checkpoint, csv_logger, early_stopping]

    indices = np.arange(X.shape[-1]).reshape(1, -1)
    train_gen = DataGenerator(X, y, teacher, bs, n_classes, shuffle=False)
    # val_gen = DataGenerator(X, y, teacher, bs, n_classes, True, use_model=False)

    # student.ifit_generator(train_gen, epochs=1, verbose=1, callbacks=callbacks_list, use_multiprocessing=True, workers=0
    # X = X[100000:]
    # y = y[100000:]
    i = 0
    loss_list = []
    for batch_x, batch_y in iterate_minibatches(X, y, bs, n_classes, shuffle=False):
	i = i+1
	out = student.test_on_batch(batch_x, batch_y)
        loss_list.append(out[0])
	out = student.train_on_batch(batch_x, batch_y)
	if i%10==0:
	    sys.stdout.flush()
	    print('Batch {}/{} Loss = {:4f}'.format(i, len(y)//bs, np.mean(loss_list)), end='\r')
        if i%100000==0:
            np.save('transfer_explosslist', np.array(loss_list))
	
    print('Training Complete Batch {}/{} Loss = {:4f}'.format(i, len(y)//bs, np.mean(loss_list)))
    np.save('transfer_explosslist', np.array(loss_list))


batch_size=64
sequence_length=64
num_epochs=10
noise = 0.0
jump = 16

def my_shape(input_shape):
    print(np.ceil(input_shape[1]/jump))
    return tuple((input_shape[0],int(np.ceil(float(input_shape[1])/jump)),input_shape[2]))


def biGRU_big(bs,time_steps,alphabet_size):
  inputs_bits = Input(shape=(time_steps,))
  x   = Embedding(alphabet_size, 8,)(inputs_bits)
  x = Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=True))(x)
  x = Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=True))(x)
  x = Lambda(lambda tensor: tensor[:,::-jump,:][:,::-1,:], output_shape=my_shape)(x)
  flat = Flatten()(x)
  prelogits = x = Dense(16, activation='relu')(flat)
  x = Add()([Dense(alphabet_size)(x),  Dense(alphabet_size)(flat)])
  x = Dense(alphabet_size, name='logits')(x)
  new_logits = Add()([Dense(alphabet_size)(prelogits),  Dense(alphabet_size)(flat)]) 
  s1 = Activation('softmax', name="1")(x)
  s2 = Activation('softmax', name="2")(x)
  s3 = Activation('softmax', name="3")(x)

  model_prev = Model(inputs_bits, s1)
  emb = Embedding(alphabet_size, 16)(inputs_bits)
  d = Bidirectional(CuDNNGRU(128, stateful=False, return_sequences=True))(emb)
  d = Bidirectional(CuDNNGRU(64, stateful=False, return_sequences=True))(d)
  d = Flatten()(d)
  flat2 = d = Concatenate()([d, flat])
  d = Dense(1024, activation='relu')(d)
  next_layer = Add()([Dense(alphabet_size)(flat2), Dense(alphabet_size)(d), new_logits])
  next_layer = Dense(alphabet_size)(next_layer)
  s1 = Activation('softmax', name="1")(next_layer)
  # s3 = Activation('softmax', name="3")(x)

  model = Model(inputs_bits, s1)

  return model, model_prev

print("Calling")
sequence = np.load('output.npy')
n_classes = len(np.unique(sequence))
sequence = sequence

X, Y = generate_single_output_data(sequence, batch_size, sequence_length)
# valX, valY = generate_single_output_data(sequence[10000000:12000000], batch_size, sequence_length)
print(Y.shape[1])


model_student, model_teacher = biGRU_big(batch_size, sequence_length, n_classes)

optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_teacher.compile(loss=loss_fn, optimizer=optim, metrics=['acc'])
model_teacher.load_weights('directexp')

for l in model_teacher.layers:
    l.trainable = True

model_student.summary()

fit_model(X, Y, batch_size, num_epochs, model_student, model_teacher)
