import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

np.random.seed(0)
tf.set_random_seed(0)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, labels, batch_size=32, n_classes=10, shuffle=True, use_model=False):
        # 'Initialization'
        self.batch_size = batch_size
        self.inputs = inputs
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle

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

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

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
