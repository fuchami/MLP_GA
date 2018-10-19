# coding: utf-8
"""
mlp class

"""

import keras
from keras.layers import Activation, Dropout, BatchNormalization, Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

class MLP():
    def __init__(self, dense1=512, dense2=512,
                drop1=0.3, drop2=0.2,
                batch_size=12,
                epochs=10,
                validation_split=0.1):
        
        self.dense1 = dense1
        self.dense2 = dense2
        self.drop1  = drop1
        self.drop2  = drop2
        self.epoch  = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self._x_train, self._x_test, self._y_train, self._y_test = self.load_mnist_data()
        self.model = self.mlp_mnist()
        

    def load_mnist_data(self):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784).astype('float32')/255
        x_test = x_test.reshape(60000, 784).astype('float32')/255
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return x_train, y_train, x_test, y_test
    
    def mlp_mnist():
        inputs =Input( shape=(784, ))

        x = Dense(512, activatio='sigmoid')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='sigmoid')(x)
        x = Dropout(0.5)(x)

        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model