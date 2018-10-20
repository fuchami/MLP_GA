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
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        # load mnist data
        print(" load mnist data")
        self._x_train, self._x_test, self._y_train, self._y_test = self.load_mnist_data()
        # build mlp model
        print("  build mlp model")
        self.model = self.mlp_model()

        params = """
        validation_split:\t{0}
        dense1:\t{1}
        dense2:\t{2}
        drop1:\t{3}
        drop2:\t{4}
        epochs:\t{5}
        batch_size:\{6}
        """.format{self.validation_split,
                    self.dense1,
                    self.dense2,
                    self.drop1,
                    self.drop2,
                    self.epochs,
                    self.batch_size
                    }
        print(params)
        

    def load_mnist_data(self):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784).astype('float32')/255
        x_test = x_test.reshape(60000, 784).astype('float32')/255
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return x_train, y_train, x_test, y_test
    
    def mlp_model(self):
        inputs =Input( shape=(784, ))

        x = Dense(self.dense1, activatio='sigmoid')(inputs)
        x = Dropout(self.drop1)(x)
        x = Dense(self.dense2, activation='sigmoid')(x)
        x = Dropout(self.drop2)(x)

        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def train(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.model.summary()
        self.model.fit(self.x_train, self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=[early_stopping])
        
    def mnits_evaluate():
        self.train()

        evaluate = self.model.evaluate(self.x_test, self.y_test, 
                                    batch_size = self.batch_size, verbose=0)
        return evaluate