# coding: utf-8
"""
mlp class

"""
import keras
from keras.layers import Activation, Dropout, BatchNormalization, Dense, Input, Flatten
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, dense1=512, dense2=512,
                drop1=0.3, drop2=0.2,
                batch_size=12,
                activation='relu',
                optimizer='Adam'):
        
        self.dense1 = dense1
        self.dense2 = dense2
        self.drop1  = drop1
        self.drop2  = drop2
        self.activation = activation
        self.batch_size = batch_size
        self.validation_split = 0.2

        if opt == 'SGD1':
            print("optimizer is SGD lr = 0.01")
            self.opt = SGD(lr=0.01)
        elif str == 'SGD1':
            print("optimizer is SGD lr = 0.001")
            self.opt = SGD(lr=0.001)
        elif str == 'Adam':
            print("optimizer is Adam")
            self.opt = Adam()
        elif 
            print("Oops!")


        # load mnist data
        print(" load mnist data")
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_mnist_data()
        # build mlp model
        print("  build mlp model")
        self.model = self.mlp_model()

        params = """
        validation_split:\t{0}
        dense1:\t{1}
        dense2:\t{2}
        drop1:\t{3}
        drop2:\t{4}
        activation:\t{5}
        batch_size:\{6}
        """.format(self.validation_split,
                    self.dense1,
                    self.dense2,
                    self.drop1,
                    self.drop2,
                    self.activation,
                    self.batch_size,
                    self.opt
        )
        print(params)
        
    def plot_history(self, history):
        # 精度の履歴をプロット
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['acc', 'val_acc'], loc='lower right')
            plt.show()

            # Lossの履歴をプロット
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['loss', 'val_loss'], loc='lower right')
            plt.show()

    def load_mnist_data(self):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784).astype('float32')/255
        x_test = x_test.reshape(10000, 784).astype('float32')/255
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return x_train, x_test, y_train, y_test
    
    def mlp_model(self):
        """
        inputs =Input( shape=(784,))

        x = Dense(self.dense1, activation='sigmoid')(inputs)
        x = Dropout(self.drop1)(x)
        x = Dense(self.dense2, activation='sigmoid')(x)
        x = Dropout(self.drop2)(x)

        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        """

        model = Sequential()
        model.add(Dense(self.dense1, input_shape=(784,)))
        model.add(Activation(self.activation))
        model.add(Dropout(self.drop1))

        model.add(Dense(self.dense2))
        model.add(Activation(self.activation))
        model.add(Dropout(self.drop2))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model

    def train(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.model.compile(loss='categorical_crossentropy',
                            optimizer=self.opt,
                            metrics=['accuracy'])
        self.model.summary()

        history = self.model.fit(self.x_train, self.y_train,
                batch_size=self.batch_size,
                epochs=70,
                validation_split=self.validation_split,
                verbose=0,
                callbacks=[early_stopping])

        #self.plot_history(history)
        
    def mlp_evaluate(self):
        self.train()

        evaluate = self.model.evaluate(self.x_test, self.y_test, 
                                    batch_size = self.batch_size, verbose=0)
        return evaluate