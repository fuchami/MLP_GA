# coding:utf-8

import keras 
from keras.layers import Activation,Dropout,BatchNormalization,Dense,Input,Flatten
from keras.layers import Conv3D, MaxPool3D, MaxPooling3D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
import matplotlib as plt

import tools
import load

class Conv3DNet():
    def __init__(self, conv1, conv2, conv3, conv4, conv5,
                dense1, dense2, dropout, bn1, bn2, batch_size, opt):

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.dense1 = dense1
        self.dense2 = dense2
        self.dropout = dropout
        self.bn1 = bn1
        self.bn2 = bn2
        self.batchsize = batch_size

        if opt == 'sgd':
            print('optimizer is SGD')
            self.opt = SGD()
        elif opt == 'rmsprop':
            print('optimizer is RMSprop')
            self.opt = RMSprop()
        elif opt == 'adam':
            print("optimizer is Adam")
            self.opt = Adam()


        """ load data """
        print("load data")

        """ build 3dconv-net model """
        self.model = self.conv3_model()
    
        params = """
            optimizer:\t{0}
            conv1::\t{1}
            conv2::\t{2}
            conv3::\t{3}
            conv4::\t{4}
            conv5::\t{5}
            dense1:\t{6}
            dense2:\t{7}
            dropout:\t{8}
            bn1:\t{9}
            bn2:\t{10}
            batch_size:\{11}
            """.format(self.opt,
                        self.conv1,
                        self.conv2,
                        self.conv3,
                        self.conv4,
                        self.conv5,
                        self.dense1,
                        self.dense2,
                        self.dropout,
                        self.bn1,
                        self.bn2,
                        self.batchsize
            )
        print(params)

    def load(self):
        print("load csv file")
        

    def conv3_model(self):
        # shape = (seqlength, imgsize, imgsize, channels)
        input_shape = (10, 64, 64, 3)

        model = Sequential()
        # first layer
        model.add(Conv3D(self.conv1, (3,3,3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

        # second layer
        model.add(Conv3D(self.conv2, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

        # 3rd layer
        model.add(Conv3D(self.conv3, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
        if self.bn1:
            model.add(BatchNormalization())
        # 4th layer
        model.add(Conv3D(self.conv4, (2,2,2), activation='relu'))
        if self.conv5:
            model.add(Conv3D(self.conv4 * 2 , (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
        if self.bn2:
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(self.dense1))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.dense2))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def train(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.model.compile(loss='binary_crossentropy',
                            optimizer=self.opt,
                            metrict=[tools.f_score])
        self.model.summary()

    def conv3d_evaluate(self):
        self.train()

        return