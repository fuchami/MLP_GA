# coding:utf-8

import keras 
from keras.layers import Activation,Dropout,BatchNormalization,Dense,Input,Flatten
from keras.layers import Conv3D, MaxPool3D, MaxPooling3D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
import matplotlib as plt

class Conv3DNet():
    def __init__(self):

        """ load data """
        print("load data")

        """ build 3dconv-net model """

        #params = """
    

    def load(self):

    def conv3_model(self):
        # shape = (seqlength, imgsize, imgsize, channels)
        input_shape = (args.seqlength, args.imgsize, args.imgsize, 3)

        model = Sequential()
        # first layer
        model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

        # second layer
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

        # 3rd layer
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
        model.add(BatchNormalization())
        # 4th layer
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(args.dropout))
        model.add(Dense(128))
        model.add(Dropout(args.dropout))
        model.add(Dense(1, activation='sigmoid'))

    return model

    def train(self):

    def conv3d_evaluate(self):