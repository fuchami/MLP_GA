# coding:utf-8

"""
Cifar10でSGDとCMA-ESの比較

"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam

def mlp_model(num_classes,img_size):
    inputs = Input( shape=(img_size*img_size))

    x = Dense(128, activatoin='relu')(inputs)
    x = Dense(64, activation='relu')(x)

    outputs = (Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def main(args):
    # load cifar20 datasets
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()

    # load model
    model = mlp_model(args.numclasses, args.imgsize)
    model.summary()
    plot_model(model, to_file='./images/mlp_model.png', show_shapes=True)
    
    opt = SGD()

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,metrics=['accuracy'])

    model.fit(x_train,x_test,
            batch_size=args.batchsize,
            nb_epoch=args.epochs,
            verbose=1,
            validation_data=(x_test,y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss: ', score[0])
    print('test accuracy: ', score[1])
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare model optimize')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--numclasses', '-c', type=int, default=10)
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--imgsize', '-s', type=int, default=32)
    args = parser.parse_args()

    main(args)