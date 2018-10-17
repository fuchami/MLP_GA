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
from keras.layers import Dense, Activation, Input, Flatten
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam

def plot_history(history):
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

def mlp_model(num_classes):
    inputs = Input( shape=( 32*32*3, ))

    x = Dense(128, activation='sigmoid')(inputs)
    x = Dense(64, activation='sigmoid')(x)

    outputs = (Dense(num_classes, activation='softmax'))(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def main(args):
    # load cifar20 datasets
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 32*32*3).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 32*32*3).astype('float32')/ 255.0
    
    y_train = np_utils.to_categorical(y_train, args.numclasses)
    y_test = np_utils.to_categorical(y_test, args.numclasses)
    

    # load model
    model = mlp_model(args.numclasses)
    model.summary()
    plot_model(model, to_file='./images/mlp_model.png', show_shapes=True)
    
    opt = SGD()

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,metrics=['accuracy'])

    history = model.fit(x_train,y_train,
                batch_size=args.batchsize,
                epochs=args.epochs,
                verbose=1,
                validation_data=(x_test,y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss: ', score[0])
    print('test accuracy: ', score[1])
    # plot learning
    plot_history(history)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare model optimize')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--numclasses', '-c', type=int, default=10)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--imgsize', '-s', default=32)
    args = parser.parse_args()

    main(args)