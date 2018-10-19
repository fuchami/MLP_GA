# coding:utf-8
"""
遺伝的アルゴリズムを用いてNNの
ハイパーパラメータの最適解を探索
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import random

import keras
from keras.layers import Activation, Dropout, BatchNormalization, Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from deap import base, creator, tools, algorithms

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

def load_mnist_data():
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

def train():
    early_stopping = EarlyStopping(patience=0, verbose=1)

    x_train, y_train, x_test, y_test = load_mnist_data()
    model = mlp_mnist()

    model.summary()
    model.fit(x_train, y_train,
            batch_size=32,
            epochs=30,
            validation_split=0.2,
            callbacks=[early_stopping])
    
def mnits_evaluate():

    model = train()

    evaluate = model.evaluate(x_test, y_test, batch_size=32, verbose=0)

    return evaluate


# define Genetic Algorithm

#creator
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness= creator.FitnessMax)

# defining attributes for individual
toolbox = base.Toolbox()

# neuron size 
toolbox.register("dense1", random.choice, (64, 128, 256, 512, 1024))
toolbox.register("dense2", random.choice, (64, 128, 256, 512, 1024))

# dropout late
toolbox.register("drop1", random.uniform, 0.0, 0.5)
toolbox.register("drop2", random.uniform, 0.0, 0.5)

# trainig
toolbox.register("batch_size", random.choice, (10, 100, 500))
toolbox.register("epochs", random.choice, (10, 30, 50, 100))
toolbox.register("validation_split", random.uniform, 0.0, 0.6)

# registar attributes to individual
toolbox = register('individual', tools.initCycle, creator.Individual,
                    (toolbox.dense1, toolbox.dense2,
                     toolbox.drop1, toolbox.drop2,
                     toolbox.batch_size, toolbox.epochs, toolbox.validation_split),
                     n = 1)

# individual to pupolation
toolbox.register('population', toolbox.initRepeat, list, toolbox.individual)

# evolution
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.05)
toolbox.register('select', tools.selTournament, tournsizee=3)
toolbox.register('evaluate', run_mnist)


def genAlg(population=5, CXPB=0.5, MUTPB=0.2, NGEN=5):
    random.seed(64)
    pop = toolbox.population(n=population)

    print("start of evolution")

    fitness = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit
    print("  Evaliated %i individuals" % len(pop))

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                print("mate")
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                print("mutate")
                toolbox.mutate(mutant)
                del mutant.fitness.values

        try:
            invalid_ind = [ind for ind in offspring if not ind.fitness.]
            fitness = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness):
                ind.fitnes.values = fit
        except AssertionError:
            pass
        
        print("  Evalutated %i individuals " % len(invalid_ind))

        pop[:] = offspring

        try:
            fits = [ind.fitnes.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits)/length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s", % min(fits))
            print("  Max %s", % max(fits))
            print("  Avg %s", % mean)
            print("  Min %s", % std)
        except IndexError:
            pass
    
    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s %s " % (best_ind, best_ind.fitness.values))
    return best_ind


if __name__ == '__main__':


