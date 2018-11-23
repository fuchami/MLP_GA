# coding: utf-8
"""
GAを使って3Dconv-Netのハイパラ探索
"""

import numpy as np
import argparse
import random
from deap import base, creator, tools, algorithms

#import conv3d

def genAlg(population=5, CXPB=0.5, MUTPB=0.2, NGEN=5):
    random.seed(64)
    pop = toolbox.population(n=population)

    print("start of evolution")

    # 初期集団の個体を評価
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(" %i の個体を評価 " % len(pop))

    # 進化計算
    for g in range(NGEN):
        print(" -- %i 世代 --" % g)

        """ 選択 """
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        """ 交叉 """
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                print("mate")
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        """ 変異 """
        for mutant in offspring:
            if random.random() < MUTPB:
                print("mutate")
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        try:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        except AssertionError:
            pass

        print(" %i の個体を評価" % len(invalid_ind))

        pop[:] = offspring
        try:
            fits = [ind.fitness.values[0] for ind in pop]
            
            length = len(pop)
            mean = sum(fits)/length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print(" Min %s " % min(fits))
            print(" Max %s " % max(fits))
            print(" Avg %s " % mean)
            print(" Std %s " % std)
        except IndexError:
            pass

    print("-- 進化終了 --")

    best_ind = tools.selBest(pop, 1)[0]
    print(" 最も優れていた個体 %s %s " % (best_ind, best_ind.fitness.values))
    return best_ind


def run_conv3d():

def main():

if __name__ == "__main__":
    main()
