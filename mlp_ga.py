# coding:utf-8
"""
遺伝的アルゴリズムを用いてNNの
ハイパーパラメータの最適解を探索

CXPB: クロスオーバーの確率
MUTPB: 個々の突然変異の確率
NGEN : 生成ループの数

individual: 設計変数の1セット
population: 現世代の個体集合
offspring : 次世代の個体集合
fitness   : 適応度
selection : 現世代から次世代への淘汰
crossover : 交叉・2個体間の遺伝子の入れ替え
mutation  :

1.初期設定 
2.評価
3.選択
4.交叉
5.突然変異
突然変異

"""

import numpy as np
import argparse
import random
from deap import base, creator, tools, algorithms

import mlp

def genAlg(population=5, CXPB=0.5, MUTPB=0.2, NGEN=5):
    """
    CXPB: 交叉確率
    MUTPB: 突然変異確率
    NGEN: 進化計算のループ回数
    """
    random.seed(64)
    pop = toolbox.population(n=population)

    print("start of evolution")

    # 初期集団の個体を評価する
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        # 適合性をセットする
        ind.fitness.values = fit

    print(" %i の個体を評価 " % len(pop))

    # 進化計算開始
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
                # 交叉された個体の適応度を削除する
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
        
        print(" %i の個体を評価 " % len(invalid_ind))

        # 次世代群をoffspringにする
        pop[:] = offspring

        # すべての個体の適応度をを配列にする
        try:
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits)/length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
        except IndexError:
            pass
    
    print("-- 進化終了 -- ")

    best_ind = tools.selBest(pop, 1)[0]
    print(" 最も優れていた個体 %s %s " % (best_ind, best_ind.fitness.values))
    return best_ind

def run_mlp(bounds):
    _mlp = mlp.MLP(dense1=bounds[0],
                    dense2=bounds[1],
                    drop1=bounds[2],
                    drop2=bounds[3],
                    batch_size=bounds[4],
                    activation=bounds[5],
                    optimizer=bounds[6]
                    )

    mnist_evaluation = _mlp.mlp_evaluate()
    print(mnist_evaluation)
    
    return mnist_evaluation[0],
    

""" define Genetic Algorithm """

# 適応度クラスを作成
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness= creator.FitnessMax)

# defining attributes for individual
toolbox = base.Toolbox()

# 各パラメータを生成する関数を定義
# neuron size 
toolbox.register("dense1", random.choice, (32, 64, 128, 256, 512, 1024))
toolbox.register("dense2", random.choice, (32, 64, 128, 256, 512, 1024))

# dropout late
toolbox.register("drop1", random.uniform, 0.0, 0.5)
toolbox.register("drop2", random.uniform, 0.0, 0.5)

# trainig
toolbox.register("batch_size", random.choice, (16, 32, 64, 128, 256, 512))
toolbox.register("activation", random.choice, ('sigmoid', 'relu'))
toolbox.register("optimizer", random.choice, ('SGD1', 'SGD2', 'Adam'))

# registar attributes to individual
toolbox.register('individual', tools.initCycle, creator.Individual,
                    (toolbox.dense1, toolbox.dense2,
                    toolbox.drop1, toolbox.drop2,
                    toolbox.batch_size, toolbox.activation, toolbox.validation_split),
                    n = 1)

# individual to pupolation
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# 交叉関数を定義(2点交叉)
toolbox.register('mate', tools.cxTwoPoint)
# 変異関数を定義
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.05)
# 選択関数を定義
toolbox.register('select', tools.selTournament, tournsize=3)
# MLPを学習して評価関数を登録
toolbox.register('evaluate', run_mlp)

best_int = genAlg(population=5, CXPB=0.5, MUTPB=0.2, NGEN=10 )


