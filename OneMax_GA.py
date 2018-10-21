# coding:utf-8
"""
Min-Max問題をGAを用いて解く
サンプルコード

0,1でランダムに初期化されたビット配列をすべて1にする

1.初期設定 
2.評価
3.選択
4.交叉
5.突然変異

"""

import random
from deap import base, creator, tools

# 適応度クラスを作成
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attributeを生成する関数を定義(0,1 のランダムを選ぶ)
toolbox.register("attr_bool", random.randint, 0, 1)
# 個体を生成する関数を定義(Individualクラスでattr_boolの値を100個もつ)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
# 集団を生成する関数を定義(個体をもつリスト)
toolbox.registert("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual)

# 評価関数を登録
toolbox.register("evaluate", evalOneMax)
# 交叉関数を定義(2点交叉)
toolbox.register("mate", tool.cxTwoPoint)
# 変異関数を定義(ビット反転，変異隔離が5%)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# 選択関数を定義(トーナメント選択，tournsizeはトーナメントの数?)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    # 初期集団を生成
    pop = toolbox.population(n=300)
    # 交叉確率，突然変異確率，進化計算のループ回数
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("進化計算")

    # 初期集団の個体を評価する
    fitnesses = list(map(toolbox.evaluate, po))

    for ind, fit in zip(pop, fitnesses): # zipは複数変数の同時ループ
        # 適合性をセットする
        ind.fitness.value = fit

    print(" %i の個体を評価 " % len(pop))

    # 進化計算開始
    for g in range(NGEN):
        print( "-- %i 世代 --" % g)

        """ 選択 """
        # 次世代の個体群を選択
        offspring = toolbox.select(pop, len(pop))
        # 個体群のクローンを生成
        offspring = list(map(toolbox.clone, offspring))

        # 選択した個体群に交叉と突然変異を適応する

        """ 交叉 """
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # 交叉された個体の適応度を削除する
                del child1.fitness.value
                del child2.fitness.value

        """ 変異 """
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.value
        
        # 適応度か計算されてない個体を集めて適応度を計算
        invalid_ind = [ind for ind in offspring if ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitnesses.value = fit
        
        print(" %i の個体を評価 " % len(invalid_ind))

        # 次世代群をoffspringにする
        pop[:] = offspring

        # すべての個体の適応度を配列にする
        fits = [ind.fitness.value(0) for ind in pop]

        length = len(pop)
        mean   = sum(fits) / length
        sum2   = sum(x*x for x in fits)
        std    = abs(sum2/ length - mean**2)**0.5

        print (" Min %s " % min(fits))
        print (" Max %s " % max(fits))
        print (" Avg %s " % mean)
        print (" Std %s " % std)

    print("-- 進化終了 --")

    best_ind = tools.selBest(pop, 1)[0]
    print("最も優れていた個体: %s %s" % (best_ind, best_ind.fitness.values))




