# -*- coding: utf-8 -*-
import random

# deap.tools.cxTwoPointの画像用に改造版
# 座標cxpoint1を左上，座標cxpoint2を右下の頂点とする
# 四角形部分を交叉させる
def cxTwoPointImg(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = [random.randint(1, size), random.randint(1, size)]
    cxpoint2 = [random.randint(1, size - 1), random.randint(1, size - 1)]

    if cxpoint2[0] >= cxpoint1[0]:
        cxpoint2[0] += 1
    else:
        cxpoint1[0], cxpoint2[0] = cxpoint2[0], cxpoint1[0]

    if cxpoint2[1] >= cxpoint1[1]:
        cxpoint2[1] += 1
    else:
        cxpoint1[1], cxpoint2[1] = cxpoint2[1], cxpoint1[1]

    for i in range(cxpoint1[1], cxpoint2[1]):
        for j in range(cxpoint1[0], cxpoint2[0]):
            ind1[i][j], ind2[i][j] = ind2[i][j], ind1[i][j]

    return ind1, ind2

# deap.tools.mutFlipBitの画像用に改造版
# あるピクセルについて，色を反転させる
def mutFlipBitImg(individual, indpb):
    for i in range(len(individual)):
        for j in range(len(individual[0])):
            if random.random() < indpb:
                if individual[i][j] == 255:
                    individual[i][j] = 0
                else:
                    individual[i][j] = 255
    return individual,

