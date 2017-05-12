# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import random
import argparse
import cnn
from deap import base
from deap import creator
from deap import tools
import deap_tools as extools

class Toolbox(base.Toolbox):
    def __init__(self, src_img_path, dst_alph, model_path='./train_weights.hdf5'):
        super(Toolbox, self).__init__()
        self.src_img_path = src_img_path
        self.dst_alph = dst_alph
        self.model_path = model_path
        self._set()

    def _load_img_as_np(self, path):
        img_pil = Image.open(path)
        img_np = np.asarray(img_pil)
        img_np.flags.writeable = True
        return img_np

    def _eval_char(self, individual):
        pred = cnn.get_likelihoods(img_np=individual, model=self.model)
        dst_alph_num = ord(self.dst_alph) - 65
        return pred[dst_alph_num],

    def _set(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)
        self.register('attr_img', self._load_img_as_np, self.src_img_path)
        self.register('individual', tools.initIterate, creator.Individual, self.attr_img)
        self.register('population', tools.initRepeat, list, self.individual)
        self.model = cnn.LeNet.build(width=200, height=200, depth=1, classes=26, weight_path=self.model_path)
        self.register('evaluate', self._eval_char)
        self.register('mate', extools.cxTwoPointImg)
        self.register('mutate', extools.mutFlipBitImg, indpb=0.05)
        self.register('select', tools.selTournament, tournsize=3)

class AdversarialCharacter():
    def __init__(self, src_img_path, dst_alph, dst_path,
                 cxpb, mutpb, ngen, npop, breakacc):
        self.toolbox = Toolbox(src_img_path=src_img_path, dst_alph=dst_alph)
        self.dst_root_path = dst_path
        self.dst_best_path = os.path.join(self.dst_root_path, 'best')
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.npop = npop
        self.breakacc = breakacc
        self._make_dst_dir()

    def _make_dst_dir(self):
        if not os.path.exists(self.dst_root_path):
            os.mkdir(self.dst_best_path)
        if not os.path.exists(os.path.join(self.dst_best_path, 'best')):
            os.mkdir(os.path.join(self.dst_best_path, 'best'))

    def _save_img(self, filename, img_np):
        img_pil = Image.fromarray(np.uint8(img_np))
        img_pil.save(os.path.join(self.dst_best_path, filename), 'PNG')

    def train(self):
        # 初期集団を生成
        self.pop = self.toolbox.population(n=self.npop)

        # 初期集団の個体を評価
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.ngen):
            print ('{0}世代'.format(g))

            # 選択
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 変異
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 適合度が計算されていない個体を集めて適合度を計算
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 次世代群をoffspringにする
            self.pop[:] = offspring

            # すべての適合度を配列にする
            fits = [ind.fitness.values[0] for ind in self.pop]

            print ('Min: {0:013.10f} %'.format(min(fits) * 100))
            print ('Max: {0:013.10f} %'.format(max(fits) * 100))

            best_ind_np = tools.selBest(self.pop, 1)[0]
            self._save_img(str(g) + '.png', best_ind_np)

            if max(fits) >= self.breakacc:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make adversarial character')
    parser.add_argument('src_img_path', type=str,
                        help='source img path')
    parser.add_argument('dst_alph', type=str,
                        help='destination character')
    parser.add_argument('-d', '--dst_path', dest='dst_path', type=str, default='output',
                        help='destination path')
    parser.add_argument('--cxpb', dest='cxpb', type=float, default=0.5,
                        help='crossover probability')
    parser.add_argument('--mutpb', dest='mutpb', type=float, default=0.2,
                        help='mutation probability')
    parser.add_argument('--ngen', dest='ngen', type=int, default=1000,
                        help='num of generation')
    parser.add_argument('--npop', dest='npop', type=int, default=300,
                        help='num of population')
    parser.add_argument('--breakacc', dest='breakacc', type=float, default=0.99,
                        help='accuracy of break')
    args = parser.parse_args()
    ac = AdversarialCharacter(src_img_path=args.src_img_path, dst_alph=args.dst_alph,
                              dst_path=args.dst_path, cxpb=args.cxpb, mutpb=args.mutpb,
                              ngen=args.ngen, npop=args.npop, breakacc=args.breakacc)
    ac.train()

