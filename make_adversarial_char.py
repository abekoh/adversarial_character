# -*- coding: utf-8 -*-
from deap import base
from deap import creator
from deap import tools
import deap_tools as extools
from PIL import Image
import cnn
import numpy as np
import os
import random
# creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

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
        # img_np.flags.writeable = True
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

def make_adversarial_char():
    toolbox = Toolbox(src_img_path='../font_dataset/png_6628_200x200/A/0.png', dst_alph='B')

    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists('./output/temp'):
        os.mkdir('./output/temp')

    pop = toolbox.population(n=30)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 1000

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.value = fit

    for g in range(NGEN):
        print ('{0}世代'.format(g))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        print ('Min: {0:013.10f} %'.format(min(fits) * 100))
        print ('Max: {0:013.10f} %'.format(max(fits) * 100))

        if max(fits) >= 0.99:
            break


if __name__ == '__main__':
    make_adversarial_char()
