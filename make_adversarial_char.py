# -*- coding: utf-8 -*-
from deap import base
from deap import creator
from deap import tools
import deap_tools as extools
from cnn import LeNet
from PIL import Image
import numpy as np
import os

class Toolbox(base.Toolbox):
    def __init__(self, src_img_path, model_path='./train_weights.hdf5'):
        super(Toolbox, self).__init__()
        self.src_img_path = src_img_path
        self.model_path = model_path
        self._set()

    def _load_img_as_np(path):
        img_pil = Image.open(path)
        img_np = np.asarray(img_pil)
        img_np.flags.writable = True
        return img_np

    def _eval_char(individual):
        pred = predict.predict_likelihoods(img=individual, model=self.model)
        return

    def _set(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)
        self.register('attr_img', self._load_img_as_np, self.src_img_path)
        self.register('individual', tools.initIterate, creator.Individual, self.attr_img)
        self.register('population', tools.initRepeat, list, self.individual)
        self.model = LeNet.build(width=200, height=200, depth=1, classes=26, weight_path=self.model_path)
        self.register('evaluate', self._eval_char)
        self.register('mate', extools.cxTwoPointImg, indpb=0.05)
        self.register('select', tools.selTournament, tournsize=3)

def make_adversarial_char():
    toolbox = Toolbox(src_img_path='../font_dataset/png_6628_200x200/A/0.png')

    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists('./output/temp'):
        os.mkdir('./output/temp')


if __name__ == '__main__':
    make_adversarial_char()
