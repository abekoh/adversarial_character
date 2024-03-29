# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import imageio
import glob
import sys

from deap import base
from deap import creator
from deap import tools
import deap_tools as extools

class Toolbox(base.Toolbox):
    def __init__(self, src_img_path, dst_alph, model):
        super(Toolbox, self).__init__()
        self.src_img_path = src_img_path
        self.dst_alph = dst_alph
        self.model = model
        self._set()

    def _load_img_as_np(self, path):
        img_pil = Image.open(path)
        if img_pil.size != (200, 200):
            img_pil = img_pil.resize((200, 200))
        img_np = np.asarray(img_pil)
        img_np.flags.writeable = True
        return img_np

    def eval_char(self, individual):
        pred = self.model.get_likelihoods(img_np=individual)
        dst_alph_num = ord(self.dst_alph) - 65
        return pred[dst_alph_num],

    def _set(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)
        self.register('attr_img', self._load_img_as_np, self.src_img_path)
        self.register('individual', tools.initIterate, creator.Individual, self.attr_img)
        self.register('population', tools.initRepeat, list, self.individual)
        self.register('evaluate', self.eval_char)
        self.register('mate', extools.cxTwoPointImg)
        self.register('mutate', extools.mutFlipBitImg, indpb=0.05)
        self.register('select', tools.selTournament, tournsize=3)

class AdversarialCharacter():
    def __init__(self, src_img_path, src_alph, dst_alph, dst_path,
                 cxpb, mutpb, ngen, npop, breakacc, model):
        self.src_alph = src_alph
        self.dst_alph = dst_alph
        self.dst_root_path = dst_path
        self.dst_best_path = os.path.join(self.dst_root_path, 'best')
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.npop = npop
        self.breakacc = breakacc
        self.accuracies = []
        self.model = model
        self.toolbox = Toolbox(src_img_path=src_img_path, dst_alph=dst_alph, model=model)
        self._make_dst_dir()

    def _make_dst_dir(self):
        if not os.path.exists(self.dst_root_path):
            os.mkdir(self.dst_root_path)
        if not os.path.exists(self.dst_best_path):
            os.mkdir(self.dst_best_path)

    def _save_img(self, filename, img_np):
        img_pil = Image.fromarray(np.uint8(img_np))
        img_pil.save(os.path.join(self.dst_best_path, filename), 'PNG')

    def _log_accuracies(self, best_ind_np):
        src_alph_score, dst_alph_score = self.model.eval_char_with_before(best_ind_np, self.src_alph, self.dst_alph)
        self.accuracies.append((src_alph_score, dst_alph_score))

    def _os_font_path(self):
        if sys.platform == 'darwin':
            return '/Library/Fonts/Arial.ttf'
        if sys.platform == 'linux':
            return '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
        if sys.platform == 'win32':
            return 'C:\Windows\Fonts\arial.ttf'

    def train(self):
        # 初期集団を生成
        self.pop = self.toolbox.population(n=self.npop)

        # 初期集団の個体を評価
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.ngen):
            print ('{0}世代'.format(g + 1))

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

            print ('Max: {0:013.10f} %'.format(max(fits) * 100))
            print ('Min: {0:013.10f} %'.format(min(fits) * 100))

            best_ind_np = tools.selBest(self.pop, 1)[0]
            self._save_img(str(g) + '.png', best_ind_np)
            self._log_accuracies(best_ind_np)

            self.finish_g = g + 1
            if max(fits) >= self.breakacc:
                break

    def save_animation(self, is_acc=True):
        imgs = []
        for i in range(self.finish_g):
            path = os.path.join(self.dst_best_path, '{0}.png'.format(i))
            img_char = Image.open(path)
            if is_acc:
                img = Image.new('RGB', (img_char.size[0], img_char.size[1] + 14), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(self._os_font_path(), 14)
                src_acc_text = '{0}:{1:06.2f}%'.format(self.src_alph, self.accuracies[i][0] * 100)
                dst_acc_text = '{0}:{1:06.2f}%'.format(self.dst_alph, self.accuracies[i][1] * 100)
                iter_text = '{0:03d}/{1:03d}'.format(i + 1, self.finish_g)
                draw.text((0, img_char.size[1]), iter_text, font=font, fill='#000000')
                draw.text((58, img_char.size[1]), src_acc_text, font=font, fill='#000000')
                draw.text((130, img_char.size[1]), dst_acc_text, font=font, fill='#000000')
                img.paste(img_char, (0, 0))
            img_np = np.array(img)
            imgs.append(img_np)
        imageio.mimsave(os.path.join(self.dst_root_path, 'output.gif'), imgs, duration=0.5)
        print('save gif animation')

    def save_log(self):
        with open(os.path.join(self.dst_root_path, 'log.csv'), 'w') as log_file:
            log_file.write('{0},{1}\n'.format(self.src_alph, self.dst_alph))
            for i in range(self.finish_g):
                log_file.write('{0:012.10f},{1:012.10f}\n'.format(self.accuracies[i][0], self.accuracies[i][1]))
        print('save log')

