# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import np_utils

class LeNet(Sequential):
    def __init__(self, width, height, depth, classes, weight_path=None):
        super(LeNet, self).__init__()
        self.add(Convolution2D(20, 5, 5, border_mode='same', input_shape=(height, width, depth)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Convolution2D(50, 5, 5, border_mode='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Flatten())
        self.add(Dense(500))
        self.add(Activation('relu'))
        self.add(Dense(classes))
        self.add(Activation('softmax'))
        if weight_path is not None:
            self.load_weights(weight_path)

    def get_likelihoods(self, img_np):
        img_cnn = img_np.copy()
        img_cnn = img_cnn.astype(np.float32) / 255.0
        img_cnn = img_cnn[np.newaxis, :, :, np.newaxis]
        pred = self.predict(img_cnn, batch_size=128, verbose=0)
        likelihoods = pred[0]
        return likelihoods

    def eval_char_with_before(self, individual, src_alph, dst_alph):
        pred = self.get_likelihoods(img_np=individual)
        src_alph_num = ord(src_alph) - 65
        dst_alph_num = ord(dst_alph) - 65
        return pred[src_alph_num], pred[dst_alph_num]

    def _filelist_to_list(self, filelist_path):
        imgs, labels = [], []
        with open(filelist_path, 'r') as f:
            max_count = sum(1 for line in f)
        with open(filelist_path, 'r') as f:
            for count, readline in enumerate(f):
                if count % 1000 == 0:
                    print('making list... ({0}/{1})'.format(count, max_count))
                readline = readline.rstrip()
                readline_sp = readline.split(',')
                img_pil = Image.open(readline_sp[0])
                img_np = np.asarray(img_pil)
                img_np = img_np.astype(np.float32) / 255.0
                imgs.append(img_np)
                labels.append(int(readline_sp[1]))
        print('reshaping images...')
        imgs = np.asarray(imgs)
        labels = np.asarray(labels)
        imgs = imgs.reshape((imgs.shape[0], 200, 200))
        imgs = imgs[:, :, :, np.newaxis]
        labels = np_utils.to_categorical(labels, 26)
        return imgs, labels

    def train(self, src_train_path, src_test_path=None, dst_hdf5_path='train_weight.hdf5', batch_size=128, nb_epoch=2):
        opt = SGD(lr=0.01)
        train_imgs, train_labels = self._filelist_to_list(src_train_path)
        self.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print('training...')
        self.fit(train_imgs, train_labels, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
        if src_test_path:
            print('testing...')
            test_imgs, test_labels = self._filelist_to_list(src_test_path)
            (loss, accuracy) = self.evaluate(test_imgs, test_labels, batch_size=batch_size, verbose=1)
            print('accuracy: {:.2f}%'.format(accuracy * 100))
        self.save_weights(dst_hdf5_path, overwrite=True)
        print('saved ' + dst_hdf5_path)

