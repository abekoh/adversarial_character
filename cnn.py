# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# 参考:http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
class LeNet:
    def __init__(self, width, height, depth, classes, weight_path=None):
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        self.weight_path = weight_path
        self._build()

    def _build(self):
        # モデル初期化
        self.model = Sequential()
        # first set of CONV => RELU => POOL
        self.model.add(Convolution2D(20, 5, 5, border_mode='same', input_shape=(self.depth, self.height, self.width)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL
        self.model.add(Convolution2D(50, 5, 5, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # set of FC => RELU layers
        self.model.add(Flatten())
        self.model.add(Dense(500))
        self.model.add(Activation('relu'))
        # softmax clasifier
        self.model.add(Dense(self.classes))
        self.model.add(Activation('softmax'))
        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)

    def get_likelihoods(self, img_np):
        img_cnn = img_np.copy()
        img_cnn = img_cnn.astype(np.float32) / 255.0
        img_cnn = img_cnn[np.newaxis, np.newaxis, :, :]
        pred = self.model.predict(img_cnn, batch_size=128, verbose=0)
        likelihoods = pred[0]
        return likelihoods

    def eval_char_with_before(self, individual, src_alph, dst_alph):
        pred = self.get_likelihoods(img_np=individual)
        src_alph_num = ord(src_alph) - 65
        dst_alph_num = ord(dst_alph) - 65
        return pred[src_alph_num], pred[dst_alph_num]

