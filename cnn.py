# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# 参考:http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weight_path=None):
        # モデル初期化
        model = Sequential()
        # first set of CONV => RELU => POOL
        model.add(Convolution2D(20, 5, 5, border_mode='same', input_shape=(depth, height, width)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL
        model.add(Convolution2D(50, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        # softmax clasifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        if weight_path is not None:
            model.load_weights(weight_path)
        return model

# def mat_to_np(mat):
#     img = mat.flatten().astype(np.float32)/255.0
#     img = np.asarray(img)
#     img = img.reshape((200, 200))
#     img = img[np.newaxis, np.newaxis, :, :]
#     return img

def get_likelihoods(img_np, model):
    img_cnn = img_np[np.newaxis, np.newaxis, :, :]
    pred = model.predict(img_cnn, batch_size=128, verbose=0)
    likelihoods = pred[0]
    return likelihoods

