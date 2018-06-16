# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 00:02:46 2018

@author: zhaow
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


def keras_learn(X, Y):
    np.random.seed(0)
    """モデルの設定"""
    model = Sequential()
    model.add(Dense(input_dim=2, units=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

    model.fit(X, Y, epochs=200, batch_size=1)

    """学習結果の確認"""
    classes = model.predict_classes(X, batch_size=1)
    prob = model.predict_proba(X, batch_size=1)

    print('classifed')
    print(Y == classes)
    print()
    print('output probability')
    print(prob)


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [1]])
    keras_learn(X, Y)
