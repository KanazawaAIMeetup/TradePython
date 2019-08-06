#coding: utf-8
'''
MITライセンス　このプログラムについては、改変・再配布可能です
著作者： Tomohiro Ueno (kanazawaaimeetup@gmail.com)

Kerasで複数入力を受け付ける全結合層を用いたプログラムのサンプル
'''
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

def _model():
    state_size = window_size
    output_size = 1
    shp = np.array([0 for i in range(10)])
    input1 = keras.layers.Input(shape=(1, state_size), name="in1")
    x1 = keras.layers.Dense(30, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(1, state_size), name="in2")
    x2 = keras.layers.Dense(30, activation='relu')(input2)
    input3 = keras.layers.Input(shape=(1, state_size), name="in3")
    x3 = keras.layers.Dense(30, activation='relu')(input3)
    input4 = keras.layers.Input(shape=(1, state_size), name="in4")
    x4 = keras.layers.Dense(30, activation='relu')(input4)
    input5 = keras.layers.Input(shape=(1, state_size), name="in5")
    x5 = keras.layers.Dense(30, activation='relu')(input5)
    input6 = keras.layers.Input(shape=(1, state_size), name="in6")
    x6 = keras.layers.Dense(30, activation='relu')(input6)
    input7 = keras.layers.Input(shape=(1, state_size), name="in7")
    x7 = keras.layers.Dense(30, activation='relu')(input7)
    added = keras.layers.Add()([x1, x2, x3, x4, x5, x6, x7])  # equivalent to added = keras.layers.add([x1, x2])
    dense_added = keras.layers.Dense(150)(added)
    fl = Flatten()(dense_added)
    out = keras.layers.Dense(output_size, activation="tanh", name="output_Q")(fl)
    model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=[out])

    model.compile(loss={'output_Q': 'mean_squared_error'},
                  loss_weights={'output_Q': 1},
                  optimizer=Adam(lr=0.001))
    return model


model = _model()