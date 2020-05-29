# vgg and siamese network trained on these datasets.
import pickle
import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.layers import Lambda
import glob
# try out a few models, all trained here.
from tensorflow.keras.layers import Input, Dense, Conv2D as c2d, MaxPooling2D as m2d, Flatten, Softmax, AveragePooling2D as a2d, SeparableConvolution2D as sc2d
from tensorflow.keras.models import Model
data_a = Input(shape = (128,128,3,)) # three dimensional numpy array.
c1 = c2d(12, (7, 7), kernel_initializer='he_uniform', padding='same',activation = 'tanh')(data_a)
m1 = m2d((2, 2))(c1)
c2 = c2d(24, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(m1)
m2 = m2d((2, 2))(c2)
c3 = c2d(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(m2)
m3 = m2d((2, 2))(c3)
f = Flatten()(m3)
d1 = Dense(18, activation='relu', kernel_initializer='he_uniform')(f)
d2 = Dense(5, activation='relu')(d1)
s1 = Softmax(axis = -1)(d2)
model_1 = Model(inputs = data_a,outputs = [s1])
model_1.compile(loss = "categorical_crossentropy",optimizer = "adam")
model_1.save('model_1a.h5')
# recreate all the models and store them in a large enough file.
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
model_p = Model(inputs = data_a,outputs = d1)
ins1 = Input(shape = (128,128,3))
ins2 = Input(shape = (128,128,3))
ls1 = model_p(ins1)
ls2 = model_p(ins2) 
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([ls1, ls2])
model_4 = Model(inputs = [ins1, ins2],outputs = distance)
model_4.compile(optimizer = "adam", loss = contrastive_loss)
model_4.save('model_2a.h5')
data_5 = Input(shape = (128,128,3))
c51 = c2d(20, (5,5), padding = 'valid',activation = 'relu')(data_5)
m51 = m2d((2,2))(c51)
c52 = c2d(32, (5,5), padding = 'valid',strides = 2)(m51) # for the skip connections.
c53 = c2d(16, (5,5), padding = 'valid',activation = 'relu', strides = 2)(m51)
c532 = c2d(16, (3,3), padding = 'same',activation = 'relu')(c53)
c54 = c2d(32, (1,1), padding = 'same')(c532)
cb1 = bn()(c54)
cb2 = bn()(c52)
cadd = Add()([cb1, cb2])
car = rlu()(cadd)
c55 = c2d(32, (5,5), padding = 'valid',strides = 2)(car) # for the skip connections.
c56 = c2d(16, (5,5), padding = 'valid',activation = 'relu', strides = 2)(car)
c57 = c2d(16, (3,3), padding = 'same',activation = 'relu')(c56)
c58 = c2d(32, (1,1), padding = 'same')(c57)
cb3 = bn()(c58)
cb4 = bn()(c55)
cadd2 = Add()([cb3, cb4])
car2 = rlu()(cadd2)
f2 = Flatten()(car2)
fc5 = Dense(24,activation = 'relu')(f2)
e5 = Dense(5,activation = 'relu')(fc5)
sf5 = Softmax(axis = -1)(e5)
model_5 = Model(inputs = data_5,outputs = sf5)
model_5.compile(loss = "categorical_crossentropy",optimizer = "adam")
model_5.save('model_3a.h5')
data_p = Input(shape = (128,128,3,)) # three dimensional numpy array.
t1 = c2d(20, (5, 5), kernel_initializer='he_uniform', padding='valid', activation = 'tanh')(data_p)
p1 = a2d((2, 2))(t1)
t2 = c2d(35, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(p1)
p2 = a2d((2, 2))(t2)
t3 = c2d(50, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(p2)
p3 = a2d((2, 2))(t3)
f2 = Flatten()(p3)
lp1 = Dense(16, activation='relu', kernel_initializer='he_uniform')(f2)
lp2 = Dense(5, activation='relu')(lp1)
s22 = Softmax(axis = -1)(lp2)
model_2 = Model(inputs = data_p,outputs = [s22])
model_2.compile(loss = "categorical_crossentropy",optimizer = "adam")
model_2.save('model_4a.h5')
data_7 = Input(shape = (128,128,3,)) # three dimensional numpy array.
c71 = sc2d(24, (5, 5), padding='same',activation = 'tanh')(data_7)
m71 = m2d((2, 2))(c71)
c72 = sc2d(36, (3, 3), activation='relu', padding='same')(m71)
m72 = m2d((2, 2))(c72)
c73 = sc2d(64, (1, 1), activation='relu', padding='same')(m72)
m73 = m2d((2, 2))(c73)
f7 = Flatten()(m73)
d71 = Dense(16, activation='relu', kernel_initializer='he_uniform')(f7)
d72 = Dense(5, activation='relu')(d71)
s71 = Softmax(axis = -1)(d72)
model_7 = Model(inputs = data_7,outputs = [s71])
model_7.compile(loss = "categorical_crossentropy",optimizer = "adam")
model_7.save('model_5a.h5')
data_8 = Input(shape = (128,128,3,)) # three dimensional numpy array.
c81 = c2d(24, (7, 7), kernel_initializer='he_uniform', padding='same',activation = 'tanh')(data_8)
m81 = m2d((2, 2))(c81)
c82 = c2d(48, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(m81)
m82 = m2d((2, 2))(c82)
c83 = c2d(80, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(m82)
m83 = m2d((4, 4))(c83)
f8 = Flatten()(m83)
d81 = Dense(16, activation='relu', kernel_initializer='he_uniform')(f8)
d82 = Dense(5, activation='relu')(d81)
s81 = Softmax(axis = -1)(d82)
model_p8 = Model(inputs = data_8,outputs = [s81])
# Bilinear CNN.
model_8p1 = Model(inputs = data_8,outputs = m83)
model_8p2 = Model(inputs = data_8,outputs = m83) # 8*8*32
blin1 = Input(shape = (128,128,3))
bim1 = model_8p1(blin1)
bim2 = model_8p2(blin1)
from tensorflow.keras.layers import Reshape
t81 = Reshape((80,64))(bim1)
t82 = Reshape((80,64))(bim2)
bilp = Lambda(outer_pro, output_shape = (80,64,64,))([t81, t82])
opt = Add()(tf.split(bilp,80,axis = 1))
rt = Flatten()(opt)
srt = Dense(20,activation = 'relu')(rt)
fr = Dense(5,activation = 'relu')(srt)
fr2 = Softmax(axis = -1)(fr)
model_8 = Model(inputs = blin1, outputs = fr2)
model_8.compile(loss = "categorical_crossentropy",optimizer = "adam")
model_8.save('model_6a.h5')
