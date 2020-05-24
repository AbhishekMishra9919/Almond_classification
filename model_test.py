# to test out an initial portion of the project.
import os
import pickle
import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
import glob
# try out a few models, all trained here.
from tensorflow.keras.layers import Input, Dense, Conv2D as c2d, MaxPooling2D as m2d, Flatten, Softmax, AveragePooling2D as a2d, SeparableConvolution2D as sc2d, Lambda
from tensorflow.keras.models import Model, load_model
import sys
# to print the losses into a test file when we are done.
epoch, perc, batch = str(sys.argv)[1:4] # two things given.

def callback_training(model):
    cpf1 = 'best_model_'+str(model)+'.h5'
    mc1 = tf.keras.callbacks.ModelCheckpoint(
      filepath=cpf1,
      save_weights_only=False,
      monitor='val_loss',
      mode='auto',
      save_best_only=False,
      verbose = 1,
      save_freq = 'epoch')
     mc2 = tf.keras.callbacks.CSVLogger(str(model)+'_training')
    return [mc1 ,mc2]

def predict_log2(model,p,(x1,x2),(label,y)):
    f = open(str(model)+'_'+str(p)+'_'+epoch+'_'+perc),w+)
    y_t = model.predict((x1,x2))
    for i in range(0,y_t.shape[0]):
        b = np.argmax(y_t[i])
        if y_t[i] > 0.5 and label[i]==1:
            count_tp[b] = count_tp[b] + 1
        elif y_t[i] > 0.5:
            count_fp[b] = count_fp[b] + 1
        elif label[i]==1:
            count_fn[b] = count_fn[b] + 1
        else:
            count_tn[b] = count_tn[b] + 1
    res = np.array([count_tp, count_tn, count_fp, count_fn])
    for i in range(0,5):
        se = count_tp[i]/(count_tp[i]+count_fn[i])
        sp = count_tn[i]/(count_tn[i]+count_fp[i])
        pr = count_tp[i]/(count_tp[i]+count_fp[i])
        f.write('Class_'+str(i+1))
        f.write('Sensitivity '+str(se))
        f.write('Specificity '+str(sp))
        f.write('Precision '+str(pr))
        f.write('F1_Score '+str(2*(pr*se)/(pr+se)))
        f.write('')
    f.close()

def predict_log(model,p,x,y):
    f = open(str(model)+'_'+str(p)+'_'+epoch+'_'+perc),w+)
    y_t = model.predict(x)
    for i in range(0,y.shape[0]):
    a = np.argmax(y_pred[i])
    b = np.argmax(y[i])
    if a == b:
      count_tp[a] = count_tp[a] + 1
      count_tn = count_tn + 1
      count_tn[a] = count_tn[a] - 1
    else:
      count_fp[a] = count_fp[a] + 1
      count_fn[b] = count_fn[b] + 1
    count_tn = count_tn / 5 # to scale basically.
    res = np.array([count_tp, count_tn, count_fp, count_fn])
    for i in range(0,5):
        se = count_tp[i]/(count_tp[i]+count_fn[i])
        sp = count_tn[i]/(count_tn[i]+count_fp[i])
        pr = count_tp[i]/(count_tp[i]+count_fp[i])
        f.write('Class_'+str(i+1))
        f.write('Sensitivity '+str(se))
        f.write('Specificity '+str(sp))
        f.write('Precision '+str(pr))
        f.write('F1_Score '+str(2*(pr*se)/(pr+se)))
    f.close()
# loading the datasets from the git lfs storage and organizing them, for bilinear and siamese different.
x_tot = np.zeros((0,128,128,3))
x_tot2 = np.zeros((0,128,128,4))
y = np.zeros((1,5))
ptrd = glob.glob('almond_*.pkl') # change if required.
for i in range(len(ptrd)):
    pkl_file = open(ptrd[i],'rb')
    data_1 = pickle.load(pkl_file)
    x1, x2 = data_1['a'], data_1['b']
    y_t = np.repeat(np.array([ry[1]]),repeats = x1.shape[0],axis = 0)
    y = np.concatenate((y,y_t),axis = 0)
    x_tot = np.concatenate((x_tot,x1),axis = 0)
    #x_tot2 = np.concatenate((x_tot2,x2),axis = 0) // for now.

rf = perc.split('-')
t1 = int(x_tot.shape[0]*rf[0]/100)
t2 = int(x_tot.shape[0]*rf[1]/100) + t1

# creation of the new dataset for the things.
x2_tot = np.copy(x2)
y2 = np.copy(y2)
np.random.seed(seed)
np.random.shuffle(x2_tot)
np.random.seed(seed)
np.random.shuffle(y2)

yp = np.ones(y2.shape)
yp = yp * (y2 == y)

# model download from the attached files.
ptrm = glob.glob('model_*.h5') # stored in some folder, we will see.
ptrm.sort()
for p in range(len(ptrm)):
    if p!=1:
        model_t = load_model(ptrm[p])
        model_t.fit(x_tot[0:t1],y[0:t1],validation_data = (x_tot[t1+1:t2],y[t1+1:t2]),epochs = int(epoch),batch_size = int(batch_size),callbacks = callback_training,shuffle = True)
        predict_log(model_t,p+1,x_tot[t2+1:],y[t2+1:])
    else:
        model_t = load_model(ptrm[p])
        model_t.fit((x_tot[0:t1],x2_tot[0:t1]),yp[0:t1],validation_data=((x_tot[t1+1:t2],x2_tot[t1+1:t2]),yp[t1+1:t2]),epochs = int(epoch),batch_size = int(batch_size),callbacks = callback_training,shuffle = True)
        # drawing plots and shit.
        predict_log2(model_t,p+1,(x_tot[t2+1:],x2_tot[t2+1:]),(yp[t2+1],y[t2+1]))
# storing the results, push the result to github, collect the answers and plot.
