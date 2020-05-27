import glob
import numpy as np

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

np.save('x1.pkl',x1_tot,allow_pickle = True)
np.save('x2.pkl',x2_tot,allow_pickle = True)
np.save('y1.pkl',y,allow_pickle = True)
np.save('y2.pkl',y2,allow_pickle = True)
