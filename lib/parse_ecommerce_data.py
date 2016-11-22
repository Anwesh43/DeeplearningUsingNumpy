import pandas as pd
import numpy as np
def getData():
    data = pd.read_csv('ecommerce_data.csv')
    XY = data.as_matrix()
    Y_ = XY[:,-1]
    X_ = XY[:,:-1]
    Xr = X_[:,-1]
    r,c = X_.shape
    X = np.hstack([X_[:,:-1],np.zeros([r,4])])
    c = c-1
    Y = np.zeros([r,4])
    for i in range(0,r):
        X[i,c+Xr[i]] = 1
        Y[i,Y_[i]] = 1
    return (X,Y)
