import numpy as np
def initWeights(M1,M2):
    W = np.random.randn(M1,M2)
    b = np.random.randn(M2)
    return (np.float32(W),np.float32(b))


def getIndicator(Y):
    N = Y.shape[0]
    K = len(set(Y))
    ind = np.zeros((N,K))
    for i in range(0,N):
        ind[i,Y[i]] = 1
    return ind


def cost(T,Y):
    return -(T*np.log(Y)).sum()

def signmoid_cost(T,Y):
    return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()

def relu(x):
    return x*(x>0)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    if len(x.shape) == 1:
        return np.exp(x)/np.exp(x).sum()
    return np.exp(x)/np.exp(x).sum(axis=1,keepdims=True)

def error_rate(T,Y):
    return 1-np.mean(np.argmax(Y,axis=1) == T)
