import numpy as np
from feed_forward import FeedForwardNN

def cost(T,Y):
    return (T*(np.log(Y))).sum()
def derivative_W2(T,Y,Z):
    return Z.T.dot(T-Y)
def derivative_b2(T,Y):
    return (T-Y).sum(axis=0)
def derivative_W1(T,Y,Z,W2,X):
    return  (X.T).dot(((T-Y).dot(W2.T)*Z*(1-Z)))

def derivative_b1(T,Y,W,Z):
    return (((T-Y).dot(W.T)*Z*(1-Z))).sum(axis=0)
def main():
    ff = FeedForwardNN(500)
    ff.train()
    output = ff.output
    n, = ff.Y.shape
    T = np.zeros([n,ff.K])
    learning_rate = 10e-7
    print learning_rate
    for i in range(ff.N):
        T[i,ff.Y[i]] = 1
    for epoch in range(1000000):
        ff.train()
        output = ff.output
        if epoch%100 == 0:
            print cost(T,output)
            print classification_rate(ff.Y,np.argmax(output,axis=1))
        ff.W2+=learning_rate*derivative_W2(T,output,ff.Z)
        ff.b2+=learning_rate*derivative_b2(T,output)
        ff.W1+=learning_rate*derivative_W1(T,output,ff.Z,ff.W2,ff.X)
        ff.b1+=learning_rate*derivative_b1(T,output,ff.W2,ff.Z)

def classification_rate(Y,P):
    total = len(Y)
    correct = 0
    for i in range(0,len(Y)):
        if(Y[i] == P[i]):
            correct+=1
    return (correct*1.0)/total
def createSingleLayerXWB(N,D,M,K,Xis):
    Xs = []
    for i in range(0,len(Xis)):
        Xs.append(np.random.randn(N,D)+Xis[i])
    X = np.vstack(Xs)
    print X
    W = [np.random.randn(D,M),np.random.randn(M,K)]
    B = [np.random.randn(M),np.random.randn(K)]
    return (X,W,B)
if __name__ == "__main__":
    main()
