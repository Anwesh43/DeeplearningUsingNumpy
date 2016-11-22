import numpy as np
import matplotlib.pyplot as plt
class FeedForwardNN:
    def __init__(self,n):
        X1 = np.random.randn(n,2)+np.array([0,-2])
        X2 = np.random.randn(n,2)+np.array([2,2])
        X3 = np.random.randn(n,2)+np.array([-2,2])
        self.X = np.vstack([X1,X2,X3])
        self.N = n
        self.Y = np.array([0]*500+[1]*500+[2]*500)
        self.M = 3
        self.D = 2
        self.K = 3
        self.W1 = np.random.randn(self.D,self.M)
        self.b1 = np.random.randn(self.M)
        self.W2 = np.random.randn(self.M,self.K)
        self.b2 = np.random.randn(self.K)

    def train(self):
        Y1 = self.X.dot(self.W1)+self.b1
        Z = 1/(1+np.exp(-Y1))
        Y2 = Z.dot(self.W2)+self.b2
        expY2  = np.exp(Y2)
        OUTPUT = expY2/expY2.sum(axis=1,keepdims=True)
        self.output = OUTPUT
        self.Z = Z
        self.PRED_Y = np.argmax(OUTPUT,axis=1)

    def test(self):
        correct = 0
        i = 0
        for y in self.Y:
            if(self.PRED_Y[i] == y):
                correct = correct+1
            i = i+1
        print 'Accuracy is {0}'.format((correct*1.0)/i)
    def plot_input(self):
        plt.scatter(self.X[:,0],self.X[:,1])
        plt.show()
