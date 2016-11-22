from ecomm_project import getData
import numpy as np
import time
class NeuralNet:
    def __init__(self,X,Y,M=3,softmax_required=True):
        assert(X.shape[0] == Y.shape[0])
        self.X = X
        self.T = Y
        self.softmax_required = softmax_required
        self.D = X.shape[1]
        self.M = M
        self.K = 1
        if not(len(Y.shape) == 1):
            self.K = Y.shape[1]
        self.W1 = np.random.randn(self.D,self.M)
        self.W2 = np.random.randn(self.M,self.K)
        if(len(self.T.shape) == 1):
            self.W2 = np.random.randn(self.M)
        self.b1 = np.random.randn(self.M)
        self.b2 = np.random.randn(self.K)
    def __cost(self):
        if len(self.T.shape) == 1:
            return np.sum(self.T*np.log(self.output)+(1-self.T)*(np.log(1-self.output)))
        return (self.T*np.log(self.output)).sum()
    def __derivative_b2(self):
        if(len(self.T.shape) == 1):
            return (self.T-self.output).sum()
        return (self.T-self.output).sum(axis=0)
    def __derivative_w2(self):
        if(len(self.T.shape) == 1):
            return (self.T-self.output).dot(self.Z)
        return self.Z.T.dot(self.T-self.output)
    def __derivative_w1(self):
        if(len(self.T.shape) == 1):
            front = np.outer(self.T-self.output,self.W2)*self.Z*(1-self.Z)
            return front.T.dot(self.X).T
        return self.X.T.dot(((self.T-self.output).dot(self.W2.T))*self.Z*(1-self.Z))
    def __derivative_b1(self):
        if(len(self.T.shape) == 1):
            front = np.outer(self.T-self.output,self.W2)*self.Z*(1-self.Z)
            return front.sum(axis=0)
        return (((self.T-self.output).dot(self.W2.T))*self.Z*(1-self.Z)).sum(axis=0)
    def __feedforward(self,X):
        self.Z = 1/(1+np.exp(-(X.dot(self.W1)+self.b1)))
        self.output = self.Z.dot(self.W2)+self.b2
    def __softmax(self):
        if(self.softmax_required):
            expA = np.exp(self.output)
            if(len(self.T.shape) == 1):
                self.output = expA/expA.sum()
                self.Y = np.argmax(self.output)
            else:
                self.output = expA/(expA.sum(axis=1,keepdims=True))
                self.Y = np.argmax(self.output,axis=1)
        else:
            self.output = 1/(1+np.exp(-self.output))
            self.Y = self.output

    def __test(self):
        if len(self.T.shape) == 1:
            return np.mean(self.T == np.round(self.Y))
        return np.mean(np.argmax(self.T,axis=1) == self.Y)
    def train(self,learning_rate=10e-4,iterations=100000,regularization=0):
        for epoch in xrange(iterations):
            self.__feedforward(self.X)
            self.__softmax()
            if epoch%100 == 0:
                print self.__cost()
                print self.__test()
            self.W2+=learning_rate*self.__derivative_w2()-regularization*self.W2
            self.b2+=learning_rate*self.__derivative_b2()-regularization*self.b2
            self.W1+=learning_rate*self.__derivative_w1()-regularization*self.W1
            self.b1+=learning_rate*self.__derivative_b1()-regularization*self.b1
    def predict(self,X):
        self.__feedforward(X)
        self.__softmax()
        return self.output

X,Y = getData()
N = X.shape[0]
