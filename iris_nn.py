import sklearn.datasets as datasets
from lib.util import *
def derivative_W2(T,Y,Z):
    return Z.T.dot(T-Y)
def derivative_b2(T,Y):
    return (T-Y).sum(axis=0)
def derivative_W1(T,Y,Z,Xd,W2):
    return Xd.T.dot((T-Y).dot(W2.T)*Z*(1-Z))

def derivative_b1(T,Y,Z,W2):
    return ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)
def forward(X,W1,W2,b1,b2):
    Z = tanh(X.dot(W1)+b1)
    Y = softmax(Z.dot(W2)+b2)
    return (Z,Y)

iris = datasets.load_iris()
data =  iris.data
target = iris.target
T = getIndicator(target)
X = data
N1,D = X.shape
N,K = T.shape

M = 8
W1,b1 = initWeights(D,M)
W2,b2 = initWeights(M,K)
learning_rate = 1.3*10e-6
for epoch in xrange(100000):
    Z,Y = forward(X,W1,W2,b1,b2)
    W2+=learning_rate*derivative_W2(T,Y,Z)
    b2+=learning_rate*derivative_b2(T,Y)
    W1+=learning_rate*derivative_W1(T,T,Z,X,W2)
    b1+=learning_rate*derivative_b1(T,Y,Z,W2)
    accuracy = 1-error_rate(target,Y)
    if epoch%1000 == 0:
        print accuracy

def predict(X1):
    Z,Y = forward(X1,W1,W2,b1,b2)
    return np.argmax(Y)
x_index = int(raw_input())
print predict(X[x_index])
print target[x_index]
