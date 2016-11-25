from lib.util import *
import numpy as np
import time
(W,b) = initWeights(3,4)
print "printing weights"
time.sleep(1)
print W
print "printing biases"
time.sleep(1)
print b
Y = np.array([1,0,2,1,0,0,0,3,2,1,1,1,1,3,0,2,2,1,3,3])
print "printing inidicator matrix"
time.sleep(1)
print getIndicator(Y)
print "print relu"
time.sleep(1)
print relu(np.array([-3,-4,1,2,5,-2]))
print "sodtmax"
time.sleep(1)
print softmax(np.array([2,3,1,0,5,3]))
time.sleep(1)
print softmax(np.random.randn(3,2))
