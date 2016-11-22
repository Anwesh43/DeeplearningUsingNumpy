from lib.parse_ecommerce_data import getData
from lib.neural_net_single_layer import NeuralNet
import time
X,Y = getData()
N = X.shape[0]
bp = NeuralNet(X[0:N/2,:],Y[0:N/2,:])
bp.train()
time.sleep(1)
print "lets predict a value now"
time.sleep(1)
print bp.predict(X[N/2+1:N/2+3,:])
print (Y[N/2+1:N/2+3,:])
