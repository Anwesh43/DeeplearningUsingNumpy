import numpy as np
from lib.neural_net_single_layer import NeuralNet
def main():
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    Y = np.array([0,1,1,0])
    nn = NeuralNet(X,Y,5,softmax_required=False)
    nn.train(learning_rate=10e-3)
if __name__ == "__main__":
    main()
