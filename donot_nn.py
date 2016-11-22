import numpy as np
from lib.neural_net_single_layer import NeuralNet
import matplotlib.pyplot as plt
def main():
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    plt.show()
    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))
    plt.scatter(X_inner[:,0],X_inner[:,1])
    plt.scatter(X_outer[:,0],X_outer[:,1])
    plt.show()
    nn = NeuralNet(X,Y,M=3,softmax_required=False)
    nn.train(learning_rate=0.00005,iterations=16000,regularization=0.0002)
if __name__ == "__main__":
    main()
