import numpy as np
import os
import NN

## XOR training set
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0, 1, 1, 0]]).T

nn = NN.nn((2,3,1))
nn.randomize()
for i in range(10000):
    nn.train(x, y)
    
for i in range(len(x)):
    print(y[i], nn.evaluate(x[i]))




