import matplotlib.pyplot as plt
from neuronal_network import *
from data import *
from trainer import *

NN = Neural_Network()
T = trainer(NN)


T.train(X,Y)

print T.J
plt.plot(T.J)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
