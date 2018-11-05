import matplotlib.pyplot as plt
from neuronal_network import *
from data import *

NN = Neural_Network()
yHat = NN.forward(X)
#print yHat
#print Y

cost1 = NN.costFunction(X,Y)
dJdW1,dJdW2 = NN.costFunctionPrime(X,Y)

print dJdW1
print dJdW2

scalar = 3
NN.W1 = NN.W1 + scalar*dJdW1
NN.W2 = NN.W2 + scalar*dJdW2
cost2 = NN.costFunction(X,Y)
print cost1, cost2

dJdW1, dJdW2 = NN.costFunctionPrime(X,Y)
NN.W1 = NN.W1 - scalar*dJdW1
NN.W2 = NN.W2 - scalar*dJdW2
cost3 = NN.costFunction(X, Y)
print cost2, cost3

grad = NN.computeGradients(X,Y)
print grad
