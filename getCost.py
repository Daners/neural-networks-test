import time
from neuronal_network import *
from data import *

NN = Neuronal_Network()

X = np.array(([3,5],[5,1],[10,2]),dtype=float)
y = np.array(([75],[82],[93]),dtype=float)

weightstoTry = np.linspace(-5,5,1000)
costs = np.zeros((1000,1000))

startTime = time.clock()
for i in range(1000):
    for j in range(1000):
        NN.W1[0,0]=weightstoTry[i]
        NN.W1[0,1]=weightstoTry[j]
        yHat = NN.forward(X)
        costs[i] =  0.5*sum((y-yHat)**2)

endTime = time.clock()
timeElapsed = endTime - startTime

print timeElapsed
