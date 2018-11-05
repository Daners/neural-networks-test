import matplotlib.pyplot as plt
from neuronal_network import *
from data import *
from trainer import *

NN = Neural_Network()

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
Y = np.array(([75], [82], [93], [70]), dtype=float)

fig = plt.figure(0,(8,3))

plt.subplot(1,2,1)
plt.scatter(X[:,0], Y)
plt.grid(1)
plt.xlabel('Hours Sleeping')
plt.ylabel('Test Score')

plt.subplot(1,2,2)
plt.scatter(X[:,1], Y)
plt.grid(1)
plt.xlabel('Hours Studying')
plt.ylabel('Test Score')
plt.show();
