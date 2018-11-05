import matplotlib.pyplot as plt
import numpy as np



def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

testValue  =  np.arange(-5,5,0.01)
plt.plot(testValue,sigmoid(testValue),linewidth=2)
plt.plot(testValue,sigmoidPrime(testValue),linewidth=2)
plt.grid(1)
plt.legend(['sigmoid','sigmoidPrime'])
plt.show()
#print sigmoide(1)
#print sigmoide(np.array([-1,0,1]))
#print sigmoide(np.random.randn(3,3))
