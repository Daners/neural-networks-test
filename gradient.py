import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**2

epsilon = 1e-4
x = 1.5

numericalGradient = (f(x+epsilon)- f(x-epsilon))/(2*epsilon)

print numericalGradient, 2*x

xv = np.arange(-5,5,0.01)
plt.plot(xv,xv**2,'b')
plt.grid(1)
plt.legend(['gradient'])
plt.show()
