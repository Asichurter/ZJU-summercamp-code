import matplotlib.pyplot as plt
import numpy as np

def fx(x):
    return (2*x+5)/(3*x-2)

def fy(y):
    return (1.5*y+2.5)/(-2*y+3)

a = np.linspace(5,10,1000)
xx = fx(a)
yy = fy(a)

plt.plot(xx,yy,color='blue')
plt.plot(xx[[0,-1]],yy[[0,-1]],color='red')
plt.show()