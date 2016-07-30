# -*- coding: utf-8 -*-

import numpy as np
import seaborn
from matplotlib import pyplot as plt

def ZeroOneLoss(z):
    return 0 if z>0 else 1

def HingeLoss(z):
    return max(0,1-z)


def ExponentialLoss(z):
    return np.exp(-z)


def LogisticLoss(z):
    return np.log2(1+np.exp(-z))


z = [i/1000 for i in range(-2000,2000,5)]

y0 = [ZeroOneLoss(i) for i in z]
y1 = [HingeLoss(i) for i in z]
y2 = [ExponentialLoss(i) for i in z]
y3 = [LogisticLoss(i) for i in z]

plt.title('Loss function',fontsize=15)
plt.plot(z,y0,'k',label='Zero-One loss')
plt.plot(z,y1,'r',label='Hinge loss')
plt.plot(z,y2,'g--',label='ExponentialLoss loss')
plt.plot(z,y3,'b-.',label='Logistic loss')
plt.xlabel('z',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.ylim(0,3)
plt.legend(fontsize=15)
plt.show()
