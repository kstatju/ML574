# -*- coding: utf-8 -*-
"""
@author: %(Kanak)s
"""


path = 'Kanak_lab1\\';


import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")

X = np.array([0, 5, 8, 12]).reshape(-1,1)
Y = np.array([10, 5, 12, 0])
X = PolynomialFeatures(degree=3).fit_transform(X)
lm = linear_model.LinearRegression()
model = lm.fit(X, Y)
model.coef_
Xnew = np.arange(-5, 20).reshape(-1,1)
pred = model.predict(PolynomialFeatures(degree=3).fit_transform(Xnew))
pred1 = model.predict(X)

plt.scatter(X[:,1], Y, alpha=0.5, facecolors='black', edgecolors='black', s=100)
plt.plot(X[:,1], pred1, color = 'red')
plt.title('Scatter plot with polynomial line')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-5,20)
plt.xlim(-5,20)
plt.show()