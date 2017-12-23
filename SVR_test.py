#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:36:29 2017

@author: virajdeshwal
"""

import pandas as pd

file = pd.read_csv('Position_Salaries.csv')
X= file.iloc[:,1:2].values
y=file.iloc[:,2].values



from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y= StandardScaler()

X = sc_x.fit_transform(X)
#y=sc_y.fit_transform(y)



from sklearn.svm import SVC

model = SVC(kernel = 'rbf')

model.fit(X,y)
model.predict(X)


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.show()

