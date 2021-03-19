# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:12:00 2020

@author: Berk
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


walkData = pd.read_excel('D:\Documents\WorkStation\Bitirme Projesi\datas\YururkenVeri.xlsx')
rakimData = pd.read_excel('D:\Documents\WorkStation\Bitirme Projesi\datas\gpsverisi.xlsx')

rakimList = rakimData['Rakim'].values.tolist()
walkGyX = walkData['GyroX'].values.tolist()
 
X_train, X_test, y_train, y_test = train_test_split(rakimList, walkGyX, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
    