# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:48:29 2020

@author: Berk
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_excel('D:\Documents\Spyder\Deep Learning\datas\YururkenVeri.xlsx')

X = data['AccX'].values.reshape(-1,1)
Y = data['AccY'].values.reshape(-1,1)

X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.3,random_state=3)
print( len(X_test), len(y_test))

lr = LinearRegression()
lr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) 
rr100.fit(X_train, y_train)

rr = Ridge(alpha=0.01)
rr.fit(X,Y)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)

Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

print ("linear regression train score:", train_score)
print ("linear regression test score:", test_score)
print ("ridge regression train score low alpha:", Ridge_train_score)
print ("ridge regression test score low alpha:", Ridge_test_score)
print ("ridge regression train score high alpha:", Ridge_train_score100)
print ("ridge regression test score high alpha:", Ridge_test_score100)

plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7)

plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') 

plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()