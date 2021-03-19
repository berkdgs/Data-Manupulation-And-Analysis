# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:51:35 2020

@author: Berk
"""
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import folium
import codecs
import os
from sklearn import metrics
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm
import pandas_profiling as pp
from tkinter import *
import time
#Libraries

#-------------------------------------GUI---------------------------------------

root = Tk()
root.geometry("400x400")
root.title("Data Analysis")
clickedData = StringVar()
#clickedData.set("OtururkenVeri2.xlsx")
clickedFunc = StringVar()
#clickedFunc.set("Random Forest")

dropDatas = OptionMenu(root, clickedData, "OtururkenVeri2.xlsx","KosarkenVeri2.xlsx","YururkenVeri.xlsx")
dropDatas.pack()

dropFunc = OptionMenu(root, clickedFunc,"Random Forest","Ridge Reggression","Linear Reggression","Desicion Tree Reggression","Exploratory Data Analysis")
dropFunc.pack()

    
def result():
    global labelData
    global labelFunc
    labelData = Label(root, text = clickedData.get()).pack()    
    labelFunc = Label(root, text = clickedFunc.get()).pack()
    labelInfo = Label(root, text = "Please Close This Window",fg="red",font=("Halvetica,16")).pack()
    
#--------------------------- SHOWING ON MAP ----------------------

def showMap():
    
    data = pd.read_excel('D:\Documents\WorkStation\Bitirme Projesi\datas\gpsverisi.xlsx')
    Y = data['Enlem'].values.tolist()
    Z = data['Boylam'].values.tolist()
    
    #40.7514794,30.3494019 verilerin alındığı mekanın lokasyon bilgisi.
    
    folium_map = folium.Map(location=[40.7514794,30.3494019],
                            zoom_start=13,
                            tiles="CartoDB dark_matter")
    for i in range (1,150):
        marker = folium.CircleMarker(location=[Y[i],Z[i]]) #GPS verilerini yerleştir.
        marker.add_to(folium_map)
    
    folium_map.save("my_map.html")
    print("my_map.html saved in working directory")    
    root.mainloop()
    
#--------------------------- SHOWING ON MAP ----------------------

    
button = Button(root, text="Calculate Selected", command=result)
button.pack()

buttonMap = Button(root, text="Show on Map", command=showMap)
buttonMap.pack()

root.mainloop()

selectionData = clickedData.get()
selectionFunc = clickedFunc.get()
#-------------------------------------GUI---------------------------------------

#Reading Datas
path = 'D:\Documents\WorkStation\Bitirme Projesi\datas\\'+ selectionData
selectedData = pd.read_excel(path)
#Reading Datas


#------------------------------RANDOM FOREST---------------------------------------

def randFore(data): 
    
    plt.figure(figsize=(12,8))
    X = data.drop(['AccX'], axis=1)
    y = data['AccX']
    sns.distplot(y)
    plt.show()
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth = 3, random_state=42)
    rf.fit(X_train, y_train)
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    
    mae = mean_absolute_error(rf.predict(X_test), y_test)
    mse = mean_squared_error(rf.predict(X_test), y_test)
    rmse = np.sqrt(mse)
    
    print('Mean Absolute Error (MAE): %.2f' % mae)
    print('Mean Squared Error (MSE): %.2f' % mse)
    print('Root Mean Squared Error (RMSE): %.2f' % rmse)
    print("----------------------------------------------------------")
    rf = RandomForestRegressor(n_estimators=100, max_depth = 8, random_state=42)
    rf.fit(X_train, y_train)
    
    mae = mean_absolute_error(rf.predict(X_test), y_test)
    mse = mean_squared_error(rf.predict(X_test), y_test)
    rmse = np.sqrt(mse)
    
    print('Mean Absolute Error (MAE): %.2f' % mae)
    print('Mean Squared Error (MSE): %.2f' % mse)
    print('Root Mean Squared Error (RMSE): %.2f' % rmse)
    
    plt.figure(figsize=(16, 9))
    
    ranking = rf.feature_importances_
    features = np.argsort(ranking)[::-1][:10]
    columns = X.columns
    
    plt.title("Feature importances based on Random Forest Regressor", y = 1.03, size = 18)
    plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
    plt.xticks(range(len(features)), columns[features], rotation=80)
    plt.show()
#------------------------------RANDOM FOREST---------------------------------------

# ------------------------------------ RIDGE REGRESSION ---------------------------------------

def ridgeRegg(data):
    
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

# ------------------------------------ RIDGE REGRESSION ---------------------------------------


#------------------------------------LINEAR REGGRESSION------------------------------------

def linRegg(data):
    
    data.shape
    data.describe()
    
    data.plot(x='AccX', y='AccY', style='o')  
    plt.title('AccX vs AccY')  
    plt.xlabel('AccX')  
    plt.ylabel('AccY') 
    plt.show()
    
    X = data['AccX'].values.reshape(-1,1)
    y = data['AccY'].values.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    print(regressor.intercept_)
    print(regressor.coef_)
    
    
    y_pred = regressor.predict(X_test)
    
    data = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    
    print(data)
    
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

#------------------------------------LINEAR REGGRESSION------------------------------------

#--------------------------------DECISION TREES REGGRESSION-----------------------------------

def decTreeRegg(data):
    
    X = data.drop(['AccX'], axis=1)
    y = data['AccX']
    sns.distplot(y)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    d_tree1 = DecisionTreeRegressor(max_depth = 3, random_state=42)
    d_tree1.fit(X_train, y_train)
    
    predictions = d_tree1.predict(X_test)
    errors = abs(predictions - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 3), '%.')
    
    d_tree2 = DecisionTreeRegressor(max_depth = 8, random_state=42)
    d_tree2.fit(X_train, y_train)
    predictions = d_tree2.predict(X_test)
    
    errors = abs(predictions - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 3), '%.')
    
    plt.figure(figsize=(16, 9))
    
    ranking = d_tree2.feature_importances_
    features = np.argsort(ranking)[::-1][:10]
    columns = X.columns
    
    plt.title("Feature importances based on Decision Tree Regressor", y = 1.03, size = 18)
    plt.bar(range(len(features)), ranking[features], color="lime", align="center")
    plt.xticks(range(len(features)), columns[features], rotation=80)
    plt.show()

#--------------------------------DECISION TREES REGGRESSION-----------------------------------

#-------------------------------EXPLORATORY DATA ANALYSIS--------------------------------------

def eda (data):
        
    data.head()
    data.shape
    data.size
    data.sample()
    data.tail()
    data.info()
    data.describe()
    pp.ProfileReport(data)
    
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title('Sensor Verileri')
    aX=data['AccX']
    plt.plot(aX, label = "AccX", color='red')
    bX=data['AccY']
    plt.plot(bX, label = "AccY", color='blue')
    cX=data['AccZ']
    plt.plot(cX, label = "AccZ", color='purple')
    plt.legend()
    
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title('Sensor Verileri')
    aX=data['GyroX']
    plt.plot(aX, label = "GyroX", color='red')
    bX=data['GyroY']
    plt.plot(bX, label = "GyroY", color='blue')
    cX=data['GyroZ']
    plt.plot(cX, label = "GyroZ", color='green')
    plt.legend()
    
    sns.pairplot(data)
    
#-------------------------------EXPLORATORY DATA ANALYSIS--------------------------------------

if selectionFunc == "Random Forest":
    randFore(selectedData)
elif selectionFunc == "Ridge Reggression":
    ridgeRegg(selectedData)
elif selectionFunc == "Linear Reggression":
    linRegg(selectedData)
elif selectionFunc == "Desicion Tree Reggression":
    decTreeRegg(selectedData)
elif selectionFunc == "Exploratory Data Analysis":
    eda(selectedData)


cputime = time.process_time()
print("CPU TIME:")
print(cputime)