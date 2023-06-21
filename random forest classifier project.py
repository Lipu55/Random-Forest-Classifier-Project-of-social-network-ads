# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:00:15 2023

@author: MRUTYUNJAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'D:\Datascience Classes\30,31th march\Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=120,random_state=0,max_depth=7)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
bias=classifier.score(X_train, y_train)
bias
variance = classifier.score(X_test, y_test)
variance
