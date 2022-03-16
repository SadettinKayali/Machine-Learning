# -*- coding: utf-8 -*-
"""
127. Model Değerlendirmesi : k-fold Cross Validation
128. Model Seçimi : GridSearchCV
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Data Importing
veriler=pd.read_csv('Social_Network_Ads.csv')
print("Verilerin İLk Okunduğu Hali : \n",veriler)

# Data Preprocessing
X=veriler.iloc[:,[2,3]].values #Bağımsız Değişkenler  
Y=veriler.iloc[:,4].values # Bağımlı Değişken

# Data Train and Test dividing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

# Data Scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# SVC Classification
from sklearn.svm import SVC
# Confusion Matrix
from sklearn.metrics import confusion_matrix

classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred0=classifier.predict(X_test)
print("classifier SVC-rbf y_pred : \n",y_pred0)
cm0=confusion_matrix(y_test,y_pred0)
print("classifier SVC-rbf CM : \n",cm0)

# Model Selection : k-fold Cross Validation
from sklearn.model_selection import cross_val_score
# 1. estimator : classifier (bu durum için)
# 2. X 
# 3. Y 
# 4. cv katlama değeri
basari=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=4)

print("Başarı Ortalaması : \n ",basari.mean())
print("Başarı StandartSapma : \n ",basari.std())

# Model Selection : GridSearch Izgara araması
from sklearn.model_selection import GridSearchCV
p=[{'C':[1,2,3,4,5],'kernel':['linear','rbf']},
   {'C':[1,10,100,1000],'kernel':['rbf'],
    'gamma':[1,0.5,0.1,0.01,0.001]}             ] 
# estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
# param_grid: parametreler/ denenecekler
# scoring : neye göre skorlanacak örn, accuracy
# cv : kaç katlamalı olacağı
# n_jobs : aynı anda çalışılacak iş
gs=GridSearchCV(estimator=classifier,#SVM algoritması
                param_grid=p,
                scoring='accuracy',
                cv=10,
                n_jobs=-1)

grid_search=gs.fit(X_train,y_train)

eniyisonuc=grid_search.best_score_
print("En iyi sonuç : \n",eniyisonuc)
eniyiparametreler=grid_search.best_params_
print("En iyi parametreler : \n",eniyiparametreler)
























