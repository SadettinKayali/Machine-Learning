# -*- coding: utf-8 -*-
"""
123. PCA, Python ile kodlanması
PCA (Principal Component Analysis) Birncil Bileşen Analizi

125. LDA: Python ile kodlaması
LDA: Linear Discriminant Analysis

"""
#Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Data Importing
veriler=pd.read_csv('Wine.csv')
print("Verilerin İLk Okunduğu Hali : \n",veriler)

# Data Preprocessing
X=veriler.iloc[:,0:13].values #Bağımsız Değişkenler  
Y=veriler.iloc[:,13].values # Bağımlı Değişken

# Data Train and Test dividing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

# Data Scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# PCA : Principal Component Analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=2) # kaç boyuta indirgemesini istiyoruz

X_train2=pca.fit_transform(X_train) # boyutu indirgenmiş eğitim
X_test2=pca.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
# PCA dönüşümünden önce gelen LogisticResgression
classifier=LogisticRegression(random_state=0) # Aynı yapıyı kulllanmak istenildiği için randomstate 0 yapılıyor tekrar kullanımlarda sorun olmaması için
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#print("Before PCA LogReg y_pred : \n",y_pred)
# PCA dönüşümünden sonra gelen LogisticResgression
classifier2=LogisticRegression(random_state=0) # Aynı yapıyı kulllanmak istenildiği için randomstate 0 yapılıyor tekrar kullanımlarda sorun olmaması için
classifier2.fit(X_train2,y_train)
y_pred2=classifier2.predict(X_test2)
#print("After PCA LogReg y_pred : \n",y_pred2)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
# Actual vs Before PCA
cm0=confusion_matrix(y_test,y_pred)
print("Actual vs Before PCA Confusion Matrix : \n",cm0)
# Actual vs After PCA
cm1=confusion_matrix(y_test,y_pred2)
print("Actual vs After PCA Confusion Matrix : \n",cm1)
# Before vs After PCA
cm2=confusion_matrix(y_pred,y_pred2)
print("Before vs After PCA Confusion Matrix : \n",cm2)

# LDA : Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2) # kaç boyuta indirgemesini istiyoruz

X_train3=lda.fit_transform(X_train,y_train) # boyutu indirgenmiş eğitim
X_test3=lda.transform(X_test) # Sınıflar arası mesafeyi maksimize ettiği için 2 eğitim giriyoruz.
# LDA After
classifier3=LogisticRegression(random_state=0) # Aynı yapıyı kulllanmak istenildiği için randomstate 0 yapılıyor tekrar kullanımlarda sorun olmaması için
classifier3.fit(X_train3,y_train)
y_pred3=classifier3.predict(X_test3)

cm3=confusion_matrix(y_pred,y_pred3)
print("Actual vs After LDA Confusion Matrix : \n",cm3)










