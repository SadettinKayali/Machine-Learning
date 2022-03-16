# -*- coding: utf-8 -*-
"""
SVM : Support Vector Machines      amaç marjini maksimize eden doğruyu bulmak aralığın en fazla olması ve hiç veri içermemesi
SVR : Support Vector Regression    amaç aralığa en fazla veriyi almak

scikit learn .org sitesini karıştırmakta fayda var.
Classifier Comparison..

maksimum marjin aralığına sahip doğruyu seçmek.  

53. Python ile Sınıflandırma - Logistic Regression Uygulaması
54. Confusion Matrix (Karmaşıklık Matrisi) ve Sınıflandırma Şablonu
56. Python ile KNN Kodlaması
58. SVM Algoritmasının Python ile kodlanması
60. Python ile Çekirdek Hilesi Kodlama
62. Python Kodu : Naive Bayes ve GaussianNB, MultinominalNB, BernolliNB
64. Python ile karar ağacı sınıflandırma kodu
66. Python Kodu: Rassal orman ile sınıflandırma
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# DATA PREPROCESSİNG (VERİ ÖN İŞLEME)
# DATA IMPORTING     (Veri Yükleme)

veriler=pd.read_csv('veriler.csv')
print("Okunan Veriler : \n",veriler)
# DataFrame değerleri
x=veriler.iloc[:,1:4] # Bağımsız Değişkenler
y=veriler.iloc[:,4:]  # Bağımlı Değişken
# Numpy dizisi değerleri
X=x.values
Y=y.values
print("Boy Kilo Yaş : \n",x)
print("Cinsiyet : \n",y)

# Datas Train and Test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

# DataScaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

# CLASSIFICATION ALGORTIHMS
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train) # Eğitilmiş LogReg modeli

y_pred=log_reg.predict(X_test)
print(" Logistic Regression y_pred : \n",y_pred)
print("y_test : \n",y_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix Çıktısı : \n",cm)

# kNN 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print("kNN y_pred : \n",y_pred)

cm=confusion_matrix(y_test,y_pred)
print("KNN - confusion matrix : \n",cm)

# SVM , SVM-kernel Trick

from sklearn.svm import SVC

svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print("SVC-linear y_pred : \n",y_pred)
cm=confusion_matrix(y_test,y_pred)
print("SVC-linear CM : \n",cm)

svc1=SVC(kernel='rbf')
svc1.fit(X_train,y_train)
y_pred=svc1.predict(X_test)
print("SVC-rbf y_pred : \n",y_pred)
cm0=confusion_matrix(y_test,y_pred)
print("SVC-rbf CM : \n",cm0)
    
svc1=SVC(kernel='poly')
svc1.fit(X_train,y_train)
y_pred=svc1.predict(X_test)
print("SVC-poly y_pred : \n",y_pred)
cm0=confusion_matrix(y_test,y_pred)
print("SVC-poly CM : \n",cm0)
    
# Naive Bayes
from sklearn.naive_bayes import GaussianNB # diğerlerini de denemekte fayda var

gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
print("Naive Bayes -Gaussian y_pred : \n",y_pred)
cm0=confusion_matrix(y_test,y_pred)
print("Naive Bayes -Gaussian CM : \n",cm0)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')  # default olanı 'gini'
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
print("Decision Tree y_pred : \n",y_pred)
cm0=confusion_matrix(y_test,y_pred)
print("Decision Tree  CM : \n",cm0)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')# default olanı 'gini' , estimator değiştirilip denenebilir
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
print("Random Forest y_pred : \n",y_pred)
cm0=confusion_matrix(y_test,y_pred)
print("Random Forest  CM : \n",cm0)

y_proba=rfc.predict_log_proba(X_test) # prediction probabilities
print("y_probility : \n",y_proba[:,0])
print("y_test : \n",y_test)

# ROC, AUC RECEIVER OPERATING CHARACTERISTIC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc_curve
fpr,tpr,threshold=roc_curve(y_test,y_proba[:,0],pos_label='e')

print("True Positive : \n",tpr,"\nFalse Positive : \n",fpr,"\nThreshold :\n",threshold)
















