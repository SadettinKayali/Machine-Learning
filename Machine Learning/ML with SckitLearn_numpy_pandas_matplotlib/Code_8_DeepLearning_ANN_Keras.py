# -*- coding: utf-8 -*-
"""
118. Problemin Tanımı ve Veri Kümesi
119. YSA Kodlamasına Giriş ve Keras ile Tanışma
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Data Importing
veriler=pd.read_csv('Churn_Modelling.csv')
print("Verilerin İLk Okunduğu Hali : \n",veriler)

# Data Preprocessing
X=veriler.iloc[:,3:13].values #Bağımsız Değişkenler RowNumber,CustomerID,Surname kısımları çıkarılıyor. İleride sıkıntı çıkarabileceği için
Y=veriler.iloc[:,13].values # Bağımlı Değişken
#ANN sıfır ile bir arasında değer aldığı için ülke,cinsiyet encoding işlemi uygulanacak
# ENCODER : Categoric to Numeric
from sklearn import preprocessing
le1=preprocessing.LabelEncoder()
X[:,1]=le1.fit_transform(X[:,1]) # Yeni X'in 1.kolonu olacak coğrafi kısım encoding edilmiş olacak.

le2=preprocessing.LabelEncoder()
X[:,2]=le1.fit_transform(X[:,2]) # Yeni X'in 2.kolonu olacak cinsiyet kısmı encoding edilmiş olacak.

# Ülkelerde saçmalamamak için OneHotEncoding uygulanacak her ülkeye tek tek değer verilmesi için
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")

X=ohe.fit_transform(X)
X=X[:,1:]

# Data Train and Test dividing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

# Data Scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#ANN Keras
import keras
from keras.models import Sequential # Yapay Sinir Ağı kullanacağımızı belirtmek için yapay sinir ağı oluşturmaya yarıyor.
from keras.layers import Dense # Yapay Sinir ağındaki katman oluşturmak için

classifier=Sequential() # burada ANN sınıflandırma olarak kullandık
# Genellikle giriş katmanları ve çıkış katmanlarının toplamının yarısı şeklinde gizli katman kullanılır. üçgensel yapı elde edilmeye çalışır, güzel çalışır.

#classifier.add(Dense(6,init="uniform",activation="relu",input_dim=11))
classifier.add(Dense(6,activation='relu',input_dim=11)) # ilki olduğu için inputu var o yüzden iput_dim ekliyoruz
classifier.add(Dense(6,activation='relu',)) # ikinci gizli katman olduğu için inputu yok oyüzden input_dim eklemiyoruz.
classifier.add(Dense(1,activation="sigmoid",)) # çıkış katmanı

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,epochs=50) #epochs değeri kaç tur ANN çalıştırılacağını velirtiyor., Bağımsız değişkenlerden bağımlı değişken y_train i öğren.

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5) # True False 1 0 döndürecek

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm) # diagonal değerler doğruları verir.



























