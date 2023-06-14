# -*- coding: utf-8 -*-
"""
Udemy Ders 19. Veri Yükleme ve Ön İşleme Şablonunun Kullanılması ve Regresyona Hazırlık
Udemy Ders 20. Phyton ile Basit Doğrusal Regresyon Model İnşası
Udemy Ders 21. Python ile Basit Doğrusal Regresyon Uygulaması
Udemy Ders 22. Basit Doğrusal Regresyon Görselleştirilmesi
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# DATA PREPROCESSİNG (VERİ ÖN İŞLEME)
# DATA IMPORTING     (Veri Yükleme)

veriler=pd.read_csv('satislar.csv')
print(veriler)

aylar=veriler[['Aylar']]
print(aylar)
satislar=veriler[['Satislar']]
print(satislar)

#satislar2=veriler.iloc[:,0:1].values
#satislar2=veriler.iloc[:,:1].values
#print(satislar2)

# SEPARATION OF DATA FOR TRAINING AND TESTING
# VERİLERİN EĞİTİM ve TEST için BÖLÜNMESİ

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

"""
# DATA SCALER  Verileri Ölçekleme
# Öznitelik Ölçekleme : Farklı ortamlardaki değerleri aynı ortama kendi içlerinde oranlayıp ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()  # Birbirlerine göre ölçeklendirilerek ortak bir ortama getirildi.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

# ctrl+i basılarak help komutu çalışır. 

# MODEL CREATİNG - LINEAR REGRESSION  
# Model İnşası Linear Regresyon
from sklearn.linear_model import LinearRegression # Linear Regression kütüphanesinin yüklenmesi
lr=LinearRegression()
lr.fit(X_train,Y_train)  # Ölçeklenmiş X_train ve Y_train değerleri ile bir model inşa edilmesi

# UYGULAMA
# X_train'den Y_train öğrendi, X_test'ten Y_test öğrendi. Bizde X_test'ten tahmin ettiriyoruz.

tahmin=lr.predict(X_test)
"""

# MODEL CREATİNG - LINEAR REGRESSION  
# Model İnşası Linear Regresyon
from sklearn.linear_model import LinearRegression # Linear Regression kütüphanesinin yüklenmesi
lr=LinearRegression()
lr.fit(x_train,y_train)  # Ölçeklenmiş X_train ve Y_train değerleri ile bir model inşa edilmesi

tahmin=lr.predict(x_test)

# Visualization part
# Rastgele seçilmiş 0.67 lik kısımlarda işlemler yapılıyor şuanda
x_train=x_train.sort_index()
y_train=y_train.sort_index()

plot.plot(x_train,y_train)
plot.plot(x_test,lr.predict(x_test))

plot.title("Aylara Göre Satış")
plot.xlabel("Aylar")
plot.ylabel("Satışlar")



