# -*- coding: utf-8 -*-
"""
33. Ödev 1: Çözüm 1. Parça: Verinin hazırlanması ve Çoklu Doğrusal Regresyon
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# DATA PREPROCESSİNG (VERİ ÖN İŞLEME)
# DATA IMPORTING     (Veri Yükleme)

veriler=pd.read_csv('odev_tenis.csv')
print(veriler)

# KATEGORİK VERİLER => Numeric Veriler
#sci-kit learn : sci:scientifict kit:kit, araç kutusu, kütüphane

from sklearn import preprocessing
le=preprocessing.LabelEncoder() # Her bir değere sayısal değer veriyor 0 1 2 3 4 5 gibi
ohe=preprocessing.OneHotEncoder() # Kolon başlıklarını etiketlere ayırmak ve her etikete 0 1 değerlerini yerleştirmek
# Windy ve Play Label Encoding yapıldı. Tek tek uğraşmamak için alternatif fonksiyon kullanıldı. Örnek olması için
veriler2=veriler.apply(le.fit_transform)

# Outlook 3 çeşitli olduğu için OneHotEncoding uygulandı
outlook_le=veriler2.iloc[:,:1]
print("LabelEncoder : Outlook : \n",outlook_le)
outlook_ohe=ohe.fit_transform(outlook_le).toarray()
print("OneHotEncoder : Outlook : \n",outlook_ohe)

# DataFrame Oluşturma
outlook_OHE=pd.DataFrame(data=outlook_ohe,index=range(14),columns=["overcast","rainy","sunny"])
# OneHotEncoding uygulanan Outlook ile Temperature ve Humidity birleştirildi. kolon birleşimi için axis=1
outlook_OHE_temperature_humidty=pd.concat([outlook_OHE,veriler.iloc[:,1:3]],axis=1)
# Son olarak LabelEncoding uygulanan windy ve play kolonları eklendi.
sonveriler=pd.concat([veriler2.iloc[:,-2:],outlook_OHE_temperature_humidty],axis=1)


# Veri Kümelerinin Test ve Eğitim olarak Bölünmesi  Test and Train
from sklearn.model_selection import train_test_split
# Humidity bağımlı değişken, son kolona kadar öğrenecek, son kolon ile tahmin yapacak 
x_train,x_test,y_train,y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)
# x_train : bağımsız değişken  y_train : bağımlı değişken için
# x_test  : bağımsız değişken  y_train : bağımlı değişken için


# Öznitelik Ölçekleme : Farklı ortamlardaki değerleri aynı ortama kendi içlerinde oranlayıp ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()  # Birbirlerine göre ölçeklendirilerek ortak bir ortama getirildi.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


# MULTIPLE Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) # Tahmin Model
print("Y prediction : ",y_pred)


# BACKWARK ELIMINATION - OLS model raporu oluşturma
# Sistemin doğruluğunu bozan verileri çıkarmak , anlamsız değişkenleri çıkarmak 
import statsmodels.api as sm # Modelin ve modeldeki değişkenlerin başarısı ile alakalı kütüphane
# Beta0 değerlerini için kolonların sonuna bir '1' değerinden oluşan 22x1 matrisini ilave ediyor.
# Values=veri kısmı nereye ekleyeceğini belirtiyor.
# axis=1 satır olarak yeni matris ekleme yapar, axis yazılırsa satır olarak ekleme yapar.
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
print("Yeni Oluşturulmuş Beta0 değerli tablo : \n",X)
X_list=sonveriler.iloc[:,[0,1,2,3,4,5]].values
#X_list=veri.iloc[:,:].values
Result_OLS=sm.OLS(endog=sonveriler.iloc[:,-1:],exog=X_list)
# Boy bağımlı değişkeni ile bağımsız değişkenleri arasındaki bağlantıyı kurmak için uğraşılıyor.
model=Result_OLS.fit() # istatistiksel değerleri çıkarmaya yarıyor
print(model.summary())             


# P değeri en yüksek olan kolonu kaldırıyoruz.

sonveriler=sonveriler.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
print("Yeni Oluşturulmuş Beta0 değerli tablo : \n",X)
X_list=sonveriler.iloc[:,[0,1,2,3,4]].values
#X_list=veri.iloc[:,:].values

Result_OLS=sm.OLS(endog=sonveriler.iloc[:,-1:],exog=X_list)

# Boy bağımlı değişkeni ile bağımsız değişkenleri arasındaki bağlantıyı kurmak için uğraşılıyor.
model=Result_OLS.fit() # istatistiksel değerleri çıkarmaya yarıyor
print(model.summary())   
             
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]       
             
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test) # Tahmin Model  
print("Y prediction : ",y_pred)       

# Dahada iyileştirmek için işlemlere devam edilebilir.   
             
             
             
             