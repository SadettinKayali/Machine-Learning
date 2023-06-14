# -*- coding: utf-8 -*-
"""
Udemy 29. Çoklu Değişken için Veri Hazırlama
Udemy 30. Çoklu Değişken Linear Model Oluşturma ve Model
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# DATA PREPROCESSİNG (VERİ ÖN İŞLEME)
# DATA IMPORTING     (Veri Yükleme)

veriler=pd.read_csv('veriler.csv')
print(veriler)

# KATEGORİK VERİLER => Numeric Veriler
#sci-kit learn : sci:scientifict kit:kit, araç kutusu, kütüphane

Yas=veriler.iloc[:,1:4].values
print("Yaş Değerleri :\n",Yas)

# KATEGORİK VERİLER => Numeric Veriler
ulke = veriler.iloc[:,0:1].values # iloc : integer location
print("Ülke Değerlerinin Kategori halinde Düzenlenmiş hali:\n",ulke)

from sklearn import preprocessing
le=preprocessing.LabelEncoder() # Her bir değere sayısal değer veriyor 0 1 2 3 4 5 gibi
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print("Label Encoding : \n",ulke)

ohe=preprocessing.OneHotEncoder() # Kolon başlıklarını etiketlere ayırmak ve her etikete 0 1 değerlerini yerleştirmek
ulke=ohe.fit_transform(ulke).toarray()
print("One Hot Encoding : \n",ulke)

cinsiyet = veriler.iloc[:,-1:].values # iloc : integer location
print("Cinsiyet Değerlerinin Kategori halinde Düzenlenmiş hali:\n",cinsiyet)

from sklearn import preprocessing
le=preprocessing.LabelEncoder() # Her bir değere sayısal değer veriyor 0 1 2 3 4 5 gibi
cinsiyet[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print("CinsiyetLabel Encoding : \n",cinsiyet)
# Dummy Variable , Kukla Değişken durumuna dikkat etmek gerekir.
ohe=preprocessing.OneHotEncoder() # Kolon başlıklarını etiketlere ayırmak ve her etikete 0 1 değerlerini yerleştirmek
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
print("CinsiyetOne Hot Encoding : \n",cinsiyet)


# DataFrame Oluşturma
print(list(range(22))); # 22 veri olduğu için liste oluşturuldu.
sonuc1=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print("Ülkelerin Ayrışması :\n",sonuc1)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print("Boy-Yas-Kilo değerlerinin ayrışması :\n",sonuc2)

cinsiyet0=veriler.iloc[:,-1].values # -1 yazılarak tersten veri alınabiliyor sondan başlayarak
sonuc3=pd.DataFrame(data=cinsiyet[:,:1],index=range(22),columns=['cinsiyet'])
print("Cinsiyetlerin Ayrışması : :\n",sonuc3)

# Verileri Birleştirme
s_1=pd.concat([sonuc1,sonuc2]) # concatenated : birleştirilmiş
print("Kolon başlıklarından tutuşan yerlerin birbirine eklenmesi : \n",s_1)

s=pd.concat([sonuc1,sonuc2], axis=1) # concatenated : birleştirilmiş
print("Satır başlıklarından tutuşan yerleri birbirine eklenmesi : \n",s)

s2=pd.concat([s,sonuc3], axis=1) # concatenated : birleştirilmiş
print("Cinsiyetin ilave edilmesi : \n",s2)


# Veri Kümelerinin Test ve Eğitim olarak Bölünmesi  Test and Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

# Öznitelik Ölçekleme : Farklı ortamlardaki değerleri aynı ortama kendi içlerinde oranlayıp ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()  # Birbirlerine göre ölçeklendirilerek ortak bir ortama getirildi.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


# MULTIPLE Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


boy=s2.iloc[:,3:4].values
print("Boylar : \n",boy)
sol=s2.iloc[:,:3]
print("Boy kolonunun solu : \n",sol)
sag=s2.iloc[:,4:]
print("Boy kolonunun sağı : \n",sag)

veri=pd.concat([sol,sag],axis=1) # sol ve sag kısımları 'concatenated' : birleştirildi.
print("Boy Kolonu hariç tüm veriler : \n",veri)

# Boy kolonuna göre yeni train ve test kümeleri oluşturulması
# Oluşturulan veri değişkeni x_train kümesini boy ise y_train kümesini oluşturdu.
x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

regressor2=LinearRegression()
regressor2.fit(x_train,y_train)

y_pred=regressor2.predict(x_test)





























