# -*- coding: utf-8 -*-
"""
Udemy Ders 9  : Eksik verilerin ortalaması alınarak nan ifadeleriyle (boş değerler ile) yer değiştirilmesi.
Udemy Ders 10 : Verilerin Kategorik hale getirilmesi veri tipi dönüşümü
Udemy Ders 11 : Verilerin Birleştirilmesi ve DataFrame oluşturulması
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# VERİ YÜKLEME,   Data Import
#veriler=pd.read_csv('veriler.csv')

# EKSİK VERİLER , Missing Values
eksikveriler=pd.read_csv('eksikveriler.csv')

# Veri ön işleme , veriler iki şekilde de gösterilebilir.
print("İlk Okunan Eksik Veriler Dosyası : \n",eksikveriler)
boy=eksikveriler[['boy']]
print("Boy Değerleri :\n",boy)
print("Boy ve Kilo Değerleri : \n",eksikveriler[['boy','kilo']])


#sci-kit learn : sci:scientifict kit:kit, araç kutusu, kütüphane
from sklearn.impute import SimpleImputer
# EKSİK DEĞERLERİ DOLDURMA numpy kütüphenesinin nan özelliği ile eksik değerlerin olduğu yere ortalama değerlerini yerleştireceğiz.
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=eksikveriler.iloc[:,1:4].values
print("İlk 'nan' içeren Eksik Değerli :\n",Yas)
imputer=imputer.fit(Yas[:,1:4]) # Makine Öğrenmesi Öğrendiği süreç
Yas[:,1:4]=imputer.transform(Yas[:,1:4]) # Makine Öğrenmesi Uygulandığı süreç
print("Eksik Değerlerin yerin ortalamaların yerleştirilmiş hali : \n",Yas)

# KATEGORİK VERİLER
ulke = eksikveriler.iloc[:,0:1].values
print("Ülke Değerlerinin Kategori halinde Düzenlenmiş hali:\n",ulke)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(eksikveriler.iloc[:,0])
print("Label Encoding : \n",ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print("One Hot Encoding : \n",ulke)

# DataFrame Oluşturma
print(list(range(22))); # 22 veri olduğu için liste oluşturuldu.
sonuc1=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print("Ülkelerin Ayrışması :\n",sonuc1)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print("Boy-Yas-Kilo değerlerinin ayrışması :\n",sonuc2)

cinsiyet=eksikveriler.iloc[:,-1].values # -1 yazılarak tersten veri alınabiliyor sondan başlayarak
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print("Cinsiyetlerin Ayrışması : :\n",sonuc3)

# Verileri Birleştirme
sonuc_1=pd.concat([sonuc1,sonuc2]) # concatenated : birleştirilmiş
print("Kolon başlıklarından tutuşan yerlerin birbirine eklenmesi : \n",sonuc_1)

sonuc_2=pd.concat([sonuc1,sonuc2], axis=1) # concatenated : birleştirilmiş
print("Satır başlıklarından tutuşan yerleri birbirine eklenmesi : \n",sonuc_2)

sonuc=pd.concat([sonuc_2,sonuc3], axis=1) # concatenated : birleştirilmiş
print("Cinsiyetin ilave edilmesi : \n",sonuc)
















