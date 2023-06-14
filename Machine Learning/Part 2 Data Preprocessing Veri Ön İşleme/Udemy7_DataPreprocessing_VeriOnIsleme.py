# -*- coding: utf-8 -*-
"""
Udemy Ders 7 : Verinin Python'dan Yüklenmesi ve içeri alınması (data import)
"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# VERİ YÜKLEME,   Data Import
# Dosyalar aynı yerde ise doğrudan okunabilir. Farklı konumlarda ise dosya yolu yazılmalıdır.
veriler=pd.read_csv('veriler.csv')
#veriler=pd.read_csv('veriler.csv') # iki gösterimide doğrudur.

# Veri ön işleme , veriler iki şekilde de gösterilebilir.
print(veriler)
boy=veriler[['boy']]
print(boy)
print(veriler[['boy','kilo']])


# class , def tanımları

class insan:
    boy=180
    def kosmak(self,b):   # y=f(x)
        return b+10       # f(x)=x+10

ali=insan()
print("Ali'nin Boyu : ",ali.boy)
print("Ali'nin Koşusu : ",ali.kosmak(100))

# list , liste tanımı
liste=[1,2,34]
print(liste)

# EKSİK VERİLER , Missing Values
eksikveriler=pd.read_csv('eksikveriler.csv')

# Veri ön işleme , veriler iki şekilde de gösterilebilir.
print(eksikveriler)
boy=eksikveriler[['boy']]
print(boy)
print(veriler[['boy','kilo']])


#sci-kit learn : sci:scientifict kit:kit, araç kutusu, kütüphane
from sklearn.impute import SimleImpute
# Eksik değerleri numpy kütüphenesinin nan özelliği ile eksik değerlerin olduğu yere ortalama değerlerini yerleştireceğiz.
imputer=SimpleImputer(missing_vales=np.nan,strategy='mean')
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])