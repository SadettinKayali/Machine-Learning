# -*- coding: utf-8 -*-
"""
Udemy Ders 7 : Verinin Python'dan Yüklenmesi ve içeri alınması (data import)
Udemy Ders 9  : Eksik verilerin ortalaması alınarak nan ifadeleriyle (boş değerler ile) yer değiştirilmesi.

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
# Eksik değerleri numpy kütüphenesinin nan özelliği ile eksik değerlerin olduğu yere ortalama değerlerini yerleştireceğiz.
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=eksikveriler.iloc[:,1:4].values
print("İlk 'nan' içeren Eksik Değerli :\n",Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print("Eksik Değerlerin yerin ortalamaların yerleştirilmiş hali : \n",Yas)
