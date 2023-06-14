# -*- coding: utf-8 -*-
"""
Udemy Ders 6  : Kütüphenelerin Yüklenmesi
Udemy Ders 7  : Verinin Python'dan yüklenmesi ve içeri alınması DATA  IMPORT
Udemy Ders 9  : Eksik verilerin ortalaması alınarak nan ifadeleriyle (boş değerler ile) yer değiştirilmesi.
Udemy Ders 10 : Verilerin Kategorik hale getirilmesi veri tipi dönüşümü
Udemy Ders 11 : Verilerin Birleştirilmesi ve DataFrame oluşturulması
Udemy Ders 12 : Veri Kümesinin Eğitim ve Test olarak Bölünmesi
Udemy Ders 13 : Öznitelik Ölçekleme
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

# KATEGORİK VERİLER => Numeric Veriler
ulke = eksikveriler.iloc[:,0:1].values # iloc : integer location
print("Ülke Değerlerinin Kategori halinde Düzenlenmiş hali:\n",ulke)

from sklearn import preprocessing
le=preprocessing.LabelEncoder() # Her bir değere sayısal değer veriyor 0 1 2 3 4 5 gibi
ulke[:,0]=le.fit_transform(eksikveriler.iloc[:,0])
print("Label Encoding : \n",ulke)

ohe=preprocessing.OneHotEncoder() # Kolon başlıklarını etiketlere ayırmak ve her etikete 0 1 değerlerini yerleştirmek
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












































