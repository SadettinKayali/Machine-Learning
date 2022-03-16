# -*- coding: utf-8 -*-
"""
Birliktelik Kural Çıkarımı 
ARM : Associative Rule Mining   / Association Rule Mining
ARL : Associative Rule Learning / Association Rule Learning
88. Python ile Apriori Algoritmasının Kodlanması
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

veriler=pd.read_csv('sepet.csv',header=None) # read_csv komutu başlıkları da alır. ama bu verilerde başlık yok
# List of list oluşturucaz. Verileri Listenin içerisindeki listeler halinde modifiye edilecek.

# Create List of List
t=[] # list()  creating transaction list
for i in range(0,7501): #ï 7501 veri olduğu için
    t.append([str(veriler.values[i,j]) for j in range(0,20)])  # Dosyadaki her bir satırdaki verileri satır satır liste biçiminde ayıracak 20 yazılma sebebi bir satırda max 20 tane ürün vardır diye düşünüldü.

from apyori import apriori
kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2) # kavramların ne olduğunun ekran görüntüsü var zaten

kurallar1=list(kurallar)

print(kurallar1) # yazdırırken tekrardan list şeklinde dönüşüm yapmak gerekiyor, class olarak ayrımlarını yaptıüı için

