# -*- coding: utf-8 -*-
"""
93. Python ile Rasgele Yaklaşımın Kodlanması
94. Python ile UCB kodlama   Upper Confidence Bound  Üst Güven Sınırı
: Her bir sonucu ayrı ayrı hafızasında tutarak öğrenme

En yüksek UCB değerine sahip değeri seçmeye çalışmak.
UCB algoritması eski değerleri hafızasında tutarak daha mantıklı değeri seçmeye çalışıyor.

96. Python Kodlaması : Thompson Sampling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math

veriler= pd.read_csv('Ads_CTR_Optimisation.csv')

# Random Selection
import random

N=10000  # satır sayısı
d=10     # sütun sayısı
toplam=0
secilenler=[]
# ad = advertisement reklam
# 0'dan N. satıra kadar rastgele bir sütundan seçiyor, sonuc 1 ise ödül alıyor, ödül aldığı yerleri yeni bir listeye ekliyor. sonunda histogram grafiği ile görselleştirme yapılıyor
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad] # verilerdeki n. satır = 1
    toplam=toplam+odul
print('Random - Toplam Ödül : \n',toplam)
plot.hist(secilenler)
plot.show()

# Upper Confidence Bound   Üst Güven Sınırı
N=10000 # Reklama tıklanma miktarı
d=10    # reklam,ilan sayısı

oduller=[0]*d # Ri(n) ödüller 'd' elemanlı liste olacak. Başlangıçta bütün ilanların ödülü 0 olacak 
tiklamalar=[0]*d # Ni(n) o ana kadar ki tıklamalar
toplam=0 # toplam ödül
secilenler=[]
for n in range(1,N):
    ad=0  # seçilen ilan
    max_ucb=0 # max_ucb değeri başlangıçta sıfır
    for i in range(0,d):
        if(tiklamalar[i]>0): # tıklamalar yapılmaya başlandıktan sonra.
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i]) # di(n)
            ucb=ortalama+delta
        else:
            ucb=N*10  # eğer hiç tıklanmamış olursa diye
        if max_ucb<ucb: # max'tan büyük bir ucb çıktı, max_ucb güncelle
            max_ucb=ucb
            ad=i
                
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
print('UCB - Toplam Ödül : \n',toplam)
plot.hist(secilenler)
plot.show()

# Thompson Sampling
import random
N=10000 # Reklama tıklanma miktarı
d=10    # reklam,ilan sayısı

toplam=0 # toplam ödül
secilenler=[]
birler=[0]*d    # birlere tıklanma sayısı her bi reklam ilanı için
sifirlar=[0]*d  # sifirara tiklanma sayisi her bir reklam ilanı için
for n in range(1,N):
    ad=0  # seçilen ilan
    max_th=0 # max_ucb değeri başlangıçta sıfır
    for i in range(0,d):
        rand_beta=random.betavariate(birler[i]+1, sifirlar[i]+1)
        if rand_beta>max_th:
            max_th=rand_beta
            ad=i
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1
    
    toplam=toplam+odul
print('Thompson Sampling - Toplam Ödül : \n',toplam)
plot.hist(secilenler)
plot.show()




