"""
81. Python ile K-Means Kodlaması
WCSS : within clusters sum of squares
84. Hiyerarşik Bölütleme / Kümeleme ve Dendogramların kullanılması
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

veriler=pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:].values # Hacim ve Maaş

#K-Means
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

print("Kmeans Cluster_centers : \n",kmeans.cluster_centers_)

# ("WCSS : within clusters sum of squares") hesaplanması ve tespiti için
sonuclar=[] # bos liste olusturup sonuçları içine yazdırcaz
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123) #☻ Her seferide aynı başlangıç değeriyle cluster denemesi yapması için 123 yazıldı, değer fark etmezdi
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # sonuclar listesine yazdırılıyor. ekleniyor

plot.plot(range(1,11),sonuclar)
plot.xlabel('number of clusters')
plot.ylabel("WCSS : within clusters sum of squares")
plot.title("Kmeans Clusters/WCSS")
plot.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=123) #☻ Her seferide aynı başlangıç değeriyle cluster denemesi yapması için 123 yazıldı, değer fark etmezdi
y_pred=kmeans.fit_predict(X)
print("KMeans c=3 y_pred : \n",y_pred)
plot.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='r') # X in Y_pred değeri 0 ise 0 yap, 0 ise 1 yap
plot.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='b')
plot.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='g')
plot.title("Kmeans Clusters =3 ")
plot.show()

#Hierarchical
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
#ac.fit(X) # eğitim sistemi inşa ediyor
y_pred=ac.fit_predict(X) # eğitim sistemini inşa edip tahmin eder
print("Hierarchical c=3 y_pred : \n",y_pred)

plot.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='r') # X in Y_pred değeri 0 ise 0 yap, 0 ise 1 yap
plot.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='b')
plot.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='g')
plot.title("Hierarchical Clusters =3 ")
plot.show()

# Dengdogram,ward distance
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plot.show()











# KArşılaştırmalı denemeler
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=123) #☻ Her seferide aynı başlangıç değeriyle cluster denemesi yapması için 123 yazıldı, değer fark etmezdi
y_pred=kmeans.fit_predict(X)
print("KMeans c=3 y_pred : \n",y_pred)
plot.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='r') # X in Y_pred değeri 0 ise 0 yap, 0 ise 1 yap
plot.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='b')
plot.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='g')
plot.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='y')
plot.title("Kmeans Clusters =4 ")
plot.show()

#Hierarchical
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
#ac.fit(X) # eğitim sistemi inşa ediyor
y_pred=ac.fit_predict(X) # eğitim sistemini inşa edip tahmin eder
print("Hierarchical c=3 y_pred : \n",y_pred)

plot.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='r') # X in Y_pred değeri 0 ise 0 yap, 0 ise 1 yap
plot.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='b')
plot.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='g')
plot.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='y')
plot.title("Hierarchical Clusters =4 ")
plot.show()

