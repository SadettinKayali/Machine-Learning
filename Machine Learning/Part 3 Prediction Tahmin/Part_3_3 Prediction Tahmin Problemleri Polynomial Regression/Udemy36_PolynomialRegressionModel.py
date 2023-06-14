# -*- coding: utf-8 -*-
"""
36. Polinomal Regresyonun Python ile Uygulama kodu

"""

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# DATA PREPROCESSİNG (VERİ ÖN İŞLEME)
# DATA IMPORTING     (Veri Yükleme)

veriler=pd.read_csv('maaslar.csv')
print(veriler)
# DataFrame değerleri
x=veriler.iloc[:,1:2] # Eğitim Seviyesi
y=veriler.iloc[:,2:]
# Numpy dizisi değerleri
X=x.values
Y=y.values



# LINEAR REGRESSION Gösterimi
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(x,y) # Linear Model oluşturuluyor.

plot.scatter(x,y,color="red") # Scatter plotu çiziliyor önce
plot.plot(x,lin_reg.predict(x),color="blue") # X değerleri ile tahmin edilen x değerlerinin plot'u çiziliyor
plot.title("Simple Linear Regression")
plot.xlabel("diğer")
plot.ylabel("Maaşlar")
plot.show()

# POLYNOMIAL REGRESSION Gösterimi
from sklearn.preprocessing import PolynomialFeatures
# 2. dereceden
poly_reg=PolynomialFeatures(degree=2) # PolynomialFeatures kullanılarak 2.derecen 
x_poly=poly_reg.fit_transform(X)
print(" 2. Dereceden \n",x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plot.scatter(x,y,color="black")
plot.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="green")
plot.title(" 2.Degree Polynomial Regression")
plot.xlabel("diğer")
plot.ylabel("Maaşlar")
plot.show()
# 4. dereceden
poly_reg=PolynomialFeatures(degree=4) # PolynomialFeatures kullanılarak 4.derecen 
x_poly=poly_reg.fit_transform(X)
print(" 4. Dereceden \n",x_poly)
lin_reg4=LinearRegression()
lin_reg4.fit(x_poly,y)
plot.scatter(x,y,color="black")
plot.plot(x,lin_reg4.predict(poly_reg.fit_transform(x)),color="red")
plot.title(" 4.Degree Polynomial Regression")
plot.xlabel("diğer")
plot.ylabel("Maaşlar")
plot.show()

# Tek seferde PLOT gösterimi 
# 1.Dereceden
lin_reg=LinearRegression()
lin_reg.fit(x,y) # Linear Model oluşturuluyor.
plot.scatter(x,y,color="black") # Scatter plotu çiziliyor önce
plot.plot(x,lin_reg.predict(x),color="red")
# 2. Dereceden
poly_reg2=PolynomialFeatures(degree=2) # PolynomialFeatures kullanılarak 2.derecen 
x_poly=poly_reg2.fit_transform(X)
print(" 2. Dereceden \n",x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plot.scatter(x,y,color="black")
plot.plot(x,lin_reg2.predict(poly_reg2.fit_transform(x)),color="orange")
# 4. Dereceden
poly_reg4=PolynomialFeatures(degree=4) # PolynomialFeatures kullanılarak 4.derecen 
x_poly=poly_reg4.fit_transform(X)
print(" 4. Dereceden \n",x_poly)
lin_reg4=LinearRegression()
lin_reg4.fit(x_poly,y)
plot.scatter(x,y,color="black")
plot.plot(x,lin_reg4.predict(poly_reg4.fit_transform(x)),color="green")
plot.title(" Regression")
plot.xlabel("diğer")
plot.ylabel("Maaşlar")
plot.show()

# Tahminler



print(" 1. Dereceden Tahmin (11) : \n ",lin_reg.predict([[11]]))    # 11. değerdeki eğitim derecesine göre maaş tahmini
print(" 1. Dereceden Tahmin (6.6) : \n ",lin_reg.predict([[6.6]]))


print(" 2. Dereceden Tahmin (11) : \n ",lin_reg2.predict(poly_reg2.fit_transform([[11]])))
print(" 2. Dereceden Tahmin (6.6) : \n ",lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))

print(" 4. Dereceden Tahmin (11) : \n ",lin_reg4.predict(poly_reg4.fit_transform([[11]])))
print(" 4. Dereceden Tahmin (6.6) : \n ",lin_reg4.predict(poly_reg4.fit_transform([[6.6]])))





























