import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = rpath = r"C:\Users\Azat Titiz\Desktop\03-PolynomialRegression\03-PolynomialRegression\kalite-fiyat.csv"
dataset = pd.read_csv(path, sep=",")
print(dataset)  # veri başarı ile çekildi
X = dataset.iloc[:, :-1]  # bağımsız değişkenler
y = dataset.iloc[:, -1]  # bağımlı değişkenler
print("X",X)
print("y",y)
lr = LinearRegression() # linear regression ne yapar
lr.fit(X,y)



print("polynominial Regression modelini öğrenmesi")

pol_reg = PolynomialFeatures(degree=5)
X_Pol =pol_reg.fit_transform(X)
print(X_Pol)
lr2 = LinearRegression()
lr2.fit(X_Pol,y)
plt.scatter(X, y, color="red")  # X_train'in birinci sütununu kullanabilirsiniz
plt.plot(X, lr2.predict(X_Pol), color="blue")  # Aynı sütunu kullanarak tahminleri çizin
plt.title("fiyat-Kalite")
plt.xlabel("kalite")
plt.ylabel("fiyat")
plt.show()
