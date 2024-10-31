import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = r"C:\Users\Azat Titiz\Downloads\Advertising.csv"
dataset = pd.read_csv(path, sep=",")
print(dataset)  # veri başarı ile çekildi
X = dataset.iloc[:, :-1]  # bağımsız değişkenler
y = dataset.iloc[:, -1]  # bağımlı değişkenler

# Verileri dönüştür
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = ct.fit_transform(X).toarray()  # X'i seyrek matrisden düzgün bir NumPy dizisine çevir

print("***** feature scaling ********")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("X_Train",X_train)
print("*************")
print(X_test)
print("*************")
print(y_train)
print("*************")
print(y_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)

print("***** TEST SONUÇLARININ GÖRSELLEŞTİRİLMESİ *******")
plt.scatter(X_train[:, 0], y_train, edgecolors="red")  # X_train'in birinci sütununu kullanabilirsiniz
plt.plot(X_train[:, 0], lr.predict(X_train), color="blue")  # Aynı sütunu kullanarak tahminleri çizin
plt.title("Grafik")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()