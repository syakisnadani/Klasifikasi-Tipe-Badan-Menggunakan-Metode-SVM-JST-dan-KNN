from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import pandas 


# Assign colum names to the dataset
names = ['tinggi', 'berat', 'leher', 'perut','pinggang', 'kategori']

# Read dataset to pandas dataframe
dataset = pd.read_csv('G:\Semester 5\Sistem Pengenalan Pola\Project Klasifikasi Bentuk Badan\Klasifikasi\KNN\KNN Penjelasan\dataset.csv', names=names)
dataset.head()
print(dataset)

#Pre-processing
x = dataset.iloc[:,  :  4].values
y = dataset.iloc[:,    -1].values

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

#feature scaling
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#training dan prediksi 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)