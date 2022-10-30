from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import metrics


# Assign colum names to the dataset
names = ['Fitur1', 'Fitur2', 'Fitur3', 'Fitur4','Fitur5', 'Kategori']

# Read dataset to pandas dataframe
dataset = pd.read_csv('G:\Semester 5\Sistem Pengenalan Pola\Project Klasifikasi Bentuk Badan\Klasifikasi\KNN\KNN Penjelasan\dataset.csv', names=names)
dataset.head()
print(dataset)

#Pre-processing
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#train test split
iris = datasets.load_iris()
# split
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#model
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print(' prediction: ', prediction)
print('accuracy: ', accuracy)

