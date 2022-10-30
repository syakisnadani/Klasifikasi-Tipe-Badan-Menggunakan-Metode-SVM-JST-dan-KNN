import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from numpy import array
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# Proses Input Dataset
path = r"G:\Semester 5\Sistem Pengenalan Pola\Project Klasifikasi Bentuk Badan\Klasifikasi\KNN\KNN Penjelasan\dataset.csv"
data = pd.read_csv(path)
print(data)

#Data Scaling
x = data[data.columns[:5]]
y = data['kelas']
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print ('----------- Data Scaling ----------')
print (x)

#Data Split
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# KNN Model
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('[[prediction KNN]] ')
print(' prediction: ', prediction)
print('accuracy: ', accuracy)

#JST
modelJST = MLPClassifier(alpha=0.0001,hidden_layer_sizes=(15,), max_iter=2000)
modelJST.fit(x_train, y_train)
predictionJST = modelJST.predict(x_test)
accuracyJST = metrics.accuracy_score(y_test, predictionJST)
print('[[prediction JST]] ')
print(' prediction: ', predictionJST)
print(' actual : ', y_test)
print('accuracy: ', accuracyJST)


#SVM
modelSVM = svm.SVC()
modelSVM.fit(x_train, y_train)
predictionSVM = modelSVM.predict(x_test)
accuracySVM = metrics.accuracy_score(y_test, predictionSVM)
print('[[prediction SVM]] ')
print(' prediction: ', predictionSVM)
print(' actual : ', y_test)
print('accuracy: ', accuracySVM)