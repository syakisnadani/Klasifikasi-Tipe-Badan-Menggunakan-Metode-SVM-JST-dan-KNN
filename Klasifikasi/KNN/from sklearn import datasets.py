import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('G:\Semester 5\Sistem Pengenalan Pola\Project Klasifikasi Bentuk Badan\Klasifikasi\KNN\KNN Penjelasan\diabetes.csv')
len(dataset)
dataset.head()




