import pandas as pd
 
read_file = pd.read_excel ("G:\Semester 5\Sistem Pengenalan Pola\Project Klasifikasi Bentuk Badan\Klasifikasi\KNN\KNN Penjelasan\dataset.xlsx")
read_file.to_csv ("dataset.csv",
                  index = None,
                  header=True)
df = pd.DataFrame(pd.read_csv("dataset.csv"))
df
