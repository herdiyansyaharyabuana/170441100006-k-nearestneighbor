### Import librari terlebih dahulu
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

### Memanggil file csv dengan librari pandas
data_orj = pd.read_csv("wine_dataset.csv")

### Mengambil semua data untuk percobaan
data = data_orj.loc [:,'Alcohol':'Wine Type']

### Mengambil 3 kolom
data_knn = data[['Magnesium','Wine Type','Proline']]

### Input jumlah K atau jumlah tetangga terdekat
inK = int(input("Masukkan K : "))

### Mencari knn dengan menggunakan sklearn neighbors clssifier
knn = KNeighborsClassifier(n_neighbors = inK)  
x,y = data_knn.loc[:,data_knn.columns != 'Wine Type'], data_knn.loc[:,'Wine Type']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print('KNN dengan (K =',inK,') Akurasinya adalah: ', knn.score(x_test,y_test))
print("")

datatest = pd.DataFrame(x_test)
datatest["Wine Type"] = y_test
datatest["prediksi"] = prediction
print ("Data Asli dan Prediksinya")
print (datatest)

