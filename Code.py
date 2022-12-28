import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
sonar_data=pd.read_csv('/content/Copy of sonar data.csv',header=None)
print(sonar_data.head(),sonar_data.shape,sonar_data.describe(),sonar_data[60].value_counts(),sonar_data.groupby(60).mean(),)
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X_train.shape,X_test.shape,Y_test.shape,Y_train.shape)
model=LogisticRegression()
model.fit(X_train,Y_train)
X_train_pred=model.predict(X_train)
prediction_accuracy=accuracy_score(X_train_pred,Y_train)
X_test_pred=model.predict(X_test)
prediction_test_accuracy=accuracy_score(X_test_pred,Y_test)
print("ACCURACY SCORE : ",prediction_test_accuracy*100,"%")
X_test_pred=model.predict(X_test)
prediction_test_accuracy=accuracy_score(X_test_pred,Y_test)
print("ACCURACY SCORE ON TEST DATA : ",prediction_accuracy*100,"%")
pd.crosstab(Y_test, X_test_pred, rownames=['True'], 
colnames=['Predicted'],
margins=True)
input_data = (0.0123,0.0309,0.0169,0.0313,0.0358,0.0102,0.0182,0.0579,0.1122,0.0835,0.0548,0.0847,0.2026,0.2557,0.1870,0.2032,0.1463,0.2849,0.5824,0.7728,0.7852,0.8515,0.5312,0.3653,0.5973,0.8275,1.0000,0.8673,0.6301,0.4591,0.3940,0.2576,0.2817,0.2641,0.2757,0.2698,0.3994,0.4576,0.3940,0.2522,0.1782,0.1354,0.0516,0.0337,0.0894,0.0861,0.0872,0.0445,0.0134,0.0217,0.0188,0.0133,0.0265,0.0224,0.0074,0.0118,0.0026,0.0092,0.0009,0.0044)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock',"\U0001F48E")
else:
    print('The object is a Mine',"\U0001F4A5")
input_data = (0.0130,0.0120,0.0436,0.0624,0.0428,0.0349,0.0384,0.0446,0.1318,0.1375,0.2026,0.2389,0.2112,0.1444,0.0742,0.1533,0.3052,0.4116,0.5466,0.5933,0.6663,0.7333,0.7136,0.7014,0.7758,0.9137,0.9964,1.0000,0.8881,0.6585,0.2707,0.1746,0.2709,0.4853,0.7184,0.8209,0.7536,0.6496,0.4708,0.3482,0.3508,0.3181,0.3524,0.3659,0.2846,0.1714,0.0694,0.0303,0.0292,0.0116,0.0024,0.0084,0.0100,0.0018,0.0035,0.0058,0.0011,0.0009,0.0033,0.0026)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock',"\U0001F48E")
else:
    print('The object is a Mine',"\U0001F4A5")
import matplotlib.pyplot as plt
sonar_data[60].value_counts().plot(kind='bar')
sonar_data[60].value_counts().plot(kind='box')
sonar_data[60].value_counts().plot(kind='line')
