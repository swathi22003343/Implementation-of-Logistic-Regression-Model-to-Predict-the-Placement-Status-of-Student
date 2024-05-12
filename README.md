# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:SWATHI D 
RegisterNumber:212222230154 
```
```

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
df1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y=df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:\n",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print("Classification Report:\n",classification_report1)
model.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## 1.Placement Data
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/46863473-6b16-4512-9d98-7cf57d0f6313)

## 2.Salary Data
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/afffd083-be30-4b93-892d-f3cb5e27baa8)

## 3. Checking the null function()
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/f68a87b5-1934-4d2b-bdc8-f3ad7cc95c85)

## 4.Data Duplicate
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/c1a64188-5f1c-4259-8a4f-8e4728221a06)

## 5.Print Data
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/c2c4c431-3b0e-4a42-ac20-d9d5519e3238)

## 6.Data Status
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/51fde253-852f-4de3-9746-b7fb1780f681)

## 7.y_prediction array
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/7be9b3d3-2903-417f-b9a5-192ef9035772)

## 8.Classification Report
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/4765cae3-f624-44c7-938e-8eb761e42a98)

## 9.Prediction of LR
![image](https://github.com/swathi22003343/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120440439/cedb3d64-4e5d-48e1-847e-88c2c1d8b75f)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
