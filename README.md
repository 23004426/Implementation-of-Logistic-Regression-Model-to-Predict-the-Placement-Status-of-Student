# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Tirupathi Jayadeep
RegisterNumber: 212223240169 
*/

import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"]= le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
c_report = classification_report(y_test,y_pred)
c_report

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
# Head
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/eefa62b7-f638-4ca7-a02f-46fc2e53cdec)

# After removing sl_no , salary
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/be21cb66-40f2-485d-936a-5105265df776)

# Null data
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/33431ddf-c114-4f35-a73f-34aef59c2abf)

# Duplicated sum
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/4edbb4d0-ec2e-4e45-a62b-4e32e72a5c5e)

# Label Encoder
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/c813c224-8210-4868-bb71-21b3dd89a987)

# After removing the last column
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/ccc79230-9ad7-4554-9b58-7e233550d787)

# Displaying the status
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/8c9554ab-a951-4f27-88b1-f133bb828fb1)

# Prediction of y
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/b1681be5-0496-4a4d-8f23-1cdc35f73079)

# Accuracy score
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/30d6b905-4409-433d-8b79-edb7b5215df0)

# Confusion
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/56d033b3-e691-492e-a2e8-ea641518f60b)

# Classification report 
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/a710f9c6-f40b-4588-9971-ba75d3ed0704)

# Prediction
![image](https://github.com/23004426/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144979327/42e740b2-8e40-489a-b04c-50e4b76ce393)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
