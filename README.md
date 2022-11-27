# Implementation of SVM For Spam Mail Detection

# AIM:
To write a program to implement the SVM For Spam Mail Detection.

# EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# ALGORITHM:
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

# PROGRAM:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SITHI HAJARA I
Register Number: 212221230102
*/
```

```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

# OUTPUT:
![image](https://user-images.githubusercontent.com/94219582/204132780-893d7743-a37b-4105-8b6e-5ecca6da742f.png)
![image](https://user-images.githubusercontent.com/94219582/204132799-d87de440-57e2-4845-81ec-6ec8f427cb43.png)
![image](https://user-images.githubusercontent.com/94219582/204132812-c88d32f4-d086-4180-a7f7-46e05e47feba.png)

# RESULT:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
