# Importing the libraries

import pandas as pd
from arff2pandas import a2p
import numpy as np

with open('output.arff') as f:
    df_features = a2p.load(f)
    print(df_features)
    

df_features.set_index('name@STRING',inplace=True,drop=True)

df_emotions = pd.read_csv('annotation.csv', sep=';')
df_emotions.set_index('filenames',inplace=True,drop=True)


#Include dataset load for emotions (dependent variable)
df_all = df_emotions.join(df_features, how = 'inner')



# Importing the dataset
X = df_all.iloc[:, 1:].values
y = df_emotions.iloc[:, 0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, average_precision_score

accuracy_score(y_test, y_pred)

