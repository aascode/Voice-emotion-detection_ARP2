# Import the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np

# Open the ouput ARFF file containing the features from audio files
with open('output.arff') as f:
    df_features = a2p.load(f)
    print(df_features)
    
df_features.set_index('name@STRING',inplace=True,drop=True)

# Open the annotation CSV file

df_emotions = pd.read_csv('annotation.csv', sep=';')
df_emotions.set_index('filenames',inplace=True,drop=True)

# Join dataset for emotions (dependent variable) with features
df_all = df_emotions.join(df_features, how = 'inner')

# Set independent variables (X) and dependent variable (y)
X = df_all.iloc[:, 1:].values
y = df_emotions.iloc[:, 0].values

# Impute missing values with mean values to prevent errors due to missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# Split the dataset into the Training set and Test set (Test Size 15%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate accuracy
from sklearn.metrics import accuracy_score, average_precision_score
accuracy_score(y_test, y_pred)

