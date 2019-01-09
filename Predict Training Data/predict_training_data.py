# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Open the acotrs ARFF file containing the features from audio files
with open('actors.arff') as f:
    df_features = a2p.load(f)
    print(df_features)
    
df_features.set_index('name@STRING',inplace=True,drop=True)

# Open the annotation CSV file
df_emotions = pd.read_csv('annotation.csv', sep=';')
df_emotions.set_index('filenames',inplace=True,drop=True)

# Join dataset for emotions (dependent variable) with features
df_all = df_emotions.join(df_features, how = 'inner')

# Set independent variables (X) and dependent variable (y)
X = df_all.iloc[:, 1:-1].values
y = df_emotions.iloc[:, 0].values

# Repeat Fitting and prediction to show consistency of values
svc_accuracy = []
rf_accuracy = []
lr_accuracy = []
knn_accuracy = []
nb_accuracy = []

# 100 times, each model will be fitted with different parts of the training data
# this shows the variance and median of accuracies of accuracy that can be achieved
# despite the variance due to probabilistic modelling and use of different training data

for i in range(100):
    
    # Split the dataset into the Training set and Test set (Test Size 15%)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    
    # Fit Kernel SVM to the Training set
    svc_classifier = SVC(kernel = 'rbf')
    svc_classifier.fit(X_train, y_train)
    
    # Fitting Random Forest Classification to the Training set
    rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    rf_classifier.fit(X_train, y_train)
    
    
    # Fit Logistic Regression to the Training Set
    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train, y_train)
    
    # Fitting K-NN to the Training set
    knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn_classifier.fit(X_train, y_train)
    
    # Fitting Gaussian Naive Bayes to the Training set
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    
    # Predict the Test set results
    svc_y_pred = svc_classifier.predict(X_test)
    rf_y_pred = rf_classifier.predict(X_test)
    lr_y_pred = lr_classifier.predict(X_test)
    knn_y_pred = knn_classifier.predict(X_test)
    nb_y_pred = nb_classifier.predict(X_test)
    
    svc_accuracy.append(accuracy_score(y_test, svc_y_pred))
    rf_accuracy.append(accuracy_score(y_test, rf_y_pred))
    lr_accuracy.append(accuracy_score(y_test, lr_y_pred))
    knn_accuracy.append(accuracy_score(y_test, knn_y_pred))
    nb_accuracy.append(accuracy_score(y_test, nb_y_pred))
    
df_svc = pd.DataFrame({'classifier' : 'SVC', 'accuracy' : svc_accuracy})
df_rf = pd.DataFrame({'classifier' : 'RF', 'accuracy' : rf_accuracy})
df_lr = pd.DataFrame({'classifier' : 'LR', 'accuracy' : lr_accuracy})
df_knn = pd.DataFrame({'classifier' : 'KNN', 'accuracy' : knn_accuracy})
df_nb = pd.DataFrame({'classifier' : 'NB', 'accuracy' : knn_accuracy})

df_accuracies = df_svc.append(df_rf, ignore_index=True)
df_accuracies = df_accuracies.append(df_lr, ignore_index = True)
df_accuracies = df_accuracies.append(df_knn, ignore_index = True)
df_accuracies = df_accuracies.append(df_nb, ignore_index = True)

df_accuracies.to_csv('accuracies_training_data.csv')


sns.boxplot(x = 'classifier', y = 'accuracy', data = df_accuracies)
