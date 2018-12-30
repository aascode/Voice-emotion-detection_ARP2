# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np

# Open the ouput ARFF file containing the features from audio files
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

# Encoding categorical data
# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Impute missing values with mean values to prevent errors due to missing values
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X)
# X = imputer.transform(X)

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

k_best_selector = SelectKBest(f_classif, k=15).fit(X_train, y_train)
X_train = k_best_selector.transform(X_train)
X_test = k_best_selector.transform(X_test)

classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate accuracy
from sklearn.metrics import accuracy_score, average_precision_score
accuracy_score(y_test, y_pred)

# Open the ouput ARFF file containing the features from audio files
with open('interviews.arff') as f:
    df_interview_features = a2p.load(f)
    print(df_interview_features)
    
df_interview_features.set_index('name@STRING',inplace=True,drop=True)

# Open the annotation CSV file

df_interview_emotions = pd.read_csv('interviews_annotation.csv', sep=',')
df_interview_emotions.set_index('Answer_File',inplace=True,drop=True)

# Join dataset for emotions (dependent variable) with features
df_interview_all = df_interview_emotions.join(df_interview_features, how = 'inner')

# Set independent variables (X) and dependent variable (y)
X_interviews = df_interview_all.iloc[:, 5:-1].values
y_interviews_content = df_interview_all.iloc[:, 3].values
y_interviews_voice = df_interview_all.iloc[:,4].values

# Encoding categorical data
# Encoding the dependent Variable
y_interviews_content = labelencoder_y.transform(y_interviews_content)
y_interviews_voice = labelencoder_y.transform(y_interviews_voice)


X_interviews = sc.transform(X_interviews)

#K-Best Selection
X_interviews = k_best_selector.transform(X_interviews)


# Predict the interview results
y_interview_pred = classifier.predict(X_interviews)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_interviews_voice, y_interview_pred)


#Calculate accuracy
from sklearn.metrics import accuracy_score
#Predict content based results
accuracy_score(y_interviews_content, y_interview_pred)
#Predict voice based results
accuracy_score(y_interviews_voice, y_interview_pred)

emotions_found = labelencoder_y.inverse_transform(y_interview_pred)
pred_emotions = pd.DataFrame(emotions_found)
pred_emotions['file'] = df_interview_all.index
pred_emotions['team_estimate'] = labelencoder_y.inverse_transform(y_interviews_voice)
pred_emotions['equal'] = (y_interview_pred == y_interviews_voice)
