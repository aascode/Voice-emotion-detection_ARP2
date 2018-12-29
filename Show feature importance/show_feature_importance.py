# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
import matplotlib.pyplot as plt

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


# Build a forest and compute the feature importances
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=100,
                              random_state=0)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()