# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
forest = ExtraTreesClassifier(n_estimators=100)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the 20 most important features of random forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(20),importances[indices[0:20]]*100,
#        color = "r", yerr=std[indices[0:20]],align="center")
            color = "r", align="center")
plt.xticks(range(20), indices[0:20])
plt.xlim([-1, 20])
plt.xlabel('feature number')
plt.ylabel('feature importance in %')



plt.show()