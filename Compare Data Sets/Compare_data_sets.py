# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

# Open the ouput ARFF file containing the features from audio files
with open('actors.arff') as f:
    df_actors_features = a2p.load(f)
    print(df_actors_features)
    
df_actors_features.set_index('name@STRING',inplace=True,drop=True)
df_actors_features = df_actors_features.iloc[:,:-1]
# Open the annotation CSV file

df_actors_emotions = pd.read_csv('annotation.csv', sep=';')
df_actors_emotions.set_index('filenames',inplace=True,drop=True)

# Join dataset for emotions (dependent variable) with features
df_actors_all = df_actors_emotions.join(df_actors_features, how = 'inner')


# Open the ouput ARFF file containing the features from audio files
with open('interviews.arff') as f:
    df_interview_features = a2p.load(f)
    print(df_interview_features)
    
df_interview_features.set_index('name@STRING',inplace=True,drop=True)
df_interview_features = df_interview_features.iloc[:,:-1]
# Open the annotation CSV file

df_interview_emotions = pd.read_csv('interviews_annotation.csv', sep=',')
df_interview_emotions.set_index('Answer_File',inplace=True,drop=True)
df_interview_emotions = df_interview_emotions.drop(["Student", "Country", "Degree","Human_Voice_Emotion"], axis=1)
df_interview_emotions = df_interview_emotions.rename(columns={"Human_Content_Emotion": "emotions"})

# Join dataset for emotions (dependent variable) with features
df_interview_all = df_interview_emotions.join(df_interview_features, how = 'inner')

df_actors_all['is_train'] = 1
df_interview_all['is_train'] = 0


#combining test and train data
df_combine = pd.concat([df_actors_all, df_interview_all], axis=0, ignore_index=True)

#Dropping features
df_combine = df_combine.drop('emotions',axis = 1)
colnames = df_combine.columns[[0,2,5,9,192,198,199,200,201]]
df_combine = df_combine.drop(colnames, axis = 1)




y = df_combine['is_train'].values #labels
x = df_combine.drop('is_train', axis=1).values #covariates or our independent variables
tst, trn = df_interview_all.values, df_actors_all.values

# Building and testing a classifier

m = RandomForestClassifier(n_estimators=100, n_jobs = -1,max_depth=5)
predictions = np.zeros(y.shape) #creating an empty prediction array

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=100)
for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
 X_train, X_test = x[train_idx], x[test_idx]
 y_train, y_test = y[train_idx], y[test_idx]
 
 m.fit(X_train, y_train)
 probs = m.predict_proba(X_test)[:, 1] #calculating the probability
 predictions[test_idx] = probs

print('ROC-AUC for train and test distributions:', roc_auc_score(y, predictions))

# Density Ratio Estimation

#plt.figure(figsize=(20,5))
predictions_train = predictions[:len(trn)] #filtering the actual training rows
predictions_test = predictions[len(trn):]
weights = (1./predictions_train) - 1.
weights /= np.mean(weights) # Normalizing the weights
#plt.xlabel('Computed sample weight')
#plt.ylabel('# Samples')
#sns.distplot(weights, kde=False)

m = RandomForestClassifier(n_jobs=-1,max_depth=5)
m.fit(X_train, y_train, sample_weight=weights)