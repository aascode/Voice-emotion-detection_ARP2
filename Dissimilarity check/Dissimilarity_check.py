# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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


df_interview_all = pd.DataFrame(columns=df_actors_all.columns)

for question_no in range(1, 6):


    # Open the subfiles ARFF file containing the features from audio files
    with open('question_' + str(question_no) + '.arff') as f:
        df_interview_features = a2p.load(f)
    
    df_interview_features.set_index('name@STRING',inplace=True,drop=True)
    df_interview_features = df_interview_features.iloc[:, range(len(df_interview_features.columns)-1)]
    
    # Open the annotation CSV file
    df_interview_emotions = pd.read_csv('interviews_annotation.csv', sep=',')
    
    # Open the subfiles CSV file
    df_subfiles = pd.read_csv('question_'+ str(question_no) + '_subfiles.csv')
    df_subfiles.iloc[:,1]
    for i, i_value in enumerate(df_subfiles.iloc[:,1]):
        df_subfiles.iloc[i,1] = i_value + '.wav'
        
    del i, i_value    
    
    # Merge the emotion classification subfiles list with their belonging to one of the interviewees
    df_interview_subfiles_emotions = pd.merge(left= df_interview_emotions, right= df_subfiles,
                                       left_index=False, right_index=False, 
                                       left_on= 'Answer_File', right_on= 'folder',
                                       how = 'right')
    
    df_interview_subfiles_emotions = df_interview_subfiles_emotions.drop(['Unnamed: 0'],axis = 1)
    df_interview_subfiles_emotions = df_interview_subfiles_emotions.drop(["Student", "Country", "Degree","Human_Voice_Emotion", "Answer_File","folder"], axis=1)
    df_interview_subfiles_emotions = df_interview_subfiles_emotions.rename(columns={"Human_Content_Emotion": "emotions"})
    
    
    df_interview_subfiles_emotions.set_index('file',inplace=True,drop=True)
    
    # Join dataset for emotions (dependent variable) with features
    df_interview_all = df_interview_all.append(df_interview_subfiles_emotions.join(df_interview_features, how = 'inner'))
    
# Taking care of missing data
from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(df_interview_all)
#df_interview_all = imputer.transform(df_interview_all)

#adding a column to identify whether a row comes from train or not

df_actors_all['is_train'] = 1
df_interview_all['is_train'] = 0



# combining test and train data
df_combine = pd.concat([df_actors_all, df_interview_all], axis=0, ignore_index=True)

# Dropping dependent variable as it is not present in the test
df_combine = df_combine.drop('emotions',axis = 1)
#df_combine = df_combine.drop(df_combine.columns[197:199],axis = 1)
##### Building and testing a classifer ######


y = df_combine['is_train'].values #labels
x = df_combine.drop('is_train', axis=1).values #covariates or our independent variables
tst, trn = df_interview_all.values, df_actors_all.values


m = tree.DecisionTreeClassifier()
#m = RandomForestClassifier(n_estimators=100, n_jobs = -1,max_depth=5)
predictions = np.zeros(y.shape) #creating an empty prediction array

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=100)
for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
 X_train, X_test = x[train_idx], x[test_idx]
 y_train, y_test = y[train_idx], y[test_idx]
 
 m.fit(X_train, y_train)
 probs = m.predict_proba(X_test)[:, 1] #calculating the probability
 predictions[test_idx] = probs

# Building and testing a classifier

print('ROC-AUC for train and test distributions:', roc_auc_score(y, predictions))

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
s


dot_data = StringIO()

export_graphviz(m, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=df_combine.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
