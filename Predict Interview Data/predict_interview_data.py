# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
import matplotlib.pyplot as plt

def find_strongest_emotion(y_interview_pred, df_interview_all):
    
    # Create a dataframe with all single emotions that have been found out
    predicted_single_emotions = pd.DataFrame(labelencoder_y.inverse_transform(y_interview_pred), columns = ['predicted emotion'])
    predicted_single_emotions['file'] = df_interview_all.index
    predicted_single_emotions.set_index('file',inplace=True,drop=True)
    predicted_single_emotions = predicted_single_emotions.join(pd.DataFrame(df_interview_all['folder'], index = df_interview_all.index))
    
    emotion_count= predicted_single_emotions.groupby('folder')['predicted emotion'].value_counts().unstack().fillna(0)
    strongest_emotion = []
    strongness = []
    folders = []
    
    for i in range(len(emotion_count)):
        total = 0
        max_val = 0
        emotion = ''
        total = 0
        
        for j in range(len(emotion_count.columns)):
            total += emotion_count.iloc[i,j]
            if (emotion_count.iloc[i,j] > max_val):
                max_val = emotion_count.iloc[i,j]
                emotion = emotion_count.columns[j]
            elif (emotion_count.iloc[i,j] == max_val):
                emotion = emotion + ' or ' + emotion_count.columns[j]
            
        strongest_emotion.append(emotion)
        strongness.append(max_val / total)
        folders.append(emotion_count.index[i])
                
        
    del total, i, j, max_val, emotion

    return folders, strongest_emotion, strongness

# Open the actors training ARFF file containing the features from audio files
with open('actors.arff') as f:
    df_features = a2p.load(f)
    
df_features.set_index('name@STRING',inplace=True,drop=True)

# Open the annotation CSV file

df_emotions = pd.read_csv('annotation.csv', sep=';')
df_emotions.set_index('filenames',inplace=True,drop=True)

# Join dataset for emotions (dependent variable) with features
df_all = df_emotions.join(df_features, how = 'inner')

# Set independent variables (X) and dependent variable (y)
X = df_all.iloc[:, 1:-1].values
y = df_emotions.iloc[:, 0].values

del df_features
del df_emotions

# Encoding categorical data
# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
random_forest_classifier.fit(X, y)

# Fit Kernel SVM to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf')
svm_classifier.fit(X, y)

del X, y, df_all


# Open the subfiles ARFF file containing the features from audio files
with open('question_3.arff') as f:
    df_interview_features = a2p.load(f)
    
df_interview_features.set_index('name@STRING',inplace=True,drop=True)

# Open the annotation CSV file
df_interview_emotions = pd.read_csv('interviews_annotation.csv', sep=',')

# Open the subfiles CSV file
df_subfiles = pd.read_csv('subfiles.csv')
df_subfiles.iloc[:,1]
for i, i_value in enumerate(df_subfiles.iloc[:,1]):
    df_subfiles.iloc[i,1] = i_value + '.wav'
    
del i, i_value

df_interview_subfiles_emotions = pd.merge(left= df_interview_emotions, right= df_subfiles,
                                   left_index=False, right_index=False, 
                                   left_on= 'Answer_File', right_on= 'folder',
                                   how = 'right')

df_interview_subfiles_emotions = df_interview_subfiles_emotions.drop(['Unnamed: 0'],axis = 1)

df_interview_subfiles_emotions.set_index('file',inplace=True,drop=True)




    


# Join dataset for emotions (dependent variable) with features
df_interview_all = df_interview_subfiles_emotions.join(df_interview_features, how = 'inner')
del df_interview_features

#df_interview_emotions = df_interview_emotions.loc[df_interview_emotions['Answer_File'].isin(df_interview_all['folder'])]
df_interview_emotions.set_index('Answer_File',inplace=True,drop=True)

# Set independent variables (X) and dependent variable (y)
X_interviews = df_interview_all.iloc[:, 7:-1].values
y_interviews_content = df_interview_all.iloc[:, 4].values
y_interviews_voice = df_interview_all.iloc[:,5].values

# Encoding categorical data
# Encoding the dependent Variable
y_interviews_content = labelencoder_y.transform(y_interviews_content)
y_interviews_voice = labelencoder_y.transform(y_interviews_voice)


X_interviews = sc.transform(X_interviews)

# Count how many times a emotion has been found for each answer in random forest classifier
rf_folders, rf_strongest_emotion, rf_strongness = find_strongest_emotion(random_forest_classifier.predict(X_interviews), df_interview_all)
svm_folders, svm_strongest_emotion, svm_strongness = find_strongest_emotion(svm_classifier.predict(X_interviews), df_interview_all)

# create human readable table for our analysis      
pred_emotions = pd.DataFrame({'rf_strongest_emotion' : rf_strongest_emotion, 'rf_strongness': rf_strongness}, index = rf_folders)
del rf_strongness, rf_strongest_emotion
pred_emotions = pred_emotions.join(pd.DataFrame({'svm_strongest_emotion' : svm_strongest_emotion, 'svm_strongness': svm_strongness}, index = svm_folders))
pred_emotions = pred_emotions.join(pd.DataFrame({'Human_Content_Emotion':df_interview_emotions['Human_Content_Emotion'],'Human_Voice_Emotion': df_interview_emotions['Human_Voice_Emotion']},index=df_interview_emotions.index))


#Calculate accuracy
from sklearn.metrics import accuracy_score
#Predict content based rf results
print('rf_accuracy against content annotation: ' + str(accuracy_score(pred_emotions['Human_Content_Emotion'],pred_emotions['rf_strongest_emotion'])))
#Predict voice based rf results
print('rf_accuracy against voice annotation: ' + str(accuracy_score(pred_emotions['Human_Voice_Emotion'],pred_emotions['rf_strongest_emotion'])))
#Predict content based rf results
print('svm_accuracy against content annotation: ' + str(accuracy_score(pred_emotions['Human_Content_Emotion'],pred_emotions['svm_strongest_emotion'])))
#Predict voice based rf results
print('svm_accuracy against voice annotation: ' + str(accuracy_score(pred_emotions['Human_Voice_Emotion'],pred_emotions['svm_strongest_emotion'])))

pred_emotions.to_csv('predicted_emotions.csv')