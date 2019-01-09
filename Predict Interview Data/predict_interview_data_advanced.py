# Importing the libraries
import pandas as pd
from arff2pandas import a2p
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Go through all predictions of the audio subfiles and find out which of them was predicted most often
def find_strongest_emotion(y_interview_pred, df_interview_all):
    
    # As we have perform the model fitting multiple times, all predictions need to first be stored
    # into the dataframe below to find the most frequent prediction
    all_single_emotions = pd.DataFrame(columns = ['predicted emotion','file', 'folder'])
    
    # For every instance of model fitting, the single predictions are sextracted
    for i in range(len(y_interview_pred)):
        
        #The label encording has to be reverted to make the results readable
        predicted_single_emotions = pd.DataFrame(labelencoder_y.inverse_transform(y_interview_pred[i]), columns = ['predicted emotion'])
        predicted_single_emotions['file'] = df_interview_all.index
        
        # The subfile records hve to be extended by the answer folder that they belong to
        predicted_single_emotions = predicted_single_emotions.merge(right=pd.DataFrame({'folder' : df_interview_all['folder'], 'file': df_interview_all.index}),
                                                                    left_on = 'file', right_on = 'file')
        
        all_single_emotions = all_single_emotions.append(predicted_single_emotions)
        
    # Count the frequency of each emotion per folder (answer) that they belong to
    emotion_count= all_single_emotions.groupby('folder')['predicted emotion'].value_counts().unstack().fillna(0)
    
    strongest_emotion = []
    strongness = []
    folders = []
    
    # This double for loop will loop through every emotion and look for the one, for which the count is the highest
    # Also, all counts are summed, so that the accuracy strength can be calculated
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




for question_no in range(1,6):

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
    
    # Open the subfiles ARFF file containing the features from audio files
    with open('question_' + str(question_no) + '.arff') as f:
        df_interview_features = a2p.load(f)
    
    df_interview_features.set_index('name@STRING',inplace=True,drop=True)
    
    # Open the annotation CSV file
    df_interview_emotions = pd.read_csv('interviews_annotation.csv', sep=',')
    
    # Open the subfiles CSV file
    df_subfiles = pd.read_csv('question_'+ str(question_no) + '_subfiles.csv')
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
    
    rf_interview_pred = []
    svc_interview_pred = []
    lr_interview_pred = []
    knn_interview_pred = []
    nb_interview_pred = []
    
    
    for i in range(50):
        
        # Fitting Random Forest Classification to the Training set
        
        rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
        rf_classifier.fit(X, y)
        
        # Fit Kernel SVM to the Training set
        svc_classifier = SVC(kernel = 'rbf')
        svc_classifier.fit(X, y)
        
        # Fit Logistic Regression to the Training Set
        lr_classifier = LogisticRegression()
        lr_classifier.fit(X, y)
        
        # Fitting K-NN to the Training set
        knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        knn_classifier.fit(X, y)
        
        # Fitting Gaussian Naive Bayes to the Training set
        nb_classifier = GaussianNB()
        nb_classifier.fit(X, y)
    
        rf_interview_pred.append(rf_classifier.predict(X_interviews))
        svc_interview_pred.append(svc_classifier.predict(X_interviews))
        lr_interview_pred.append(lr_classifier.predict(X_interviews))
        knn_interview_pred.append(knn_classifier.predict(X_interviews))
        nb_interview_pred.append(nb_classifier.predict(X_interviews))
    
    #del X, y, df_all
    
    
    
    # Count how many times a emotion has been found for each answer in random forest classifier
    
    rf_folders, rf_strongest_emotion, rf_strongness = find_strongest_emotion(rf_interview_pred, df_interview_all)
    svc_folders, svc_strongest_emotion, svc_strongness = find_strongest_emotion(svc_interview_pred, df_interview_all)
    lr_folders, lr_strongest_emotion, lr_strongness = find_strongest_emotion(lr_interview_pred, df_interview_all)
    knn_folders, knn_strongest_emotion, knn_strongness = find_strongest_emotion(knn_interview_pred, df_interview_all)
    nb_folders, nb_strongest_emotion, nb_strongness = find_strongest_emotion(nb_interview_pred, df_interview_all)
    
    
    # create human readable table for our analysis      
    pred_emotions = pd.DataFrame({'rf_strongest_emotion' : rf_strongest_emotion, 'rf_strongness': rf_strongness}, index = rf_folders)
    del rf_strongness, rf_strongest_emotion
    pred_emotions = pred_emotions.join(pd.DataFrame({'svc_strongest_emotion' : svc_strongest_emotion, 'svc_strongness': svc_strongness}, index = svc_folders))
    pred_emotions = pred_emotions.join(pd.DataFrame({'lr_strongest_emotion' : lr_strongest_emotion, 'lr_strongness': lr_strongness}, index = lr_folders))
    pred_emotions = pred_emotions.join(pd.DataFrame({'knn_strongest_emotion' : knn_strongest_emotion, 'knn_strongness': knn_strongness}, index = knn_folders))
    pred_emotions = pred_emotions.join(pd.DataFrame({'nb_strongest_emotion' : nb_strongest_emotion, 'nb_strongness': nb_strongness}, index = nb_folders))
    pred_emotions = pred_emotions.join(pd.DataFrame({'Human_Content_Emotion':df_interview_emotions['Human_Content_Emotion'],'Human_Voice_Emotion': df_interview_emotions['Human_Voice_Emotion']},index=df_interview_emotions.index))
    
    
    #Calculate accuracy
    from sklearn.metrics import accuracy_score
    #Predict content based rf results
    #print('rf_accuracy against content annotation: ' + str(accuracy_score(pred_emotions['Human_Content_Emotion'],pred_emotions['rf_strongest_emotion'])))
    #Predict voice based rf results
    #print('rf_accuracy against voice annotation: ' + str(accuracy_score(pred_emotions['Human_Voice_Emotion'],pred_emotions['rf_strongest_emotion'])))
    #Predict content based rf results
    #print('svm_accuracy against content annotation: ' + str(accuracy_score(pred_emotions['Human_Content_Emotion'],pred_emotions['svm_strongest_emotion'])))
    #Predict voice based rf results
    #print('svm_accuracy against voice annotation: ' + str(accuracy_score(pred_emotions['Human_Voice_Emotion'],pred_emotions['svm_strongest_emotion'])))
    
    pred_emotions.to_csv('predicted_emotions_question_'+ str(question_no) + '_advanced.csv')