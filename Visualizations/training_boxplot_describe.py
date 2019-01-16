import pandas as pd
import seaborn as sns
import numpy as np

df_accuracies = pd.read_csv('accuracies_training_data.csv')

df_accuracies['accuracy'] = df_accuracies['accuracy'] * 100

df_accuracies = df_accuracies.drop('Unnamed: 0', axis = 1)

for i in range(len(df_accuracies)):
    if df_accuracies.iloc[i,0] == 'Naive Bayes':
        df_accuracies.iloc[i,0] = 'NB'
    if df_accuracies.iloc[i,0] == 'Random Forest':
        df_accuracies.iloc[i,0] = 'RF'
    if df_accuracies.iloc[i,0] == 'K-Nearest Neighbor':
        df_accuracies.iloc[i,0] = 'KNN'
    if df_accuracies.iloc[i,0] == 'Logistic Regression':
        df_accuracies.iloc[i,0] = 'LR'
        
        

sns.boxplot(x = 'classifier', y = 'accuracy' , data = df_accuracies).set(
    xlabel='Classifier', 
    ylabel='Accuracy in %'
)



df_accuracies.groupby('classifier').accuracy.describe().unstack()

df_accuracies.groupby('classifier').accuracy.median()