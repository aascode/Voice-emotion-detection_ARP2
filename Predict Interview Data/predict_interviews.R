rm(list=ls(all=TRUE))
library('e1071')  
library('tidyverse')
# Open the ouput ARFF file containing the features from audio files
library(foreign)



df_features = read.arff("actors.arff")
df_features<- df_features[-c(183,186,386)]

# Open the annotation CSV file
dataAnno = read.csv("annotation.csv",sep = ';', header = TRUE)
df_emotions <- data.frame(dataAnno)
View(df[1:10])
head(df)

# Joined dataset for emotions (dependent variable) with features

#df_joined = read.csv("joined_emotion.csv", header = TRUE)

df_joined <- merge(x = df_emotions, y = df_features, by.x = 'filenames', by.y = 'name')


head(df_joined)
#df_joined$


#df_features= read.csv("C:/Users/User/Documents/output_features.csv", header = TRUE)
df_features = read.arff("actors.arff")
df_features1 = read.arff("question_1.arff")
df_features2 = read.arff("question_2.arff")
df_features3 = read.arff("question_3.arff")
df_features4 = read.arff("question_4.arff")
df_features5 = read.arff("question_5.arff")

df_interview_features <- rbind(df_features1,df_features2,df_features3,df_features4,df_features5)
df_interview_features <- df_interview_features[-c(183,186,386)]

# Open the interviews annotation CSV file
df_interviews_emotions <- read.csv('interviews_annotation.csv', sep=',')

# Open the subfiles CSV file

df_subfiles = data.frame(X=list(),folder=list(),file=list())

for (question_no in 1:5){
  df_subfiles <- rbind(df_subfiles,read.csv(paste('question_',question_no,'_subfiles.csv', sep= '')))

}

#df_subfiles <- distinct(df_subfiles,file)

df_subfiles$folder <- paste(df_subfiles$folder,'.wav',sep='')



df_interview_subfiles_emotions <- merge(x= df_interviews_emotions,
                                        y= df_subfiles,
                                        by.x= 'Answer_File',
                                        by.y= 'folder',
                                        all.y = TRUE)

df_interview_all <- merge(x = df_interview_subfiles_emotions,
                          y = df_interview_features,
                          by.x = 'file',
                          by.y = 'name'
                          )

# Set independent variables (X) and dependent variable (y)

X_interviews <- df_interview_all[, 9:390]
y_interviews_content <- df_interview_all[, 6]
y_interviews_voice <- df_interview_all[,7]


training_set <- df_joined

# Feature Scaling
#dfNormZ <- as.data.frame( scale(training_set[3:]))
training_set[3:384] = scale(training_set[3:384])
df_interview_all[, 9:390] = scale(df_interview_all[, 9:390])



#fitting kernel svm

classifier_svm = svm(formula = as.factor(training_set$emotions)  ~ .,
                 data = training_set[-c(1,2)],
                 type = 'C-classification',
                 kernel = 'radial')


# Fitting Random Forest Classification to the Training set
library(randomForest)
set.seed(123)
classifier_rf = randomForest(x = training_set[-c(1,2)],
                          y = as.factor(training_set$emotions),
                          ntree = 100)

# Predicting the Test set results
y_pred_svm <- predict(classifier_svm, newdata = df_interview_all[, 9:390])
y_pred_rf <- predict(classifier_rf, newdata = df_interview_all[, 9:390])

#### Finding strongest emotion ######

# As we have perform the model fitting multiple times, all predictions need to first be stored
# into the tibble below to find the most frequent prediction

#all_single_emotions <- tibble(predicted_emotion = list(),file = list(),folder = list())


  
  #The label encording has to be reverted to make the results readable
 # predicted_single_emotions = pd.DataFrame(labelencoder_y.inverse_transform(y_interview_pred[i]), columns = ['predicted emotion'])
#predicted_single_emotions['file'] = df_interview_all.index

# The subfile records hve to be extended by the answer folder that they belong to
predicted_single_emotions <- data.frame('file' = df_interview_all$file,
                                        'predicted_emotion' =y_pred_rf )

predicted_single_emotions <- merge(x = predicted_single_emotions,
                                   y = data.frame(file = df_interview_all$file,
                                                  answer_file = df_interview_all$Answer_File),
                                   by = 'file') 


  
emotion_count <- count(predicted_single_emotions,answer_file,predicted_emotion)

emotion_count <- spread(data = emotion_count,
                        key = predicted_emotion,
                        value = n)

emotion_count[is.na(emotion_count)] <- 0

rf_emotions <- data.frame(folder = list(),
                         strongest_emotion = list(),
                         strength = list())

for (i in 1:length(emotion_count$answer_file)){
  total <- 0
  max_val <- 0
  emotion <- ''

  
  for (j in 2:length(emotion_count)){
    total <- total + emotion_count[i,j]
    if (emotion_count[i,j] > max_val){
      max_val <- emotion_count[i,j]
      emotion <- colnames(emotion_count)[j]
    } else {
        if (emotion_count[i,j] == max_val){
          emotion <- paste(emotion,'or',colnames(emotion_count)[j])
        }
    }
  }
  
  tmp <- data.frame(folder = emotion_count[i,1],
                    strongest_emotion = emotion,
                    strength = (max_val / total))
  
  names(tmp) <- names(rf_emotions) 
  
  rf_emotions <- rbind(rf_emotions, tmp)
}

names(rf_emotions) <- c('folder','strongest_emotion','strength')


write.csv(rf_emotions,file ='r_random_forest_emotion_predictions.csv')

# Making the Confusion Matrix
cm_svm = table(y_pred_svm,test_set[,2])
cm_rf = table(y_pred_rf,test_set[,2])

#apply(test_set[-2], 2, function(x) any(is.na(x)))
#install.packages('rfUtilities')
library(rfUtilities)
accuracy(y_pred_svm, test_set[,2])
accuracy(y_pred_rf, test_set[,2])

################ Find na values ############################


for (i in 1:length(df_features1[,-1])){
  for (j in 1:length(is.na(df_features1[i]))){
    if (is.na(test_set[i])[j]){
      print(paste('row',j,'column',i,' is na'))
    }
  }
}
