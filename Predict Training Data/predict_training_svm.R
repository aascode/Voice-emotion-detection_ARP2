rm(list=ls(all=TRUE))
library('e1071')  



# Open the ouput ARFF file containing the features from audio files
library(foreign)
#df_features= read.csv("C:/Users/User/Documents/output_features.csv", header = TRUE)
df_features = read.arff("actors.arff")

head(df_features)

# Open the annotation CSV file
dataAnno = read.csv("annotation.csv",sep = ';', header = TRUE)
df_emotions <- data.frame(dataAnno)
View(df[1:10])
head(df)

# Joined dataset for emotions (dependent variable) with features

#df_joined = read.csv("joined_emotion.csv", header = TRUE)

df_joined <- merge(x = df_emotions, y = df_features, by.x = 'filenames', by.y = 'name')
df_joined<- df_joined[-387]

head(df_joined)
#df_joined$


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df_joined$emotions, SplitRatio = 0.85)
training_set = subset(df_joined, split == TRUE)
test_set = subset(df_joined, split == FALSE)

# Feature Scaling
training_set$X
#dfNormZ <- as.data.frame( scale(training_set[3:]))
training_set[3:386] = scale(training_set[3:386])
test_set[3:386] = scale(test_set[3:386])
#training_set<-training_set[,-(100:386)]
#training_set[1:20,5]
#training_set<- training_set[,(-4:386)]
#summary(training_set)
#head(training_set)
#-training_set$class.NUMERIC- training_set$X

#fitting kernel svm
training_set<-training_set[,-184]
test_set<-test_set[-184]
classifier_svm = svm(formula = training_set$emotions  ~ .,
                 data = training_set[-c(1,2)],
                 type = 'C-classification',
                 kernel = 'radial')


# Fitting Random Forest Classification to the Training set
library(randomForest)
set.seed(123)
classifier_rf = randomForest(x = training_set[-c(1,2)],
                          y = training_set$emotions,
                          ntree = 100)


# Predicting the Test set results
y_pred_svm <- predict(classifier_svm, newdata = test_set[-c(1,2)])
y_pred_rf <- predict(classifier_rf, newdata = test_set[-c(1,2)])


# Making the Confusion Matrix
cm_svm = table(y_pred_svm,test_set[,2])
cm_rf = table(y_pred_rf,test_set[,2])

#apply(test_set[-2], 2, function(x) any(is.na(x)))
#install.packages('rfUtilities')
library(rfUtilities)
accuracy(y_pred_svm, test_set[,2])
accuracy(y_pred_rf, test_set[,2])
