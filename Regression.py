# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:19:14 2021

@author: RT
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
import joblib
from joblib import dump, load
# importing diabetes dataset
diabetesDF = pd.read_csv('diabetes.csv')

#printing head of dataset
print(diabetesDF.head())

#checking data set for null values
diabetesDF.info()


#use this to check correlation in dataset 
corr = diabetesDF.corr()
print(corr)
sns.heatmap(corr, 
         xticklabels=corr.columns, 
         yticklabels=corr.columns)

#Splitting and normalizing data
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

#converting to numpy and normalizing data
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))

#calculation mean, std 
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1



diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy with logistic regression = ", accuracy * 100, "%")

#saving the model
joblib.dump([diabetesCheck, means, stds], 'diabeteseModelLogistic.pkl')

#using saved Logistic model
diabetesLoadedModel, means, stds = joblib.load('diabeteseModelLogistic.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")

print(dfCheck.head())


sampleData = dfCheck[:1]# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)

"""
#using Linear Regression
diabetesCheck = LinearRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy with linear regression = ", accuracy * 100, "%")

#saving the model
joblib.dump([diabetesCheck, means, stds], 'diabeteseModelLinear.pkl')

#using saved  Linear model
diabetesLoadedModel, means, stds = joblib.load('diabeteseModelLinear.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")

print(dfCheck.head())


sampleData = dfCheck[:1]# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds# predict
predictionProbability = diabetesLoadedModel.predict(sampleDataFeatures)


prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)

"""