# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:27:16 2018

@author: hp
"""

import pandas as pd
import numpy as np
adult_df=pd.read_csv('adult_data.csv',header=None,delimiter=' *, *',engine='python')
adult_df.head()
#  *,* DELIMITER IS USED TO REMOVE LEADING AND TRAILING IN FIELD
# ENGINE=python is used to avoid warning

adult_df.shape
#Output
""" adult_df.shape
Out[25]: (32561, 15)"""
#Assing column names to the dataset
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
adult_df.head()

#Indentify missin values
adult_df.isnull().sum()
'''age               0
workclass         0
fnlwgt            0
education         0
education_num     0
marital_status    0
occupation        0
relationship      0
race              0
sex               0
capital_gain      0
capital_loss      0
hours_per_week    0
native_country    0
income            0
dtype: int64 '''
#As missing values are present in the form of ? the op shows that there is no null value

#Number of Question marks
#Here only categorical values are used because if numerical values are sent then
#it will try to compare numerical values with '?' which is a string.It causes error
for value in ['workclass','education','marital_status','occupation','relationship'
              ,'race','sex','native_country','income']:
    print(value,":", sum(adult_df[value]=='?'))
'''workclass : 1836
education : 0
marital_status : 0
occupation : 1843
relationship : 0
race : 0
sex : 0
native_country : 583
income : 0'''

#Create copy of dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
adult_df_rev.head()
adult_df_rev.describe(include='all')

#Handle missing values
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'],
                adult_df_rev.describe(include='all')[value][2],inplace=True)
adult_df_rev.head(20)
#inplace=true is specified so that all the changes are permanent
#Another metho is to use mode function directly
"""
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'],
                adult_df_rev[value].mode()[0],inplace=True)"""
#Check whether missing values have been replaced
for value in ['workclass','education','marital_status','occupation','relationship'
              ,'race','sex','native_country','income']:
    print(value,":",sum(adult_df_rev[value]=='?'))    

"""    workclass : 0
education : 0
marital_status : 0
occupation : 0
relationship : 0
race : 0
sex : 0
native_country : 0
income : 0"""

colname=['workclass','education','marital_status','occupation','relationship','race','sex',
         'native_country','income']
colname

from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))
adult_df_rev.head(10)

#Creating arrays of dependent and independent variables

X=adult_df_rev.values[:,:-1]
Y=adult_df_rev.values[:,-1]
X
Y

#Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)

from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#Run the model
from sklearn.linear_model import LogisticRegression
#Create a model
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

print(list(zip(Y_test,Y_pred)))

#Accuracy,Matrix,Report
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification Report")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",accuracy_score)

#Store the predicted probabilities
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)

y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value<0.6:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

from sklearn.metrics import confusion_matrix,accuracy_score
cfm=confusion_matrix(Y_test.tolist(),y_pred_class)  #both parameters should be of same type
print(cfm)
accuracy_score=accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",accuracy_score)


y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value<0.8:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

from sklearn.metrics import confusion_matrix,accuracy_score
cfm=confusion_matrix(Y_test.tolist(),y_pred_class)  #both parameters should be of same type
print(cfm)
accuracy_score=accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",accuracy_score)

#To check different cut offs
#Optimizing logisitc model

for a in np.arange(0,1,0.05):
    predict_mine=np.where(y_pred_prob[:,0]<a,1,0)
    cfm=confusion_matrix(Y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ",a,":",total_err,"type 2 error:",cfm[1,0])
"""    Errors at threshold  0.0 : 2346 type 2 error: 2346
Errors at threshold  0.05 : 2096 type 2 error: 2088
Errors at threshold  0.1 : 2026 type 2 error: 2009
Errors at threshold  0.15000000000000002 : 1966 type 2 error: 1939
Errors at threshold  0.2 : 1914 type 2 error: 1869
Errors at threshold  0.25 : 1850 type 2 error: 1781
Errors at threshold  0.30000000000000004 : 1805 type 2 error: 1700
Errors at threshold  0.35000000000000003 : 1785 type 2 error: 1624
Errors at threshold  0.4 : 1750 type 2 error: 1530
Errors at threshold  0.45 : 1735 type 2 error: 1426
Errors at threshold  0.5 : 1732 type 2 error: 1318
Errors at threshold  0.55 : 1732 type 2 error: 1197
Errors at threshold  0.6000000000000001 : 1771 type 2 error: 1076
Errors at threshold  0.65 : 1840 type 2 error: 928
Errors at threshold  0.7000000000000001 : 1938 type 2 error: 752
Errors at threshold  0.75 : 2170 type 2 error: 594
Errors at threshold  0.8 : 2526 type 2 error: 428
.......Errors at threshold  0.8500000000000001 : 3099 type 2 error: 265
Errors at threshold  0.9 : 3841 type 2 error: 155
Errors at threshold  0.9500000000000001 : 5130 type 2 error: 69"""

##Cross Validation
classifier=(LogisticRegression())
from sklearn import cross_validation
#Performing k_fold cross validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#Running the model using scoring metric as accuracy

kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,
                                                 scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)
#Finding the mean
print(kfold_cv_result.mean())

###Only if accuracy value of model and cross validated model is different then use this
for train_value,test_value in kfold_cv:
    classifier.fit(X_train[train_value],Y_train[train_value]).predict(X_train[test_value])
    
Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

##################################Feature Selection#################
##Recursive Feature Elimination
colname=adult_df_rev.columns[:]
from sklearn.feature_selection import RFE
rfe=RFE(classifier,6)
model_rfe=rfe.fit(X_train,Y_train)
print("Num features: ",model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname,model_rfe.support_)))
print("Feature Ranking: ",model_rfe.ranking_)

"""Num features:  6
Selected Features: 
[('age', True), ('workclass', False), ('fnlwgt', False), ('education', False), ('education_num', True), ('marital_status', True), ('occupation', False), ('relationship', False), ('race', False), ('sex', True), ('capital_gain', True), ('capital_loss', False), ('hours_per_week', True), ('native_country', False)]
Feature Ranking:  [1 5 7 6 1 1 8 3 4 1 1 2 1 9]"""

######in feature ranking values 9 highest number indicates that it was removed first.then 8...
######thode who  have value 1 that remain.

Y_pred=model_rfe.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm2=confusion_matrix(Y_test,Y_pred)
print(cfm2)
print("Classification Report: ")
print(classification_report(Y_test,Y_pred))
accuracyScore=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",accuracyScore)

##Univariate Feature Selection
X=adult_df_rev.values[:,:-1]
Y=adult_df_rev.values[:,-1]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test=SelectKBest(score_func=chi2,k=5)##k=5 is the number of variables we want
fit1=test.fit(X,Y)
print(fit1.scores_)
new_X=fit1.transform(X)
print(list(zip(colname,fit1.scores_))) #K Variables with highest scores are selected
print(new_X)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(new_X)
X=scaler.transform(new_X)
#Split the model
from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#Run the model
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

print(list(zip(Y_test,Y_pred)))

#Accuracy,Matrix,Report
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification Report")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",accuracy_score)