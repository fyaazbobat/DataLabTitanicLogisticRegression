# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:41:58 2021

@author: HP desktop
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df_fyaaz = pd.read_csv('C:/Users/HP desktop/Downloads/titanic3.csv')
# Explore the data check the column values
print(df_fyaaz.columns.values)
print (df_fyaaz.head())
print (df_fyaaz.info())
categories = []
for col, col_type in df_fyaaz.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df_fyaaz[col].fillna(0, inplace=True)
print(categories)
print(df_fyaaz.columns.values)
print(df_fyaaz.head())
df_fyaaz.describe()
df_fyaaz.info()
#check for null values
print(len(df_fyaaz) - df_fyaaz.count())  #Cabin , boat, home.dest have so many missing values


include = ['age','sex', 'embarked', 'survived']
df_fyaaz_ = df_fyaaz[include]
print(df_fyaaz_.columns.values)
print(df_fyaaz_.head())
df_fyaaz_.describe()
df_fyaaz_.dtypes
df_fyaaz_['sex'].unique()
df_fyaaz_['embarked'].unique()
df_fyaaz_['age'].unique()
df_fyaaz_['survived'].unique()
# check the null values
print(df_fyaaz_.isnull().sum())
print(df_fyaaz_['sex'].isnull().sum())
print(df_fyaaz_['embarked'].isnull().sum())
print(len(df_fyaaz_) - df_fyaaz_.count())

#8.	Drop the rows with missing values
df_fyaaz_.loc[:,('age','sex', 'embarked', 'survived')].dropna(axis=0,how='any',inplace=True) 
df_fyaaz_.info() 

categoricals = []
for col, col_type in df_fyaaz_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     
print(categoricals)


df_ohe = pd.get_dummies(df_fyaaz_, columns=categoricals, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())

from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['age'].describe())
print(scaled_df['sex_male'].describe())
print(scaled_df['sex_female'].describe())
print(scaled_df['embarked_C'].describe())
print(scaled_df['embarked_Q'].describe())
print(scaled_df['embarked_S'].describe())
print(scaled_df['survived'].describe())
print(scaled_df.dtypes)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'survived'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)


testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))


import joblib 
joblib.dump(lr, 'C:/Users/HP desktop/Downloads/model_lr2.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/HP desktop/Downloads/model_columns.pkl')
print("Models columns dumped!")
