# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:04:12 2021

@author: Soumya PC
"""

import pandas as pd
import numpy as np
######### Iris Data Set ########################
iris = pd.read_csv("iris.csv")
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ip_columns = ["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]
op_column  = ["Species"]

# Splitting data into train and test
Xtrain,Xtest,ytrain,ytest = train_test_split(iris[ip_columns],iris[op_column],test_size=0.3, random_state=0)

ignb = GaussianNB()
imnb = MultinomialNB()

# Building and predicting at the same time 

pred_gnb = ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)


# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten()) # 100%


# Confusion matrix multinomial model
confusion_matrix(ytest,pred_mnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_mnb) # confusion matrix using 
np.mean(pred_mnb==ytest.values.flatten()) # 60%

confusion_matrix(ytest,pred_mnb) # Multinomal model is the best model to build