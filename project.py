# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:34:54 2021

@author: Sharique Ahmad Khan
"""


#Importing Imp library
import pandas as pd
import numpy as np

#Accessing dataset
df = pd.read_csv(r"D:\DataSets\Datasets_Multinomial\loan.csv")

#Column name
df.head()

#Description
df.describe()

#for infer the type of data in each
df.info()

#Checking for null vlaue
n = df.isna().sum()

#Column names
df.columns

#Droping the member id and id because they are unique
df.drop(['id', 'member_id'], inplace = True, axis = 1)

#There so many coulumn with no entries we have to drop those columns
for i in range (0, 109):
    d =  df.columns
    if df.iloc[:, i:i+1].isna().sum()[0] > 12000:
       df.drop(d[i], inplace = True, axis = 1)
       
for i in range (0, 109):
    d =  df.columns
    if df.iloc[:, i:i+1].isna().sum()[0] > 12000:
       df.drop(d[i], inplace = True, axis = 1)

for i in range (0, 109):
    d =  df.columns
    if df.iloc[:, i:i+1].isna().sum()[0] > 12000:
       df.drop(d[i], inplace = True, axis = 1)

for i in range (0, 109):
    d =  df.columns
    if df.iloc[:, i:i+1].isna().sum()[0] > 12000:
       df.drop(d[i], inplace = True, axis = 1)
        
for i in range (0, 109):
    d =  df.columns
    if df.iloc[:, i:i+1].isna().sum()[0] > 12000:
       df.drop(d[i], inplace = True, axis = 1)
#Here we can see I use 5 times same code because it's doesn't give me by
#running 1 time so you have to run it 5 times and I am not able find an alternate     

#So There sum nan value
n =  df.isna().sum()    


#We use simple Imputer for null value imputer
#For Imputation nan value
from sklearn.impute import SimpleImputer

#There are relaibility if I apply mode in all feature
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df['emp_title'] = pd.DataFrame(mode_imputer.fit_transform(df[['emp_title']]))
df['emp_length'] = pd.DataFrame(mode_imputer.fit_transform(df[['emp_length']]))
df['revol_util'] = pd.DataFrame(mode_imputer.fit_transform(df[['revol_util']]))
df['title'] = pd.DataFrame(mode_imputer.fit_transform(df[['title']]))
df['last_pymnt_d'] = pd.DataFrame(mode_imputer.fit_transform(df[['last_pymnt_d']]))
df['last_credit_pull_d'] = pd.DataFrame(mode_imputer.fit_transform(df[['last_credit_pull_d']]))
df['collections_12_mths_ex_med'] = pd.DataFrame(mode_imputer.fit_transform(df[['collections_12_mths_ex_med']]))
df['chargeoff_within_12_mths'] = pd.DataFrame(mode_imputer.fit_transform(df[['chargeoff_within_12_mths']]))
df['pub_rec_bankruptcies'] = pd.DataFrame(mode_imputer.fit_transform(df[['pub_rec_bankruptcies']]))
df['tax_liens'] = pd.DataFrame(mode_imputer.fit_transform(df[['tax_liens']]))

#Checking the null value again
n =  df.isna().sum()    #No null value

#Visualization/Graph
#Univariate Analysis

import seaborn as sns

#Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loan_status", y = "loan_amnt", data = df)

sns.boxplot(x = "loan_status", y = "funded_amnt", data = df)

sns.boxplot(x = "loan_status", y = "funded_amnt_inv", data = df)

sns.boxplot(x = "loan_status", y = "installment", data = df)

sns.boxplot(x = "loan_status", y = "annual_inc", data = df)

sns.boxplot(x = "loan_status", y = "total_pymnt", data = df)

sns.boxplot(x = "loan_status", y = "total_pymnt_inv", data = df)

sns.boxplot(x = "loan_status", y = "total_rec_prncp", data = df)

sns.boxplot(x = "loan_status", y = "total_rec_int", data = df)


#Bivariate Analysis
#Scatter plot for each categorical choice of car

sns.stripplot(x = "loan_status", y = "loan_amnt", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "funded_amnt_inv", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "installment", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "annual_inc", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "total_pymnt", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "total_pymnt_inv", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "total_rec_prncp", jitter = True, data = df)

sns.stripplot(x = "loan_status", y = "total_rec_int", jitter = True, data = df)


#Changing into numerical for model
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

def encoding(i):
    df[i] = lb.fit_transform(df[i])

encoding('term')
encoding('int_rate')
encoding('grade')
encoding('sub_grade')
encoding('emp_title')
encoding('emp_length')
encoding('home_ownership')
encoding('verification_status')
encoding('issue_d')
encoding('pymnt_plan')
encoding('url')
encoding('purpose')
encoding('title')
encoding('zip_code')
encoding('addr_state')
encoding('earliest_cr_line')
encoding('revol_util')
encoding('initial_list_status')
encoding('last_credit_pull_d')
encoding('application_type')
encoding('last_pymnt_d')

#Checking for type of columns
df.info()

#Arranging columns
df = df.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]



#Correlation Matrix
cor = df.corr()


#Train test split
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2)

#Importing Logistic Regression from Sklearn.linear_model
from sklearn.linear_model import LogisticRegression

#Model Building

#multinomial option is supported only by the lbfgs solvers
model = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", max_iter = 500).fit(train.iloc[:, 1:], train.iloc[:, 0])

#Test predictions
test_predict = model.predict(test.iloc[:, 1:]) 

from sklearn.metrics import accuracy_score

#Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

#Train predictions
train_predict = model.predict(train.iloc[:, 1:])  

#Train accuracy
accuracy_score(train.iloc[:,0], train_predict)

test.tail()

sample = model.predict(test.tail().iloc[:, 1:])

accuracy_score(test.tail().iloc[:,0], sample)
