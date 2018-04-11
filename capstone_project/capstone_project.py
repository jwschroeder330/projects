#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:49:16 2018

@author: jacobschroeder
"""



# Step 1: Improved Data Wrangling ---------------------------------------------
import pandas as pd


# data wrangling - line up regional google trends data with demo data
df_google = pd.read_csv('/Users/jacobschroeder/anaconda3/projects/test_google_bank_account.csv')

df_demo = pd.read_csv('/Users/jacobschroeder/anaconda3/projects/full_dataset_bystate.csv', index_col=['Year', 'State'])

# rename the 'date' index to 'Year'
df_google.rename( columns= {'date' : 'Year'},inplace=True)

# cleanup the multi-index for joining
df_google.sort_values(by=['Year','State'], axis=0, inplace=True)
df_google.set_index(['Year', 'State'], inplace=True)

# join the datsets on their new, multilevel index
df = df_demo.join(df_google)

# clean up data
df = df[df['bank account'].isnull() == False]

# send to .csv
df.to_csv('/Users/jacobschroeder/anaconda3/projects/capstone_project/test_output/phase_one_test.csv')


# Step 2: Training and Test Set Creation --------------------------------------

'''
# features: X
X = df.iloc[:, :-1]

# target: y
y = df['bank account']


# train / test split with no shufle to preserve time series
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False, random_state=42)
'''

# predicting next year's data
X_train = df.loc[2004]
y_train = df.loc[2005]

X_test = df.loc[2006]
y_test = df.loc[2007]


# Step 3: Setup Straw Man -----------------------------------------------------

# linear regression with pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('linreg', LinearRegression())]

pipeline = Pipeline(steps)

model = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


# Compute and print R^2 and RMSE
print("R^2: {}".format(pipeline.score(X_test, y_test)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

print(pipeline)


# Step 4: Refined Prediction --------------------------------------------------
                

