# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:06:10 2018

@author: jschroeder
"""

import pandas as pd
import numpy as np

from datetime import timedelta

'''
READ ME:
    
This file contains the workflow for testing and tuning the models, including
the final model, Linear SVR.

The data for this model was generated on my Macbook Pro using the scripts:
    google_trends.py
    data_wrangling_enhanced.py
    combine_google_demo.py
    
While working between computers, the filenames and setup may change, but the data
and the project remain the same. 

You may find the rough draft of my early model testing at regression_selection.py

The .csv files included in this directory serve as examples of the data used during
this project.
    full_dataset_bystate.csv - the parsed ASCII demographic file
    test_google_bank_account.csv - the parsed Google Trends data
    bank_account.csv - the combined file (named final_dataset.csv in the combine script)


'''


# read in the final dataset produced from combine_google_demo.py
df = pd.read_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\bank_account.csv', index_col='Year', parse_dates=True)

year_set_previous = set(df.index.year)

df['bank account previous'] = ''

for y in df.index.unique(): 
    if y.year != 2004:
                
        if ((y.year - 2001) % 4) == 0:
            df['bank account previous'][df.index == y] = df.loc[y - timedelta(days=366), 'bank account']
            # print(df.loc[y - timedelta(days=366), 'bank account previous'])    
        else:
            df['bank account previous'][df.index == y] = df.loc[y - timedelta(days=365), 'bank account']
            # print(df.loc[y - timedelta(days=365), 'bank account previous'])

df = df.loc['2005':, :]

df['bank account previous'] = df['bank account previous'].astype('float')

df.rename(columns={'bank account' : 'bank account copy'}, inplace=True)

df['bank account'] = df['bank account copy']

del df['bank account copy']

del df['State']

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()
X = df.iloc[:, :-1].as_matrix()
y = df.iloc[:, -1].as_matrix()


for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# tuning the hyperparameters
from sklearn.model_selection import GridSearchCV

# credit: https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
def linearsvr_param_selection(X_train, y_train, pipeline):
    param_grid = {
            'linreg__C': np.linspace(0.001, 10, num=100),
            'linreg__loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'linreg__fit_intercept' : [True, False]
                    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search


# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('linreg', LinearSVR())]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

model = linearsvr_param_selection(X_train, y_train, pipeline)

# maybe try going straight to model.predict()? Is fitting again necessary?
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Compute and print R^2 and RMSE
# print("Optimal alpha: {}".format(param_dict['alpha']))
print("R^2: {}".format(model.score(X_test, y_test)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

print(model)

print("""
      

      The coefficients of the features are:
      {}

""".format(model.estimator.named_steps['linreg'].coef_))

coef = model.estimator.named_steps['linreg'].coef_

coef_index_sorted = coef.argsort()[-4:-1][::-1]

coef_index_sorted_desc = coef.argsort()[:3][::1]

demo_dict = {
          0 : '<1',          
          1  : '1-4',              
          2 :  '5-9',             
          3 : '10-14',         
          4 : '15-19',            
          5 : '20-24',         
          6 :  '25-29',            
          7  : '29-34',        
          8  : '35-39',      
          9 : '40-44',
          10 : '45-49',
          11 :  '49-54',
          12 :  '55-59',
          13 :  '60-64',
          14  : '65-69',
          15 : '70-74',
          16  : '75-79',
          17 :  '80-84',
          18  : '85+',
          19 : 'Last Year Keyword Data'
        }

top_three = []
bottom_three = []

for item in coef_index_sorted:
    top_three.append(item)
    
for item in coef_index_sorted_desc:
    bottom_three.append(item)


print("""
      
      Demographics are sorted into 19 age groups:
          1.  <1                11.  45-49
          2.  1-4               12.  49-54 
          3.  5-9               13.  55-59
          4.  10-14             14.  60-64
          5.  15-19             15.  65-69
          6.  20-24             16.  70-74
          7.  25-29             17.  75-79
          8.  29-34             18.  80-84 
          9.  35-39             19.  85+
          10. 40-44             
          
      
      Top 3 positively correlated demographics: 
          {}, {}, {}
          
      Bottom 3 negatively correlated demographics:
          {}, {}, {}
          
          """.format(demo_dict[top_three[0]], 
          demo_dict[top_three[1]], 
          demo_dict[top_three[2]], 
          demo_dict[bottom_three[0]],
          demo_dict[bottom_three[1]],
          demo_dict[bottom_three[2]]))

print(coef_index_sorted)
print(coef[coef_index_sorted])
print(coef_index_sorted_desc)
print(coef[coef_index_sorted_desc])





