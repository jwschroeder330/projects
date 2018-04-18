# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:06:10 2018

@author: jschroeder
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:49:16 2018
@author: jacobschroeder
"""

import pandas as pd
import sys
import numpy as np

def parse_input_file(filename, features):
    #Getthe original csv
    df = pd.read_csv(filename)
   
    #Organize into a multi-level dictionary
    results = {}
    for (state, year), group in df.groupby(["State", "Year"]):
        if state not in results:
            results[state] = {}

        #There'z only one element here, but just in case take the first
        results[state][year] = group[features].values[0]

    #Convert the dataframe to a list of dictionaries
    old_records = df.to_dict(orient="records")

    #Name the new features like this
    new_features = [elem + " prev" for elem in features]

    #Create a new list of dict by looking back into the table
    #we just made, iterating over the list of dict
    new_records = []
    for record in old_records:
        #Start from the existing
        new_record = record

        #Get the state and year for convenience
        state = record["State"]
        year = record["Year"]

        #Shouldn't ever happen, but best to be safe
        if state not in results:
            continue

        #Skip years that aren't usable -- could also 
        #fill with some sort of default value
        if year - 1 not in results[state]:
            continue

        #Above the features were put in in order of the 'features'
        #list and the new_features list was also in that same order
        #so we're just accessing by index here. could probably be cleaner
        for i, elem in enumerate(new_features):
            new_record[elem] = results[state][year-1][i]

        #Finally add in the constructed record
        new_records.append(new_record)

    #Building dataframe from this format is easy from constructor
    df_new = pd.DataFrame(new_records)
    return df_new
'''
if __name__ == '__main__':
    #If we wanted we could use this for a couple features too
    print(parse_input_file(sys.argv[1], ["bank account"]).head())
'''

df = pd.read_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\bank_account_2.csv', parse_dates=True, index_col='Year')

del df['State']

df.index.to_datetime()

df_log = df.apply(np.log)
df_log.replace([np.inf, -np.inf], np.nan, inplace=True)
df_log.dropna(inplace=True)
from sklearn.model_selection import TimeSeriesSplit

'''
Original
------------------------------------
'''

tscv = TimeSeriesSplit()
X = df.iloc[:, :-1].as_matrix()
y = df.iloc[:, -1].as_matrix()


for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import TheilSenRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# tuning the hyperparameters
from sklearn.model_selection import GridSearchCV

# credit: https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
def linearsvr_param_selection(X, y, nfolds):
    alphas = np.linspace(0.000001, 1, num= 100) # [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

# param_dict = linearsvr_param_selection(X, y, 3)

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('linreg', LinearSVR(C=10))]

pipeline = Pipeline(steps)

model = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


# Compute and print R^2 and RMSE
# print("Optimal alpha: {}".format(param_dict['alpha']))
print("R^2: {}".format(pipeline.score(X_test, y_test)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_pred)))

print(pipeline)

print("""
      

      The coefficients of the features are:
      {}

""".format(pipeline.named_steps['linreg'].coef_))

coef = pipeline.named_steps['linreg'].coef_

coef_index_sorted = coef.argsort()[-3:][::-1]

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
          18  : '85+'
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


'''

Coeficients have whatever metric you are measuring, per z-score

Google popularity metric per z-score, controlling for last year's score


Hold out split

To avoid bias

our selection of an optimized parameter

'''

''' 
two-fold cross-validation

train, test in an outer loop

another train, test in inner loop



cross-validation - making the most of a small dataset



'''














