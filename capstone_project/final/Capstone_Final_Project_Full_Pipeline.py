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
    
For easier readability, this is the unified script of the three. It may
help in understanding how everything was used.

Example with bank account as the keyword

'''


'''
===============================================================================
                         Google Trends
===============================================================================
'''
from pytrends.request import TrendReq


def pull_google_trends(kw, location):
    
    df_list = []

    print(location[-2:])
    
    pytrend = TrendReq(hl='en-US')

    pytrend.build_payload([kw], cat=0, timeframe='all', geo=location, gprop='')

    df_google = pytrend.interest_over_time()
        
    df_google_annual = df_google.resample('A').mean()
        
    df_google_annual.set_index(df_google_annual.index.year, inplace=True)
    
    df_google_annual['State'] = location[-2:]
    
    print(df_google_annual.head())
    
    df_list.append(df_google_annual)

    df = pd.concat(df_list)
    
    return df


# states
state_list = [ 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

final_state_list = []

# formatting cleanup for state
for state in state_list:
    final_state_list.append('US-' + state)

# calling pytrends
df_list = []

for s in final_state_list:    
    print(s)
    df_list.append(pull_google_trends('bank account', s))


df = pd.concat(df_list, axis=0)

df.to_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\test_google_bank_account.csv')
'''
===============================================================================
                        End Google Trends
===============================================================================
'''
'''
===============================================================================
                        Demographic Data
===============================================================================
'''
def ascii_parser(text_file):
    """ Given a raw ASCII file, parses and returns a pandas DataFrame """
    raw_file = open(text_file, 'r')

    parse_dict = {
            'Year' : [],
            'State' : [],
            # 'Race' : [],
            # 'Origin' : [],
            # 'Sex' : [],
            'Age' : [],
            'Population' : []
            }
    
    for line in raw_file:
        parse_dict['Year'].append(line[:4])
        parse_dict['State'].append(line[4:6])
        # parse_dict['Race'].append(line[13])
        # parse_dict['Origin'].append(line[14])
        # parse_dict['Sex'].append(line[15])
        parse_dict['Age'].append(line[16:18])
        parse_dict['Population'].append(line[18:25])
        
    df = pd.DataFrame(parse_dict)
    
    df['Age'] = df['Age'].astype('int')
    df['Population'] = df['Population'].astype('int')
    
    df_summed = df.groupby(['Year','State','Age']).sum()
    
    df_final = pd.pivot_table(df_summed, index=['Year', 'State'], columns = 'Age', values= 'Population')
    
    return df_final

text_file = 'C:\\Users\\jschroeder\\Documents\\Internal\\Training\\us.1990_2016.19ages.txt'

df = ascii_parser(text_file)

df.to_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\full_dataset_bystate.csv')

'''
===============================================================================
                        End Demographic Data
===============================================================================
'''

'''
===============================================================================
                        Combining the Datasets
===============================================================================
'''

# data wrangling - line up regional google trends data with demo data
df_google = pd.read_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\test_google_bank_account.csv')

df_demo = pd.read_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\full_dataset_bystate.csv', index_col=['Year', 'State'])

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
df.to_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\final_dataset.csv')


'''
===============================================================================
                        End Combining the Datasets
===============================================================================
'''

'''
===============================================================================
                        Regression Analysis
===============================================================================
'''


# read in the final dataset produced from combine_google_demo.py
df = pd.read_csv('C:\\Users\\jschroeder\\Documents\\Internal\\Training\\final_dataset.csv', index_col='Year', parse_dates=True)

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

'''
===============================================================================
                        End Regression Analysis
===============================================================================
'''



