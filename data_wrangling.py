#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:04:41 2018

@author: jacobschroeder
"""

from pytrends.request import TrendReq
import pandas as pd
# from matplotlib import pyplot as plt

# Notes:
# Perhaps loop gender and ethinicity data as well?


# Demographic Data ----------------------------------- #


# Read the dataset into a pandas DataFrame
df_demo = pd.read_csv('/Users/jacobschroeder/anaconda3/projects/np2008_d1_core.csv', index_col='YEAR', parse_dates=True)

# Remove unwanted columns
unwanted_columns = ['RACE', 'SEX', 'HISP']
df_demo.drop(unwanted_columns, axis=1, inplace=True)

# Set the index so the datasets join nicely
df_demo.set_index(df_demo.index.year, inplace=True)

# Print to .csv for testing purposes
df_demo.to_csv('/Users/jacobschroeder/anaconda3/projects/demo_test.csv')


# Google Trends -------------------------------------- #


# kw_list: 20 keywords based on most-expensive relevant keywords related to finance banking (source: Keyword Planner)
# We select and modify keywords based on Google Trends data availability (avoid 0s)

# Batch keywords in groups of five, since that is what the library allows
kw_1 = 'how to find number and routing number'
kw_2 = 'fiduciary bank'
kw_3 = 'foreign bank account'
kw_4 = 'balance your checkbook'
kw_5 = 'purchase money order'

kw_6 = 'savings account'
kw_7 = 'personal bankruptcy'
kw_8 = 'savings plan'
kw_9 = 'direct deposit'
kw_10 = '529 plan'

kw_11 = 'credit card'
kw_12 = 'bankruptcy'
kw_13 = 'check cashing'
kw_14 = 'ATM'
kw_15 = 'fafsa'

kw_16 = 'savings association'
kw_17 = 'deposit money order'
kw_18 = 'deposit check'
kw_19 = 'best bank accounts'
kw_20 = 'small business bank'

# Group the keywords accordingly
kw_list_1 = [kw_1, kw_2, kw_3, kw_4, kw_5]
kw_list_2 = [kw_6, kw_7, kw_8, kw_9, kw_10]
kw_list_3 = [kw_11, kw_12, kw_13, kw_14, kw_15]
kw_list_4 = [kw_16, kw_17, kw_18, kw_19, kw_20]

kw_list = [kw_list_2, kw_list_3, kw_list_4]

# Create the master google trend keyword list using kw_list_1
pytrend = TrendReq(hl='en-US')
    
pytrend.build_payload(kw_list_1, cat=0, timeframe='all', geo='US', gprop='')

df_google = pytrend.interest_over_time()
    
df_google_annual = df_google.resample('A').mean()
    
df_google_annual.set_index(df_google_annual.index.year, inplace=True)

# Iterate through the remaining lists in kw_list and join accordingly
for item in kw_list:
    pytrend = TrendReq(hl='en-US')
    
    pytrend.build_payload(item, cat=0, timeframe='all', geo='US', gprop='')
    
    temp_google = pytrend.interest_over_time()
    
    temp_google_annual = temp_google.resample('A').mean()
    
    temp_google_annual.set_index(temp_google_annual.index.year, inplace=True)
    # print(df_google_annual.head())
    
    df_google_annual = df_google_annual.join(temp_google_annual)

# Print to .csv for testing purposes
df_google_annual.to_csv('/Users/jacobschroeder/anaconda3/projects/google_test.csv')


# Combine the Data and Clean -------------------------- #


df = df_demo.join(df_google_annual)

# if inner join is needed
# df = pd.merge(df_google_annual, df_demo, left_index=True, right_index=True)

print(df)

df.to_csv('/Users/jacobschroeder/anaconda3/projects/test.csv')


# Plotting the Data ------------------------------------ #