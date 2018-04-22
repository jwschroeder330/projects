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
df.to_csv('/Users/jacobschroeder/anaconda3/projects/capstone_project/test_output/final_dataset.csv')

