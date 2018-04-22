#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:15:50 2018

@author: jacobschroeder
"""

import pandas as pd
from pytrends.request import TrendReq

# Google Trends -------------------------------------- #
""" Pull the google data by year, per state """

def pull_google_trends(kw, location):
    """ Pulls Google Trends for 5 keywords based on locale and returns pandas DataFrame """
    
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

for state in state_list:
    final_state_list.append('US-' + state)
    
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

kw_list = [kw_list_1, kw_list_2, kw_list_3, kw_list_4]


# calling pytrends
df_list = []

for s in final_state_list:    
    print(s)
    df_list.append(pull_google_trends('bank account', s))


df = pd.concat(df_list, axis=0)

df.to_csv('/Users/jacobschroeder/anaconda3/projects/test_google_bank_account.csv')
# End Google Trends -------------------------------------- #
