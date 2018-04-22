#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:32:20 2018

@author: jacobschroeder
"""

import pandas as pd



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

text_file = '/Users/jacobschroeder/anaconda3/projects/us.1990_2016.19ages.txt'

df = ascii_parser(text_file)

df.to_csv('/Users/jacobschroeder/anaconda3/projects/full_dataset_bystate.csv')