#!/usr/bin/env python
# coding:utf-8
"""
Name : gen_crime_data.py
Author  : issac
Time    : 2020/12/12 14:27
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

city = 'chicago'
dataset_name = "Crimes_2018.csv"
save_file_name = 'chi18.txt'
hour_stage = 3 # 
field_name = ['Stage', 'Location Description', 'Primary Type'] #8,143,33


def preprocess():
    path= os.getcwd() + os.sep + 'raw_data' + os.sep + city + os.sep + dataset_name   
    raw_data = pd.read_csv(path)
    raw_data['Time'] = raw_data['Date'].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))
    raw_data['Hour'] = raw_data['Time'].apply(lambda x: x.hour)
    raw_data['Stage'] = (raw_data['Hour'] / hour_stage).astype('int') + 1
    raw_data['Location Description'] = raw_data['Location Description'].fillna('empty')
    group_list, group_dic = event_dic(raw_data, field_name)

    regions = list(set(raw_data['Community Area'].dropna().astype('int')))
    for reg in regions:
        reg_data = raw_data[raw_data['Community Area']==reg]
        if reg_data.shape[0]<10:
            continue
        reg_data = reg_data.sort_values(by=['Time'])
#        reg_data['event_id'] = list(map(lambda x: group_dic[x], event_group(reg_data, field_name)))
        item_seq = list(map(lambda x: group_dic[x], event_group(reg_data, field_name)))
        append_writer(save_file_name, reg, item_seq)
        
def event_dic(raw_data, cols):
    group_list = event_group(raw_data, cols)
    group_set = set(group_list)
    group_dic = {}
    num = 0
    for g in group_set:
        num = num + 1
        group_dic[g] = num
    return group_list, group_dic

def event_group(data, cols):
    group = []
    for col in cols:      
        group.append(data[col]) 
    group_list = list(zip(*group))
    return group_list

def append_writer(file_name, region_id, item_seq):
    save_path = os.getcwd() + os.sep + 'data' + os.sep + file_name
    data = pd.DataFrame({'region_id':[region_id]*len(item_seq), 'item_id': item_seq})
    data.to_csv(save_path, sep=' ', mode='a', header=False, index=None)     


def main():
    preprocess()

if __name__ == '__main__':
    main()