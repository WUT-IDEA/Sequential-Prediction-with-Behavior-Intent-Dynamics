#!/usr/bin/env python
# coding:utf-8
"""
Name : gen_crime_data.py
Author  : issac
Time    : 2020/12/12 14:27
"""

import pandas as pd
import os
from functools import reduce

city = 'chicago'
dataset_name = "Crimes_2018.csv"
save_file_name = 'chi18.txt'
hour_stage = 3 # 

geo_group = ['Community Area', 'Location_Description']
item_group = ['Stage', 'Primary_Type']



def preprocess():
    path= os.getcwd() + os.sep + 'raw_data' + os.sep + city + os.sep + dataset_name   
    raw_data = pd.read_csv(path)
    raw_data['Time'] = raw_data['Date'].apply(lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p'))
    raw_data['Hour'] = raw_data['Time'].apply(lambda x: x.hour)
    raw_data['Stage'] = (raw_data['Hour'] / hour_stage).astype('int') + 1
    raw_data = raw_data.dropna(subset=['Community Area']).astype({'Community Area':'int'})
    raw_data = raw_data[raw_data['Community Area']!=0]
    
    # drop nan in 'Location_Description' except for 'DECEPTIVE PRACTICE'
    raw_data.rename(columns={'Location Description':'Location_Description', 'Primary Type':'Primary_Type'}, inplace=True)
    drop_index = raw_data.loc[(raw_data.Location_Description.isnull()) & (raw_data.Primary_Type!='DECEPTIVE PRACTICE')].index
    raw_data = raw_data.drop(drop_index)
    raw_data['Location_Description'].fillna('empty',inplace=True)
    
    # generate dictionary
    dic_data = raw_data[geo_group + item_group]
    geo_list, geo_dic = group_dic(dic_data, geo_group)
    raw_geo_list = divide_group(raw_data, geo_group)
    raw_data['geo_id'] = list(map(lambda x: geo_dic[x], raw_geo_list))
    raw_data = raw_data.sort_values(by=['geo_id', 'Time'])
    
    dic_data = raw_data[geo_group + item_group+['geo_id', 'Time']]
    item_list, item_dic = group_dic(dic_data, item_group)    
    raw_item_list = divide_group(raw_data, item_group) 
    raw_data['item_id'] = list(map(lambda x: item_dic[x], raw_item_list))
    
    geos = list(set(raw_data['geo_id']))
    
    f_geo_id = 0
    f_item_id = 0
    f_item_dic = {}
    f_item_min = 10000
    f_item_max = 0
    f_item_sum = 0
    
    for geo in geos:
        geo_data = raw_data[raw_data['geo_id']==geo]
#        if geo_data.shape[0]<3:
#            continue
        geo_data = geo_data.sort_values(by=['Time'])
        record_num = geo_data.shape[0]
        if  record_num < 5:
            continue
        f_geo_id += 1
        f_item_min = min(f_item_min,record_num)
        f_item_max = max(f_item_max,record_num)
        f_item_sum += record_num
        f_item_list = []
        for i in geo_data['item_id']:
            if i not in f_item_dic.keys():
                f_item_id += 1
                f_item_dic[i] = f_item_id
            f_item_list.append(f_item_dic[i])
        
        append_writer(save_file_name, f_geo_id, f_item_list)
    print ('total geo:%s, total item:%s, min sequence length:%s, max sequence length:%s, average sequence length:%s'\
           %(f_geo_id, f_item_id, f_item_min, f_item_max, f_item_sum/f_geo_id))
      
        
def group_dic(raw_data, cols):
    group_list = divide_group(raw_data, cols)
    
    group_tuple = reduce(lambda x,y: x if y in x else x + [y], [[],] + group_list)
    group_dic = {}
    num = 0
    for g in group_tuple:
        num = num + 1
        group_dic[g] = num
    return group_list, group_dic


def divide_group(data, cols):
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