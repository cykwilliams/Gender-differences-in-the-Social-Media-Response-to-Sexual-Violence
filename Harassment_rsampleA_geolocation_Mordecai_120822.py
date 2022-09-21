#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 10:48:05 2022

@author: chris
"""

from mordecai import Geoparser
geo = Geoparser()

import pandas as pd
import ast
from tqdm import tqdm
import csv
tqdm.pandas()

dtypes_of_2={
        'created_at'  :  str,
        'id'  :  str,
        'id_str'  :  str,
        'full_text'  :  str,
        'truncated'  :  bool,
        'display_text_range'  :  str,
        'entities'  :  str,
        'source'  :  str,
        'in_reply_to_status_id'  :  str,
       'in_reply_to_status_id_str'  :  str,
       'in_reply_to_user_id'  :  str,
       'in_reply_to_user_id_str'  :  str,
       'in_reply_to_screen_name'  :  str,
       'user'  :  str, 
       'geo'  :  str,
       'coordinates'  :  str,
       'place'  :  str,
       'contributors'  :  str,
       'retweeted_status'  :  str,
       'is_quote_status'  :  str,
       'retweet_count'  :  int,
       'favorite_count'  :  int,
       'favorited'  :  bool,
       'retweeted'  :  bool,
       'lang'  :  str,
       'quoted_status_id'  :  str,
       'quoted_status_id_str'  :  str,
       'quoted_status_permalink'  :  str,
       'extended_entities'  :  str,
       'possibly_sensitive'  :  str,
       'withheld_in_countries'  :  str,
       'withheld_scope'  :  str,
       'withheld_copyright'  :  str,
       'quoted_status' : str
        }

def transform_users(sample):
    users = sample['includes'].apply(lambda x: x.get('users'))
    users = users.apply(lambda x: x[0])
    id_str = users.apply(lambda x: x.get('id'))
    location = users.apply(lambda x: x.get('location'))
    name = users.apply(lambda x: x.get('username'))
    screen_name = users.apply(lambda x: x.get('name'))
    sample = pd.concat([
        id_str,
        screen_name,
        name,
        location
        ], axis=1)
    sample.columns=[
        'id_str',
        'screen_name',
        'name',
        'location']
    return sample

df_random_sample1_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache/df_random_sample1.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
     converters={'data' : eval,
                 'includes' : eval,
                 '__twarc' : eval},
    index_col=0)
df_random_sample1 = pd.concat(df_random_sample1_chunk)
df_random_sample1 = transform_users(df_random_sample1)
print("1of4 complete")

df_random_sample2_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache/df_random_sample2.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
     converters={'data' : eval,
                 'includes' : eval,
                 '__twarc' : eval},
    index_col=0)
df_random_sample2 = pd.concat(df_random_sample2_chunk)
df_random_sample2 = transform_users(df_random_sample2)
print("2of4 complete")

df_random_sample3_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache/df_random_sample3.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
     converters={'data' : eval,
                 'includes' : eval,
                 '__twarc' : eval},
    index_col=0)
df_random_sample3 = pd.concat(df_random_sample3_chunk)
df_random_sample3 = transform_users(df_random_sample3)
print("3of4 complete")

df_random_sample4_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache/df_random_sample4.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
     converters={'data' : eval,
                 'includes' : eval,
                 '__twarc' : eval},
    index_col=0)
df_random_sample4 = pd.concat(df_random_sample4_chunk)
df_random_sample4 = transform_users(df_random_sample4)
print("4of4 complete")

pdList = [df_random_sample1, df_random_sample2, df_random_sample3, df_random_sample4]  # List of your dataframes
df_random_sample_A = pd.concat(pdList)

#This sample only has User_location, no place or coordinates

df_random_sample_A_users = df_random_sample_A.drop_duplicates(subset = ['screen_name'], keep = 'first')
df_random_sample_A_users = df_random_sample_A_users.rename(columns = {"location":'user_location'})

print(df_random_sample_A_users[['user_location']].notnull().sum())
#user_location    421968

df_random_sample_A_users_location = df_random_sample_A_users[['user_location']]
print(df_random_sample_A_users_location.shape)
#(738586, 1)
df_random_sample_A_users_location = df_random_sample_A_users_location.drop_duplicates(subset = 'user_location')
print(df_random_sample_A_users_location.shape)
#(230040, 1)
df_random_sample_A_users_location['user_location'] = df_random_sample_A_users_location['user_location'].astype(str)
df_random_sample_A_users_location['Mordercai_output'] = df_random_sample_A_users_location['user_location'].progress_apply(lambda x: geo.geoparse(x)) #3:05:54

row_length = []
for row in range(len(df_random_sample_A_users_location)):      
    row_length.append(len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]))
max(row_length)
min(row_length)
#Max = 20, min = 0; only go to 6 as before


Mordercai_location_0 = []
Mordercai_location_1 = []
Mordercai_location_2 = []
Mordercai_location_3 = []
Mordercai_location_4 = []
Mordercai_location_5 = []

for row in range(len(df_random_sample_A_users_location)):      
    if df_random_sample_A_users_location['Mordercai_output'].iloc[row] == []:
            Mordercai_location_0.append('null')
            Mordercai_location_1.append('null')
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
            Mordercai_location_4.append('null')
            Mordercai_location_5.append('null')
    else:    
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 1:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append('null')
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
            Mordercai_location_4.append('null')
            Mordercai_location_5.append('null')
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 2:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
            Mordercai_location_4.append('null')
            Mordercai_location_5.append('null')
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 3:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append('null')
            Mordercai_location_4.append('null')
            Mordercai_location_5.append('null')
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 4:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][3])
            Mordercai_location_4.append('null')
            Mordercai_location_5.append('null')
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 5:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][3])
            Mordercai_location_4.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][4])
            Mordercai_location_5.append('null')
        if len(df_random_sample_A_users_location['Mordercai_output'].iloc[row]) == 6:
            Mordercai_location_0.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][3])
            Mordercai_location_4.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][4])
            Mordercai_location_5.append(df_random_sample_A_users_location['Mordercai_output'].iloc[row][5])
            
df_Mordercai = pd.DataFrame([df_random_sample_A_users_location['user_location'].tolist(), Mordercai_location_0, Mordercai_location_1, Mordercai_location_2, 
              Mordercai_location_3, Mordercai_location_4, Mordercai_location_5]).transpose()
df_Mordercai.columns = ['user_location', 'output_loc_0', 'output_loc_1', 'output_loc_2', 'output_loc_3', 'output_loc_4', 'output_loc_5']   
 
df_Mordercai = df_Mordercai.fillna('null')

df_Mordercai_loc_0 = df_Mordercai[['user_location', 'output_loc_0']]
df_Mordercai_loc_1 = df_Mordercai[['user_location', 'output_loc_1']]
df_Mordercai_loc_2 = df_Mordercai[['user_location', 'output_loc_2']]
df_Mordercai_loc_3 = df_Mordercai[['user_location', 'output_loc_3']]
df_Mordercai_loc_4 = df_Mordercai[['user_location', 'output_loc_4']]
df_Mordercai_loc_5 = df_Mordercai[['user_location', 'output_loc_5']]

#First explode _loc_0
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_0['output_loc_0'])):
    if df_Mordercai_loc_0['output_loc_0'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_0['output_loc_0'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_0_exploded = pd.DataFrame([df_Mordercai_loc_0['user_location'].tolist(), 
                                            df_Mordercai_loc_0['output_loc_0'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_0_exploded.columns = ['user_location', 'output_loc_0', 'place_name_loc_0', 'country_predicted_loc_0', 'country_code3_loc_0',
                                       'country_conf_loc_0', 'lat_loc_0', 'long_loc_0']

##Same for _loc_1
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_1['output_loc_1'])):
    if df_Mordercai_loc_1['output_loc_1'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_1['output_loc_1'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_1_exploded = pd.DataFrame([df_Mordercai_loc_1['user_location'].tolist(), 
                                            df_Mordercai_loc_1['output_loc_1'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_1_exploded.columns = ['user_location', 'output_loc_1', 'place_name_loc_1', 'country_predicted_loc_1', 'country_code3_loc_1',
                                       'country_conf_loc_1', 'lat_loc_1', 'long_loc_1']

##Same for _loc_2
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_2['output_loc_2'])):
    if df_Mordercai_loc_2['output_loc_2'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_2['output_loc_2'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_2_exploded = pd.DataFrame([df_Mordercai_loc_2['user_location'].tolist(), 
                                            df_Mordercai_loc_2['output_loc_2'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_2_exploded.columns = ['user_location', 'output_loc_2', 'place_name_loc_2', 'country_predicted_loc_2', 'country_code3_loc_2',
                                       'country_conf_loc_2', 'lat_loc_2', 'long_loc_2']

##Same for _loc_3
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_3['output_loc_3'])):
    if df_Mordercai_loc_3['output_loc_3'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_3['output_loc_3'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_3_exploded = pd.DataFrame([df_Mordercai_loc_3['user_location'].tolist(), 
                                            df_Mordercai_loc_3['output_loc_3'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_3_exploded.columns = ['user_location', 'output_loc_3', 'place_name_loc_3', 'country_predicted_loc_3', 'country_code3_loc_3',
                                       'country_conf_loc_3', 'lat_loc_3', 'long_loc_3']

##Same for _loc_4
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_4['output_loc_4'])):
    if df_Mordercai_loc_4['output_loc_4'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_4['output_loc_4'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_4_exploded = pd.DataFrame([df_Mordercai_loc_4['user_location'].tolist(), 
                                            df_Mordercai_loc_4['output_loc_4'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_4_exploded.columns = ['user_location', 'output_loc_4', 'place_name_loc_4', 'country_predicted_loc_4', 'country_code3_loc_4',
                                       'country_conf_loc_4', 'lat_loc_4', 'long_loc_4']

##Same for _loc_5
country_predicted = []
country_conf = []
lat = []
long = []
country_code3 = []
place_name = []

for row2 in range(len(df_Mordercai_loc_5['output_loc_5'])):
    if df_Mordercai_loc_5['output_loc_5'].iloc[row2] == 'null':
        country_predicted.append('null')
        country_conf.append('null')
        lat.append('null')
        long.append('null')
        country_code3.append('null')
        place_name.append('null')
    else:
        country_predicted.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['country_predicted'])
        country_conf.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['country_conf'])
        try:
            lat.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['geo']['lat'])
            long.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['geo']['lon'])
            country_code3.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['geo']['country_code3'])
            place_name.append(df_Mordercai_loc_5['output_loc_5'].iloc[row2]['geo']['place_name'])
        except KeyError:
            lat.append('null')
            long.append('null')
            country_code3.append('null')
            place_name.append('null')

df_Mordercai_loc_5_exploded = pd.DataFrame([df_Mordercai_loc_5['user_location'].tolist(), 
                                            df_Mordercai_loc_5['output_loc_5'].tolist(),
                                            place_name, country_predicted, country_code3, country_conf, lat, long
                                            ]).transpose()
df_Mordercai_loc_5_exploded.columns = ['user_location', 'output_loc_5', 'place_name_loc_5', 'country_predicted_loc_5', 'country_code3_loc_5',
                                       'country_conf_loc_5', 'lat_loc_5', 'long_loc_5']




import numpy as np
df_Mordercai_loc_combined = df_Mordercai_loc_0_exploded.merge(df_Mordercai_loc_1_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_2_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_3_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_4_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_5_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.replace('null', np.nan)

#Save to csv:
#df_Mordercai_loc_combined.to_csv('/home/chris/Downloads/tweet-ids-001/SarahEverard/df_sample_A_Mordercai_location_combined_120822.csv', sep=',', quotechar='"')
#Re-import:
df_Mordercai_loc_combined = pd.read_csv('/home/chris/Downloads/tweet-ids-001/SarahEverard/df_sample_A_Mordercai_location_combined_120822.csv', sep=',', quotechar='"', indexcol=0)


df_random_sample_A = df_random_sample_A.rename(columns = {'location':'user_location'})

#Pair df_Mordercai_loc_combined with 'id_str' from df_random_sample_A
df_Mordercai_loc_combined = df_random_sample_A.merge(df_Mordercai_loc_combined, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.drop_duplicates(subset = 'id_str')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.rename(columns = {'id_str':'id'})

#Import the m3sampled output dataset from which control sample statistics were calculated
df_random_sample_result_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache/random_twitter_sample_m3_result.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
    index_col=0)
df_random_sample_result = pd.concat(df_random_sample_result_chunk)
print(df_random_sample_result.shape)
#(463445, 9)

##Now make new column with Organisational status labelled
conditions_org= [
    (df_random_sample_result['org_non-org'] > 0.5),
    (df_random_sample_result['org_non-org'] < 0.5),
    (df_random_sample_result['org_non-org'] == 0.5)
    ]
values_org= ['Non_org', 'Is_org', 'Non-Org_unclear']
df_random_sample_result['Org'] = np.select(conditions_org, values_org)
df_random_sample_result['Org'].value_counts()
#Non_org    418970
#Is_org      44475
#Remove Org from dataset:
df_random_sample_result_org = df_random_sample_result.loc[df_random_sample_result['Org'] == 'Is_org']
df_random_sample_result = df_random_sample_result.loc[df_random_sample_result['Org'] == 'Non_org']
df_random_sample_result.shape
#(418970, 10)
#Assign Gender:
df_random_sample_result['Gender'] = np.where(df_random_sample_result.gender_male > 0.5, 'Male', 'Female')
#Calculate final numbers of Male vs Female for this random sample
df_random_sample_result.Gender.value_counts()
#Male      291662
#Female    127308

##Calculate age
df_random_sample_result['Age'] = df_random_sample_result[['age_<=18','age_19-29', 'age_30-39', 'age_>=40']].idxmax(axis=1)
print('Sample 1 age distribution (all) \n', df_random_sample_result['Age'].value_counts())
#age_<=18     148377
#age_19-29    111051
#age_>=40      86119
#age_30-39     73423

df_random_sample_result['id'] = df_random_sample_result['id'].astype(str)
df_random_sample_A_location = df_random_sample_result.merge(df_Mordercai_loc_combined, on = 'id', how = 'left')
print(df_random_sample_A_location.shape)
#(463445, 54)
df_random_sample_A_location['country_predicted_final'] = df_random_sample_A_location['country_predicted_loc_0']

n_location = df_random_sample_A_location['country_predicted_final'].notnull().sum()
print('Tweets with location data available:', n_location)
#107112
df_location_value_counts = pd.DataFrame(df_random_sample_A_location['country_predicted_final'].value_counts())
df_location_value_counts['percent'] = df_location_value_counts['country_predicted_final'] / n_location * 100
df_location_value_counts.head(10)

#     country_predicted_final    percent
#USA                    20208  21.233805
#IND                     8921   9.373851
#GBR                     6826   7.172504
#BRA                     4636   4.871334
#IDN                     4028   4.232471
#CRI                     3026   3.179607
#FRA                     2799   2.941084
#ESP                     2731   2.869632
#MEX                     2672   2.807637
#TUR                     2470   2.595383

#Change to USA country code
countrycode_to_USA = [
    'Brooklyn, NY', 'Los Angeles, CA', 'Los Angeles', 'San Antonio, TX',
    'Santa Rosa, CA', 'Los Angeles, California', 'los angeles', 'San Antonio, Texas', 'Palo Alto, CA', 'Los Angeles',
    'California, USA', 'California', 'Southern California', 'Cali', 'california'
]

for country in countrycode_to_USA:
    df_random_sample_A_location.loc[df_random_sample_A_location['user_location'] == country, 'country_predicted_final'] = 'USA'

print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'USA']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'IND']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'GBR']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'BRA']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'IDN']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'CRI']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'FRA']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'ESP']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'TUR']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'CAN']['user_location'].value_counts().head(10), '\n')
#All the above are accurate

df_random_sample_A_location_GBR_only = df_random_sample_A_location[df_random_sample_A_location['country_predicted_final'] == 'GBR']

###GBR only sensitivity-analysis
#Calculate final numbers of Male vs Female for df_random_sample_A_location_GBR_only
df_random_sample_A_location_GBR_only_gender = pd.DataFrame(df_random_sample_A_location_GBR_only.Gender.value_counts())
df_random_sample_A_location_GBR_only_gender['percent'] = df_random_sample_A_location_GBR_only_gender['Gender'] / (df_random_sample_A_location_GBR_only['country_predicted_final'].notnull().sum()) * 100
#Male      4857
#Female    1969
#Name: Gender, dtype: int64