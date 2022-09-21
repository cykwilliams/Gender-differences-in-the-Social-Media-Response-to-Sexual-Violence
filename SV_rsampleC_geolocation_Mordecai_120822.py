#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:53:12 2022

@author: chris
"""

from mordecai import Geoparser
geo = Geoparser()

import pandas as pd
import ast
from tqdm import tqdm
import csv
tqdm.pandas()


df_random_sample_C_chunk = pd.read_csv('/home/chris/Downloads/m3inference/data-society-twitter-user-data/Twitter sample C (jonbruner)/df_random_sample3.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
    index_col=0, encoding = 'latin-1')
df_random_sample_C = pd.concat(df_random_sample_C_chunk)


#This sample only has user_location, no place or coordinates

df_random_sample_C = df_random_sample_C.rename(columns = {"handle":'screen_name'})

df_random_sample_C_users = df_random_sample_C.drop_duplicates(subset = ['screen_name'], keep = 'first')
df_random_sample_C_users = df_random_sample_C_users.rename(columns = {"location":'user_location'})

print(df_random_sample_C_users[['user_location']].notnull().sum())
#user_location    67342

df_random_sample_C_users_location = df_random_sample_C_users[['user_location']]
print(df_random_sample_C_users_location.shape)
#(290418, 1)
df_random_sample_C_users_location = df_random_sample_C_users_location.drop_duplicates(subset = 'user_location')
print(df_random_sample_C_users_location.shape)
#(37848, 1)
df_random_sample_C_users_location['user_location'] = df_random_sample_C_users_location['user_location'].astype(str)
df_random_sample_C_users_location['Mordercai_output'] = df_random_sample_C_users_location['user_location'].progress_apply(lambda x: geo.geoparse(x)) #3:05:54

row_length = []
for row in range(len(df_random_sample_C_users_location)):      
    row_length.append(len(df_random_sample_C_users_location['Mordercai_output'].iloc[row]))
max(row_length)
min(row_length)
#Max 4, min 0

Mordercai_location_0 = []
Mordercai_location_1 = []
Mordercai_location_2 = []
Mordercai_location_3 = []


for row in range(len(df_random_sample_C_users_location)):      
    if df_random_sample_C_users_location['Mordercai_output'].iloc[row] == []:
            Mordercai_location_0.append('null')
            Mordercai_location_1.append('null')
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
    else:    
        if len(df_random_sample_C_users_location['Mordercai_output'].iloc[row]) == 1:
            Mordercai_location_0.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append('null')
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
        if len(df_random_sample_C_users_location['Mordercai_output'].iloc[row]) == 2:
            Mordercai_location_0.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append('null')
            Mordercai_location_3.append('null')
        if len(df_random_sample_C_users_location['Mordercai_output'].iloc[row]) == 3:
            Mordercai_location_0.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append('null')
        if len(df_random_sample_C_users_location['Mordercai_output'].iloc[row]) == 4:
            Mordercai_location_0.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][0])
            Mordercai_location_1.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][1])
            Mordercai_location_2.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][2])
            Mordercai_location_3.append(df_random_sample_C_users_location['Mordercai_output'].iloc[row][3])

            
df_Mordercai = pd.DataFrame([df_random_sample_C_users_location['user_location'].tolist(), Mordercai_location_0, Mordercai_location_1, Mordercai_location_2, 
              Mordercai_location_3]).transpose()
df_Mordercai.columns = ['user_location', 'output_loc_0', 'output_loc_1', 'output_loc_2', 'output_loc_3']   
 
df_Mordercai = df_Mordercai.fillna('null')

df_Mordercai_loc_0 = df_Mordercai[['user_location', 'output_loc_0']]
df_Mordercai_loc_1 = df_Mordercai[['user_location', 'output_loc_1']]
df_Mordercai_loc_2 = df_Mordercai[['user_location', 'output_loc_2']]
df_Mordercai_loc_3 = df_Mordercai[['user_location', 'output_loc_3']]

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


import numpy as np
df_Mordercai_loc_combined = df_Mordercai_loc_0_exploded.merge(df_Mordercai_loc_1_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_2_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_Mordercai_loc_3_exploded, on = 'user_location', how = 'left')
df_Mordercai_loc_combined = df_Mordercai_loc_combined.replace('null', np.nan)

#Save to csv:
#df_Mordercai_loc_combined.to_csv('/home/chris/Downloads/tweet-ids-001/SarahEverard/df_sample_C_Mordercai_location_combined_120822.csv', sep=',', quotechar='"')
#Re-import:
#df_Mordercai_loc_combined = pd.read_csv('/home/chris/Downloads/tweet-ids-001/SarahEverard/df_sample_C_Mordercai_location_combined_120822.csv', sep=',', quotechar='"', indexcol=0)

#Pair df_Mordercai_loc_combined with 'id_str' from df_random_sample_C
df_random_sample_C = df_random_sample_C.rename(columns = {'location':'user_location'})
df_Mordercai_loc_combined = df_random_sample_C[['screen_name', 'name', 'user_location', 'last_post_lat', 'last_post_lon']].merge(df_Mordercai_loc_combined, on = 'user_location', how = 'left')
print(df_Mordercai_loc_combined.shape)
#(599023, 33)

#No user_id in original dataset, hence import the m3inference input (where user_id has been determined based on screen_name)
df_sample_C_m3input = pd.read_json('/home/chris/Downloads/m3inference/test/twitter_cache3/random_twitter_sample3_m3_input.jsonl', lines=True)
df_sample_C_m3input = df_sample_C_m3input[['screen_name', 'id']]
df_sample_C_m3input['id'] = df_sample_C_m3input['id'].astype(str)
df_sample_C_m3input = df_sample_C_m3input.drop_duplicates(subset = ['screen_name'])
print(df_sample_C_m3input.shape)
#(254370, 2)
df_Mordercai_loc_combined = df_Mordercai_loc_combined.merge(df_sample_C_m3input, on = 'screen_name', how = 'left')
print(df_Mordercai_loc_combined.shape)
#(599023, 34)

#Import the m3sampled output dataset from which control sample statistics were calculated
df_random_sample_result_chunk = pd.read_csv('/home/chris/Downloads/m3inference/test/twitter_cache3/random_twitter_sample3_m3_result.csv',
     sep=',', 
     quotechar='"', 
     chunksize=100000,
    index_col=0)
df_random_sample_result = pd.concat(df_random_sample_result_chunk)
print(df_random_sample_result.shape)
#((254370, 9), 9)

##Now make new column with Organisational status labelled
conditions_org= [
    (df_random_sample_result['org_non-org'] > 0.5),
    (df_random_sample_result['org_non-org'] < 0.5),
    (df_random_sample_result['org_non-org'] == 0.5)
    ]
values_org= ['Non_org', 'Is_org', 'Non-Org_unclear']
df_random_sample_result['Org'] = np.select(conditions_org, values_org)
df_random_sample_result['Org'].value_counts()
#Non_org    243567
#Is_org      10803
#Remove Org from dataset:
df_random_sample_result_org = df_random_sample_result.loc[df_random_sample_result['Org'] == 'Is_org']
df_random_sample_result = df_random_sample_result.loc[df_random_sample_result['Org'] == 'Non_org']
df_random_sample_result.shape
#(243567, 10)
#Assign Gender:
df_random_sample_result['Gender'] = np.where(df_random_sample_result.gender_male > 0.5, 'Male', 'Female')
#Calculate final numbers of Male vs Female for this random sample
df_random_sample_result.Gender.value_counts()
#Male      148657
#Female     94910

##Calculate age
df_random_sample_result['Age'] = df_random_sample_result[['age_<=18','age_19-29', 'age_30-39', 'age_>=40']].idxmax(axis=1)
print('Sample 1 age distribution (all) \n', df_random_sample_result['Age'].value_counts())
#age_19-29    93588
#age_>=40     62808
#age_<=18     51971
#age_30-39    35200

df_random_sample_result['id'] = df_random_sample_result['id'].astype(str)
df_random_sample_C_location = df_random_sample_result.merge(df_Mordercai_loc_combined, on = 'id', how = 'left')
print(df_random_sample_C_location.shape)
#(243567, 65)
df_random_sample_C_location['country_predicted_final'] = df_random_sample_C_location['country_predicted_loc_0']

n_location = df_random_sample_C_location['country_predicted_final'].notnull().sum()
print('Tweets with location data available:', n_location)
#27641
df_location_value_counts = pd.DataFrame(df_random_sample_C_location['country_predicted_final'].value_counts())
df_location_value_counts['percent'] = df_location_value_counts['country_predicted_final'] / n_location * 100
df_location_value_counts.head(10)

#     country_predicted_final    percent
#USA                     6571  23.772657
#IDN                     2928  10.592960
#GBR                     1870   6.765312
#IND                     1227   4.439058
#MEX                     1137   4.113455
#PHL                      852   3.082378
#CRI                      826   2.988314
#BRA                      729   2.637386
#CAN                      711   2.572266
#AUS                      591   2.138128

#Change to USA country code
countrycode_to_USA = [
    'Brooklyn, NY', 'Los Angeles, CA', 'Los Angeles', 'San Antonio, TX',
    'Santa Rosa, CA', 'Los Angeles, California', 'los angeles', 'San Antonio, Texas', 'Palo Alto, CA', 'Los Angeles',
    'California, USA', 'California', 'Southern California', 'Cali', 'california', 'Florida', 'Florida, USA', 'Florida '
]

for country in countrycode_to_USA:
    df_random_sample_C_location.loc[df_random_sample_C_location['user_location'] == country, 'country_predicted_final'] = 'USA'

print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'USA']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'IDN']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'GBR']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'IND']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'MEX']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'PHL']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'CRI']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'BRA']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'CAN']['user_location'].value_counts().head(10), '\n')
print(df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'AUS']['user_location'].value_counts().head(10), '\n')

#All the above are accurate

df_random_sample_C_location_GBR_only = df_random_sample_C_location[df_random_sample_C_location['country_predicted_final'] == 'GBR']

###GBR only sensitivity-analysis
#Calculate final numbers of Male vs Female for df_random_sample_C_location_GBR_only
df_random_sample_C_location_GBR_only_gender = pd.DataFrame(df_random_sample_C_location_GBR_only.Gender.value_counts())
df_random_sample_C_location_GBR_only_gender['percent'] = df_random_sample_C_location_GBR_only_gender['Gender'] / (df_random_sample_C_location_GBR_only['country_predicted_final'].notnull().sum()) * 100
#        Gender    percent
#Male       960  51.336898
#Female     910  48.663102
#Name: Gender, dtype: int64