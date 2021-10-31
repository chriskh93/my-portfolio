# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:10:35 2021

@author: CHRIS
"""

# Importing necesarry libraries:

import pandas as pd
import numpy as np
import math

# Reading the provided datasets:

df_up=pd.read_csv('user_products_271120.csv')
df_u=pd.read_csv('users_271120.csv')
df_p=pd.read_csv('product_groups_271120.csv')


df_up_1=df_up.copy()


df_pg_lt=df_p.copy()

df_pg_lt['TB']=0

df_pg_lt['TB']=df_pg_lt['type'].apply(lambda x: 1 if x in ['Bottoms','Tops'] else 0)

df_pg_lt=df_pg_lt[df_pg_lt['TB']==1].reset_index(drop=True)

df_up_1['drop']=1


df_up_1['reason']=df_up_1['reason'].str.upper()

for index,row in df_up_1.iterrows():
    if(df_up_1.iloc[index,9]=='yes'):
        df_up_1.iloc[index,-1]=0
    else:
        if(df_up_1.iloc[index,7]=='TOO BIG' or df_up_1.iloc[index,7]=='TOO SMALL'):
            df_up_1.iloc[index,-1]=0

df_up_1=df_up_1[df_up_1['drop']==0].reset_index(drop=True)


df_up_1['fit']=0
df_up_1['user_waist_size']=0
df_up_1['user_shirt_size']=0
df_up_1['user_blazer_size']=0
df_up_1['user_weight']=0
df_up_1['user_height']=0
df_up_1['type']=0
df_up_1['user_shirts_fit']=0
df_up_1['user_pants_fit']=0

for index,row in df_up_1.iterrows():
    df_up_1.iloc[index,-9]=df_p[df_p['product_group_id']==df_up_1.iloc[index,2]]['fit'].iloc[0]
    df_up_1.iloc[index,-8]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['waist_size'].iloc[0]
    df_up_1.iloc[index,-7]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['shirt_size'].iloc[0]
    df_up_1.iloc[index,-6]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['blazer_size'].iloc[0]
    df_up_1.iloc[index,-5]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['weight'].iloc[0]
    df_up_1.iloc[index,-4]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['height'].iloc[0]
    df_up_1.iloc[index,-3]=df_p[df_p['product_group_id']==df_up_1.iloc[index,2]]['type'].iloc[0]
    df_up_1.iloc[index,-2]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['shirts_fit'].iloc[0]
    df_up_1.iloc[index,-1]=df_u[df_u['user_id']==df_up_1.iloc[index,1]]['pants_fit'].iloc[0]

df_up_1_final=df_up_1.iloc[:,[1,3,5,7,9,10,12,13,14,15,16,17,18,19,20]].reset_index(drop=True)


df_up_1_final=df_up_1_final[pd.isnull(df_up_1_final['user_weight'])!=True].reset_index(drop=True)

df_up_1_final=df_up_1_final[pd.isnull(df_up_1_final['user_height'])!=True].reset_index(drop=True)

df_up_1_final=df_up_1_final.iloc[:,[0,10,11,13,14,7,8,9,12,2,1,6,4,3,5]]

df_up_1_final.to_csv('df_up_1_final_3.csv')
    
