# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:22:46 2021

@author: CHRIS
"""

import pandas as pd
import numpy as np

df_pg=pd.read_csv('product_groups_271120.csv')
df_pz=pd.read_csv('product_group_sizes.csv')



df_pg_lt=df_pg.copy()
df_pg_lt['TB']=0

df_pg_lt['TB']=df_pg_lt['type'].apply(lambda x: 1 if x in ['Bottoms','Tops'] else 0)

df_pg_lt=df_pg_lt[df_pg_lt['TB']==1].reset_index(drop=True)

types=df_pg_lt['type'].unique()
df_fin=pd.DataFrame(columns=['Type','Brand','Category','Fit','Size_type'])
df_pg_lt['size_type']=0

for index,row in df_pg_lt.iterrows():
    df_st=df_pz[df_pz['product_group_id']==df_pg_lt.iloc[index,0]].reset_index(drop=True)
    if(len(df_st)>0):
        if(str(df_st.iloc[0,1])[0].isnumeric()):
            df_pg_lt.iloc[index,-1]='Numeric'
        else:
            df_pg_lt.iloc[index,-1]='Categorical'
    
for l in range(len(types)):
    df_b=df_pg_lt[df_pg_lt['type']==types[l]]
    brands=df_b['brand'].unique()
    for i in range(len(brands)):
        df_b1=df_b[df_b['brand']==brands[i]]
        cats=df_b1['category'].unique()
        for j in range(len(cats)):
            fits=df_b1[df_b1['category']==cats[j]]['fit'].unique()
            df_b2=df_b1[df_b1['category']==cats[j]]
            for k in range(len(fits)):
                df_b3=df_b2[df_b2['fit']==fits[k]]
                x=[]
                x.append(types[l])
                x.append(brands[i])
                x.append(cats[j])
                x.append(fits[k])
                st=df_b3['size_type'].mode().iloc[0]
                x.append(st)
                df_x=pd.DataFrame(data=x)
                df_x=df_x.transpose()
                df_x.columns=['Type','Brand','Category','Fit','Size_type']
                df_fin=pd.concat([df_fin,df_x])
df_fin['Size_type'].unique().tolist()
df_fin=df_fin.reset_index(drop=True)
df_fin['Size']=np.nan
df_fin['Found']=np.nan
df_fin=df_fin[df_fin['Size_type']!='0'].reset_index(drop=True)

df_fin.to_csv('df_fin_1.csv')

        
        
        