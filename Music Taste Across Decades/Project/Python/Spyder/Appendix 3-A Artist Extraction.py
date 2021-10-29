# -*- coding: utf-8 -*-
"""


@author: Christopher El Khouri
"""



import pandas as pd
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials
cid ="bdfd824b4fee47edb12275aad832d3f8"
secret = "25423f1b59e841d7850cb8d998cb622b"
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

df=pd.read_csv('../Datasets/dataset-of-10s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_10s.csv')



df=pd.read_csv('../Datasets/dataset-of-00s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_00s.csv')



df=pd.read_csv('../Datasets/dataset-of-90s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_90s.csv')


df=pd.read_csv('../Datasets/dataset-of-80s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_80s.csv')

df=pd.read_csv('../Datasets/dataset-of-70s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_70s.csv')

df=pd.read_csv('../Datasets/dataset-of-60s.csv')
df['track id']=0
df['artist id']=0
for index,row in df.iterrows():
    df.iloc[index,19]=row['uri'][14:]

for index,row in df.iterrows():
    df.iloc[index,20]=sp.track(row['track id'])['artists'][0]['id']

df['genres_1']=0
df['genres_2']=0
df['genres_3']=0
df['genres_4']=0

maxg=0

i=0

for index,row in df.iterrows():
    for i in range(0,len(sp.artist(row['artist id'])['genres'])):
        if(i==4):
            break
        df.iloc[index,21+i]=sp.artist(row['artist id'])['genres'][i]

df.to_csv('spot_up_60s.csv')
    
