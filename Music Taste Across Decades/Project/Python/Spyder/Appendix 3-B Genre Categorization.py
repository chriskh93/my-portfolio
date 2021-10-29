# -*- coding: utf-8 -*-
"""


@author: Christopher El Khouri
"""



import pandas as pd
import numpy as np

df=pd.read_csv('spot_up_00s.csv')
df=df.iloc[:,1:]
df['genres']=0
main_genres=pd.DataFrame(columns=['Pop','R&B','Hip hop',
                                  'Rock','Metal','Country',
                                  'Latin','Caribbean','Blues',
                                  'Jazz','Electronic','Folk',
                                  'Classical','Flamenco','Avant-garde',
                                  'Comedy','Easy listening','Max'])
avantgarde=['Avant-garde','Experimental','Noise','Outsider music','Lo-fi',
            'Musique concrète','Electroacoustic','outsider']
caribbean=['caribbean','Baithak Gana','Dancehall','Bouyon','Cadence-lypso'
           ,'Calypso','Cha-cha-chá','Chutney','Compas',
           'Mambo','Merengue','Méringue','Mozambique',
           'Pichakaree','Punta','Rasin','Reggae','Ragga'
           ,'Reggaeton','Rocksteady','Rumba','Ska','Two-tone'
           ,'Salsa','Son cubano','Songo','Soca','Timba','Twoubadou'
           ,'Zouk']
comedy=['comedy','novelty','parody']
country=['country','bluegrass','nashville sound','cowboy']
easylistening=['easy listening','Background music','Beautiful music','Elevator music'
               ,'Furniture music','Lounge music','Middle of the road music'
               ,'New-age music','calming instrumental','atmosphere','lounge',
               'environmental','sleep','healing','meditation','new age']
electronic=['House','Electro','Electronic','Trance','Dubstep','Chillstep','Downtempo','Techno','glitch hop','broken beat','freestyle','miami bass','hardcore','ambient']
flamenco=['flamenco','Tona','Soleas','Fandangos','Tango','Cantes de ida y vuelta','copla']
hiphop=['hip hop','rap','trap','chillhop']
rnb=['r&b','rnb','rhythm and blues','soul','disco','funk','new jack swing','go-go','doo-wop','motown']
classical=['classical','baroque','ballet','late romantic era','cello','orchestra','classic','early romantic era']
latin=['axe','banda','latin','brega','grupera','cumbia','cancion melodica','bolero','forro','ranchera']
metal=['metal','neo-crust']
rock=['rock','punk','new wave','corrosion','freakbeat','zeuhl']
jazz=['jazz','bossa nova']
pop=['pop','chanson','shibuya-kei','boy band','instrumental surf','deep surf','surf music']
blues=['blues','gospel']
folk=['folk','skiffle']



for index,row in df.iterrows():
    main_genres.loc[index,:]=0
    
    j=0
    for i in range(0,len(pop)):
        if((row['genres_1'].lower()).find(pop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(pop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(pop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(pop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(rnb)):
        if((row['genres_1'].lower()).find(rnb[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(rnb[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(rnb[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(rnb[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(hiphop)):
        if((row['genres_1'].lower()).find(hiphop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(hiphop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(hiphop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(hiphop[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(rock)):
        if((row['genres_1'].lower()).find(rock[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(rock[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(rock[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(rock[i].lower())!=-1):
            main_genres.iloc[index,j]+=1 
    
    j+=1
    for i in range(0,len(metal)):
        if((row['genres_1'].lower()).find(metal[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(metal[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(metal[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(metal[i].lower())!=-1):
            main_genres.iloc[index,j]+=1 
    
    j+=1
    for i in range(0,len(country)):
        if((row['genres_1'].lower()).find(country[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(country[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(country[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(country[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(latin)):
        if((row['genres_1'].lower()).find(latin[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(latin[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(latin[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(latin[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(caribbean)):
        if((row['genres_1'].lower()).find(caribbean[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(caribbean[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(caribbean[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(caribbean[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(blues)):
        if((row['genres_1'].lower()).find(blues[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(blues[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(blues[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(blues[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
    
    j+=1
    for i in range(0,len(jazz)):
        if((row['genres_1'].lower()).find(jazz[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(jazz[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(jazz[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(jazz[i].lower())!=-1):
            main_genres.iloc[index,j]+=1 
    
    j+=1
    for i in range(0,len(electronic)):
        if((row['genres_1'].lower()).find(electronic[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(electronic[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(electronic[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(electronic[i].lower())!=-1):
            main_genres.iloc[index,j]+=1 
    
    j+=1
    for i in range(0,len(folk)):
        if((row['genres_1'].lower()).find(folk[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(folk[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(folk[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(folk[i].lower())!=-1):
            main_genres.iloc[index,j]+=1 

    j+=1
    for i in range(0,len(classical)):
        if((row['genres_1'].lower()).find(classical[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(classical[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(classical[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(classical[i].lower())!=-1):
            main_genres.iloc[index,j]+=1

    j+=1
    for i in range(0,len(flamenco)):
        if((row['genres_1'].lower()).find(flamenco[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(flamenco[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(flamenco[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(flamenco[i].lower())!=-1):
            main_genres.iloc[index,j]+=1

    j+=1
    for i in range(0,len(avantgarde)):
        if((row['genres_1'].lower()).find(avantgarde[i].lower())!=-1):
            main_genres.loc[index,'Avant-garde']+=1
        if((row['genres_2'].lower()).find(avantgarde[i].lower())!=-1):
            main_genres.loc[index,'Avant-garde']+=1
        if((row['genres_3'].lower()).find(avantgarde[i].lower())!=-1):
            main_genres.loc[index,'Avant-garde']+=1
        if((row['genres_4'].lower()).find(avantgarde[i].lower())!=-1):
            main_genres.loc[index,'Avant-garde']+=1

    j+=1
    for i in range(0,len(comedy)):
        if((row['genres_1'].lower()).find(comedy[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(comedy[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(comedy[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(comedy[i].lower())!=-1):
            main_genres.iloc[index,j]+=1

    j+=1
    for i in range(0,len(easylistening)):
        if((row['genres_1'].lower()).find(easylistening[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_2'].lower()).find(easylistening[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_3'].lower()).find(easylistening[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
        if((row['genres_4'].lower()).find(easylistening[i].lower())!=-1):
            main_genres.iloc[index,j]+=1
       
    j+=1
    max=0
    for k in range(0,j):
        if(main_genres.iloc[index,k]>max):
            max=main_genres.iloc[index,k]
            main_genres.iloc[index,j]=main_genres.columns[k]
    if(max==0):
        main_genres.iloc[index,j]='Other'

df.iloc[:,25]=main_genres.loc[:,'Max']
df.to_csv('spot_up_2_00s.csv')