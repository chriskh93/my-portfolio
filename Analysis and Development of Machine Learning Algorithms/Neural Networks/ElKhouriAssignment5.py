# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:20:15 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necessary libraries:

import pandas as pd
import numpy as np
import backprop
import sys 

sys.stdout = open('output.txt','wt')
#
# Function bootstrap_sample is a function that generates sample data
# The function takes the following input parameters:
#   data: The data upon which to base the sample on
#   f: The function upon how to sample that data (mean/median)
#   n: The number of samples to generate

def bootstrap_sample( data, f, n=100):
    result = []
    for _ in range( n):
        sample = np.random.choice( data, len(data), replace=True)
        r = f( sample)
        result.append( r)
    return np.array( result)


# Dataset 1: Breast Cancer
#
# Loading the dataset and columns:

df=pd.read_csv('Datasets\Breast Cancer\\breast-cancer-wisconsin.data',header=None)
cols=['sample_code_number','clump_thickness','uniformity_cell_size',
      'uniformity_cell_shape','marginal_adhesion','single_epithelial_cell_size',
      'bare_nuclei','brand_chromatin','normal_nucleoli','mitosis','class']
df.columns=cols


# Handling the missing values:

df_1=df[df['bare_nuclei']!='?']

df_1_2=df_1[df_1['class']==2]
df_1_4=df_1[df_1['class']==4]

df_1_2['bare_nuclei']=df_1_2['bare_nuclei'].astype(int)


df_1_4['bare_nuclei']=df_1_4['bare_nuclei'].astype(int)


df_2=df[df['bare_nuclei']=='?']

missing_2s=bootstrap_sample( df_1_2['bare_nuclei'], np.median, len(df_2[df_2['class']==2]))
missing_4s=bootstrap_sample( df_1_4['bare_nuclei'], np.median, len(df_2[df_2['class']==4]))

i=0
j=0

for index,row in df.iterrows():
    if(row['class']==2):
        if(row['bare_nuclei']=='?'):
            df.iloc[index,6]=missing_2s[i]
            i=i+1
    elif(row['class']==4):
        if(row['bare_nuclei']=='?'):
            df.iloc[index,6]=missing_4s[j]
            j=j+1

df['bare_nuclei']=df['bare_nuclei'].astype(int)

df=df.iloc[:,1:]
cols=df.columns


# Minmax scalaring the atribute features for the atributes to be between -1 and 1:
minmaxdf=df.copy()
mins=df.min()
maxs=df.max()
minmaxdf.iloc[:,:-1]=(2*(minmaxdf.iloc[:,:-1]-mins[:-1])/(maxs[:-1]-mins[:-1]))-1
df_1=minmaxdf


# Defining the x-variables

xinds=np.arange(0,len(df.iloc[0,:])-1).tolist()

# Loading and fitting our dataframe into the Feedforward Neural Network class:

print("Dataset 1: Breast Cancer")

bp_1=backprop.backprop(df_1,xinds,len(df_1.iloc[0,:])-1)

w10,y10,e10,me10=bp_1.fitTest(nhl=0)
w11,y11,e11,me11=bp_1.fitTest(nhl=1)
w12,y12,e12,me12=bp_1.fitTest(nhl=2)


print("0 Hidden Layer Error: "+str(np.round(me10,2)))

print("1 Hidden Layer Error: "+str(np.round(me11,2)))

print("2 Hidden Layer Error: "+str(np.round(me12,2)))

print()



# Dataset 2: Glass

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Glass\\glass.data',header=None)
df.head()
cols=['id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
df.columns=cols

df_1=df.iloc[:,1:]


# Minmax scalaring the atribute features for the atributes to be between -1 and 1:

minmaxdf=df.copy()
mins=df.min()
maxs=df.max()
minmaxdf.iloc[:,:-1]=(2*(minmaxdf.iloc[:,:-1]-mins[:-1])/(maxs[:-1]-mins[:-1]))-1
df_1=minmaxdf

# Defining the x-variables

xinds=np.arange(0,len(df_1.iloc[0,:])-1).tolist()


# Loading and fitting our dataframe into the Feedforward Neural Network class:

print("Dataset 2: Glass")

bp_2=backprop.backprop(df_1,xinds,len(df_1.iloc[0,:])-1)

w20,y20,e20,me20=bp_2.fitTest(nhl=0)
w21,y21,e21,me21=bp_2.fitTest(nhl=1)
w22,y22,e22,me22=bp_2.fitTest(nhl=2)
print()

print("0 Hidden Layer Error: "+str(np.round(me20,2)))

print("1 Hidden Layer Error: "+str(np.round(me21,2)))

print("2 Hidden Layer Error: "+str(np.round(me22,2)))

print()


# Dataset 3: Soybean (small)

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Soybean\\soybean-small.data',header=None)


# Minmax scalaring the atribute features for the atributes to be between -1 and 1:

minmaxdf=df.copy()
cols=minmaxdf.columns
for i in range(len(cols)):
    if(len(minmaxdf.loc[:,cols[i]].unique())==1):
        minmaxdf.drop(cols[i],axis=1,inplace=True)
mins=df.min()
maxs=df.max()
minmaxdf.iloc[:,:-1]=(2*(minmaxdf.iloc[:,:-1]-mins[:-1])/(maxs[:-1]-mins[:-1]))-1
df_1=minmaxdf

# Defining the x-variables

xinds=np.arange(0,len(df_1.iloc[0,:])-1).tolist()


# Loading and fitting our dataframe into the Feedforward Neural Network class:

print("Dataset 3: Soybean (small)")


bp_3=backprop.backprop(df_1,xinds,len(df_1.iloc[0,:])-1)
w30,y30,e30,me30=bp_3.fitTest(nhl=0)
w31,y31,e31,me31=bp_3.fitTest(nhl=1)
w32,y32,e32,me32=bp_3.fitTest(nhl=2)


print("0 Hidden Layer Error: "+str(np.round(me30,2)))
#
print("1 Hidden Layer Error: "+str(np.round(me31,2)))

print("2 Hidden Layer Error: "+str(np.round(me32,2)))


print()



# Dataset 4: Abalone

# Loading the dataset and columns:
    
df=pd.read_csv('Datasets\Abalone\\abalone.data',header=None)
cols=['Sex','Length','Diameter','Height','whole_weight',
      'shucked_weight','viscera_weight','shell_weight','rings']
df.columns=cols
    
    
# Normalizing our values:

normalized_df=df.copy()
means=df.iloc[:,1:9].mean()
stds=df.iloc[:,1:9].std()
for i in range(1,9):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i-1])/stds[i-1]

# Creating dummy variables for row 'Sex':
    
normalized_df = pd.concat([normalized_df, pd.get_dummies(normalized_df["Sex"])], axis=1)
df_1=normalized_df.iloc[:,1:-1]


# Defining the x-variables

xinds=[0,1,2,3,4,5,6,8,9]

print("Dataset 4: Abalone")

# Loading and fitting our dataframe into the Feedforward Neural Network class:

bp_4=backprop.backprop(df_1,xinds,7,reg=True,norm=True,means=means,stds=stds)
w40,y40,e40,me40=bp_4.fitTest(nhl=0)
w41,y41,e41,me41=bp_4.fitTest(nhl=1)
w42,y42,e42,me42=bp_4.fitTest(nhl=2)


print("0 Hidden Layer Error: "+str(np.round(me40,2)))

print("1 Hidden Layer Error: "+str(np.round(me41,2)))

print("2 Hidden Layer Error: "+str(np.round(me42,2)))


print()


# Dataset 5: Computer Hardware

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Computer Hardware\\machine.data',header=None)
cols=['vendor_name','model_name','MYCT',
      'MMIN','MMAX','CACH','CHMIN','CHMAX',
      'PRP','ERP']
df.columns=cols
df.describe()

# Normalizing our values:

normalized_df=df.copy()
means=df.iloc[:,2:9].mean()
stds=df.iloc[:,2:9].std()
for i in range(2,9):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i-2])/stds[i-2]
df_1=normalized_df.iloc[:,2:9]

print("Dataset 5: Computer Hardware")


# Loading and fitting our dataframe into the Feedforward Neural Network class:

bp_5=backprop.backprop(df_1,np.arange(6).tolist(),6,reg=True,norm=True,means=means,stds=stds)
w50,y50,e50,me50=bp_5.fitTest(nhl=0)
w51,y51,e51,me51=bp_5.fitTest(nhl=1)
w52,y52,e52,me52=bp_5.fitTest(nhl=2)


print("0 Hidden Layer Error: "+str(np.round(me50,2)))

print("1 Hidden Layer Error: "+str(np.round(me51,2)))

print("2 Hidden Layer Error: "+str(np.round(me52,2)))


print()


# Dataset 6: Forest Fires

# Loading the dataset:

df=pd.read_csv('Datasets\Forest Fires\\forestfires.csv')

# Dropping columns 'day' and 'month'

df.drop(['day','month'],axis=1,inplace=True)

# Normalizing our values:

normalized_df=df.copy()
means=df.mean()
stds=df.std()
for i in range(len(df.iloc[0,:])):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i])/stds[i]
df_1=normalized_df



print("Dataset 6: Forest Fires")

# Defining the x-variables

xinds=np.arange(0,len(df.iloc[0,:])-1).tolist()

# Loading and fitting our dataframe into the Feedforward Neural Network class:

bp_6=backprop.backprop(df_1,xinds,len(df.iloc[0,:])-1,reg=True,norm=True,means=means,stds=stds)
w60,y60,e60,me60=bp_6.fitTest(nhl=0)
w61,y61,e61,me61=bp_6.fitTest(nhl=1)
w62,y62,e62,me62=bp_5.fitTest(nhl=2)

print("0 Hidden Layer Error: "+str(np.round(me60,2)))

print("1 Hidden Layer Error: "+str(np.round(me61,2)))

print("2 Hidden Layer Error: "+str(np.round(me62,2)))

