# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:06:49 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necessary libraries:

import pandas as pd
import numpy as np
import logReg
import Adaline
import sys

sys.stdout = open('output.txt','wt')

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

# Loading and fitting our dataframe into the Logistic Regression class:

print("Dataset 1: Breast Cancer")

lr_1=logReg.logReg(df_1,xinds,len(df_1.iloc[0,:])-1)

w1,y1,e1,me1=lr_1.fitTest()

# Loading and fitting our dataframe into the Adaline class:

ad_1=Adaline.Adaline(df_1,xinds,len(df_1.iloc[0,:])-1)

wa1,ya1,ea1,mea1=ad_1.fitTest()



print("Logistic Regression Error: "+str(np.round(me1,2)))

print("Adaline Error: "+str(np.round(mea1,2)))

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

xinds=np.arange(0,len(df.iloc[0,:])-1).tolist()


# Loading and fitting our dataframe into the Logistic Regression class:

print("Dataset 2: Glass")

lr_2=logReg.logReg(df_1,xinds,len(df_1.iloc[0,:])-1)

w2,y2,e2,me2=lr_2.fitTest()

# Loading and fitting our dataframe into the Adaline class:

ad_2=Adaline.Adaline(df_1,xinds,len(df_1.iloc[0,:])-1)

wa2,ya2,ea2,mea2=ad_2.fitTest()

print("Logistic Regression Error: "+str(np.round(me2,2)))

print("Adaline Error: "+str(np.round(mea2,2)))

print()



# Dataset 3: Iris

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Iris\\iris.data',header=None)
cols=['sepal_length','sepal_width','petal_length',
      'petal_width','class']
df.columns=cols


# Minmax scalaring the atribute features for the atributes to be between -1 and 1:

minmaxdf=df.copy()
mins=df.min()
maxs=df.max()
minmaxdf.iloc[:,:-1]=(2*(minmaxdf.iloc[:,:-1]-mins[:-1])/(maxs[:-1]-mins[:-1]))-1
df_1=minmaxdf

# Defining the x-variables

xinds=np.arange(0,len(df_1.iloc[0,:])-1).tolist()


# Loading and fitting our dataframe into the Logistic Regression class:

print("Dataset 3: Iris")

lr_3=logReg.logReg(df_1,xinds,len(df_1.iloc[0,:])-1)

w3,y3,e3,me3=lr_3.fitTest()

# Loading and fitting our dataframe into the Adaline class:
    
ad_3=Adaline.Adaline(df_1,xinds,len(df_1.iloc[0,:])-1)

wa3,ya3,ea3,mea3=ad_3.fitTest()

print("Logistic Regression Error: "+str(np.round(me3,2)))

print("Adaline Error: "+str(np.round(mea3,2)))

print()


# Dataset 4: Soybean (small)
#
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


# Loading and fitting our dataframe into the Logistic Regression class:

print("Dataset 4: Soybean (small)")

lr_4=logReg.logReg(df_1,xinds,len(df_1.iloc[0,:])-1)

w4,y4,e4,me4=lr_4.fitTest()

# Loading and fitting our dataframe into the Adaline class:

ad_4=Adaline.Adaline(df_1,xinds,len(df_1.iloc[0,:])-1)
wa4,ya4,ea4,mea4=ad_4.fitTest()

print("Logistic Regression Error: "+str(np.round(me4,2)))

print("Adaline Error: "+str(np.round(mea4,2)))

print()



# Dataset 5: Vote

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Vote\\house-votes-84.data',header=None)
cols=['class_name','handicapped_infants','water_project_cost_sharing',
      'adoption-of-the-budget-resolution','physician-fee-freeze',
      'el-salvador-aid','religious-groups-in-schools',
      'anti-satellite-test-ban','aid-to-nicaraguan-contras',
      'mx-missile','immigration','synfuels-corporation-cutback',
      'education-spending','superfund-right-to-sue','crime',
      'duty-free-exports','export-administration-act-south-africa']
df.columns=cols

# Discretizing the features:

for index,row in df.iterrows():
    for i in range(len(row)):
        if(df.iloc[index,i]=='n'):
            df.iloc[index,i]=0
        elif(df.iloc[index,i]=='y'):
            df.iloc[index,i]=1
        elif(df.iloc[index,i]=='?'):
            df.iloc[index,i]=-1

# Defining the x-variables

xinds=np.arange(1,len(df.iloc[0,:])).tolist()

# Loading and fitting our dataframe into the Logistic Regression class:

print("Dataset 5: Vote")

lr_5=logReg.logReg(df,xinds,0)

w5,y5,e5,me5=lr_5.fitTest()

# Loading and fitting our dataframe into the Adaline class:

ad_5=Adaline.Adaline(df,xinds,0)

wa5,ya5,ea5,mea5=ad_5.fitTest()

print("Logistic Regression Error: "+str(np.round(me5,2)))

print("Adaline Error: "+str(np.round(mea5,2)))
