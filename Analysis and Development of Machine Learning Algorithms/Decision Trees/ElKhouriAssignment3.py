# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:53 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necessary libraries:

import pandas as pd
import numpy as np
import decisiontree
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
df_2=df[df['class']==2]
df_4=df[df['class']==4]

# Discretizing attribute values based on the midpoint of medians of each of the classes:

for i in range(len(cols)-1):
    mid=np.percentile(df_2.iloc[:,i],50)+np.percentile(df_4.iloc[:,i],50)
    mid=mid/2
    mid=mid+np.percentile(df_2.iloc[:,i],50)
    df[cols[i]+" <"+str(mid)]=df[cols[i]].apply(lambda x:"Yes" if x<mid else "No" )
    
# Preparing our dataframe for the decision tree by only keeping the disctrete features
    
df=df.iloc[:,len(cols)-1:]

# Defining the x-variables

xinds=np.arange(1,len(df.iloc[0,:])).tolist()

# Loading and fitting our dataframe into the decision tree class:

print("Dataset 1: Breast Cancer")

dt_1=decisiontree.decisiontree(df,xinds,0)

enps1,eps1,enp1,ep1,tnp1,tp1=dt_1.ID3fit(True)

# enps1: The fold errors without pruning
# eps1: The fold errors with pruning
# enp1: The mean of the fold errors without pruning
# ep1: The mean of the fold errors with pruning
# tnp1: The generated unpruned trees
# tp1: The generated pruned trees

print("Non-Pruned Error: "+str(np.round(enp1,2)))
if(ep1!=np.nan):
    print("Pruned Error: "+str(np.round(ep1,2)))


# Dataset 2: Car Evaluation

# Loading the dataset and columns:
    
df=pd.read_csv('Datasets\Car Evaluation\\car.data',header=None)
cols=['buying','maint','doors','persons','lug_boot','safety','class']
df.columns=cols

xinds=np.arange(0,len(df.iloc[0,:])-1).tolist()

print("Dataset 2: Car Evaluation")

dt_2=decisiontree.decisiontree(df,xinds,6)
enps2,eps2,enp2,ep2,tnp2,tp3=dt_2.ID3fit(True)

# enps2: The fold errors without pruning
# eps2: The fold errors with pruning
# enp2: The mean of the fold errors without pruning
# ep2: The mean of the fold errors with pruning
# tnp2: The generated unpruned trees
# tp2: The generated pruned trees

print("Non-Pruned Error: "+str(np.round(enp2,2)))
if(ep2!=np.nan):
    print("Pruned Error: "+str(np.round(ep2,2)))


# Dataset 3: Image Segmentation
#
# Loading the dataset:

df=pd.read_csv('Datasets\Image Segmentation\\segmentation.data')
df.head()
df=df.reset_index()
df=df.rename(columns={'index':'Class'})

y_cats=df.iloc[:,0].unique()

# Removing the column "REGION-PIXEL-COUNT" due to all rows having the same value:

df=df.drop('REGION-PIXEL-COUNT',axis=1)

# Calculating the medians:

cols=df.columns
mids=[]
for i in range(1,len(cols)):
    mid1=[]
    for j in range(len(y_cats)):
        df_1=df[df.iloc[:,0]==y_cats[j]]
        mid1.append(np.percentile(df_1.iloc[:,i],50))
    mids.append(mid1)

# Removing duplicate medians:
    
for i in range(len(mids)):
    mids[i]=np.sort(mids[i]).tolist()


for i in range(len(mids)):
    for j in range(len(mids[i])):
        if(j==len(mids[i])):
            break
        k=0
        while k <(len(mids[i])):
            if(k==len(mids[i])):
                break
            if(j!=k):
                if(mids[i][j]==mids[i][k]):
                    mids[i].remove(mids[i][k])

                else:
                    k+=1
            else:
                k+=1

# Calculating the midpoint of the medians to decrease our tree size:
                
mids2=[]

for i in range(len(mids)):
    mids21=[]
    j=0
    while j<len(mids[i]):
        if(len(mids[i])>1):
            if((j+1)>=len(mids[i])):
                mids21.append(mids[i][j])
            else:
                m=mids[i][j]+mids[i][j+1]
                m=m/2
                mids21.append(m)
        else:
            mids21.append(mids[i][j])
        j+=2
    mids2.append(mids21)

# Further calculating the midpoint of the medians to further decrease our tree size:
    
mids3=[]

for i in range(len(mids2)):
    mids21=[]
    j=0
    while j<len(mids2[i]):
        if(len(mids2[i])>1):
            if((j+1)>=len(mids2[i])):
                mids21.append(mids2[i][j])
            else:
                m=mids2[i][j]+mids2[i][j+1]
                m=m/2
                mids21.append(m)
        else:
            mids21.append(mids2[i][j])
        j+=2
    mids3.append(mids21)


# Dicretizing the feature attributes based on whether or not they are <= 
#   to the calculated midpoints:
    
for i in range(1,len(cols)):
    for j in range(len(mids3[i-1])):
        
        df[cols[i]+" <="+str(mids3[i-1][j])]=df[cols[i]].apply(lambda x:"Yes" if x<=mids3[i-1][j] else "No" )


# Preparing the dataframe for our decision tree class:
        
df_1=pd.concat([df.iloc[:,0],df.iloc[:,19:]],axis=1)

xinds=np.arange(1,len(df_1.iloc[0,:])).tolist()

print("Dataset 3: Image Segmentation")

dt_3=decisiontree.decisiontree(df_1,xinds,0)
enps3,eps3,enp3,ep3,tnp3,tp3=dt_3.ID3fit(True)

# enps3: The fold errors without pruning
# eps3: The fold errors with pruning
# enp3: The mean of the fold errors without pruning
# ep3: The mean of the fold errors with pruning
# tnp3: The generated unpruned trees
# tp3: The generated pruned trees


print("Non-Pruned Error: "+str(np.round(enp3,2)))
if(ep3!=np.nan):
    print("Pruned Error: "+str(np.round(ep3,2)))

        
        
        
# Dataset 4: Abalone

# Loading the dataset and columns:
    
df=pd.read_csv('Datasets\Abalone\\abalone.data',header=None)
cols=['Sex','Length','Diameter','Height','whole_weight',
      'shucked_weight','viscera_weight','shell_weight','rings']
df.columns=cols

# Discretizing the feature attributes by calculating the median of each feature 
# and returning "Yes" or "No" whether or not the feature values are < than the 
# median values:


medians=df.median()
for i in range(1,len(cols)-1):
    df[cols[i]+" <"+str(medians[i-1])]=df[cols[i]].apply(lambda x:"Yes" if x<medians[i-1] else "No" )


# Setting up our dataframe for our decision tree class by only leaving the target
# variable and our discretized features in the dataframe
    
    
df=df.iloc[:,8:]

# Normalizing our target variable for comparative analysis of the MSE:


normalized_df=df.copy()
means=df.mean()
stds=df.std()
normalized_df.iloc[:,0]=(normalized_df.iloc[:,0]-means[0])/stds[0]
df_1=normalized_df

# Loading our data into the decision tree class:


xinds=np.arange(1,len(df_1.iloc[0,:])).tolist()

print("Dataset 4: Abalone")
dt_4=decisiontree.decisiontree(df_1,xinds,0,reg=True,norm=True,means=means,stds=stds)
enes4,ees4,ene4,ee4,tne4,tee4=dt_4.CARTfit(True)

# enes4: The fold errors without early stopping
# ees4: The fold errors with early stopping
# ene4: The mean of the fold errors without early stopping
# ee4: The mean of the fold errors with early stopping
# tne4: The generated trees without early stopping
# tee4: The generated trees with early stopping

print("No Early Stopping MSE: "+str(np.round(ene4,2)))
if(ee4!=np.nan):
    print("Early Stopping MSE: "+str(np.round(ee4,2)))


# Dataset 5: Computer Hardware

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Computer Hardware\\machine.data',header=None)
cols=['vendor_name','model_name','MYCT',
      'MMIN','MMAX','CACH','CHMIN','CHMAX',
      'PRP','ERP']
df.columns=cols

# Omitting the first 2 columns:

df=df.iloc[:,2:-1]

# Discretizing our feature attributes as described above:

medians=df.median()
cols=df.columns
for i in range(len(cols)-1):
    df[cols[i]+" <"+str(medians[i])]=df[cols[i]].apply(lambda x:"Yes" if x<medians[i] else "No" )

# Preparing our dataframe for our decision tree class:
    
df=df.iloc[:,6:]

# Normalizing our target variable:

means=df.mean()
stds=df.std()

normalized_df=df.copy()
means=df.mean()
stds=df.std()
normalized_df.iloc[:,0]=(normalized_df.iloc[:,0]-means[0])/stds[0]
df_1=normalized_df

# Loading our dataframe into the decision tree class:

xinds=np.arange(1,len(df_1.iloc[0,:])).tolist()

print("Dataset 5: Computer Hardware")
dt_5=decisiontree.decisiontree(df_1,xinds,0,reg=True,norm=True,means=means,stds=stds)
enes5,ees5,ene5,ee5,tne5,tee5=dt_5.CARTfit(True)

# enes5: The fold errors without early stopping
# ees5: The fold errors with early stopping
# ene5: The mean of the fold errors without early stopping
# ee5: The mean of the fold errors with early stopping
# tne5: The generated trees without early stopping
# tee5: The generated trees with early stopping

print("No Early Stopping MSE: "+str(np.round(ene5,2)))
if(ee5!=np.nan):
    print("Early Stopping MSE: "+str(np.round(ee5,2)))



# Dataset 6: Forest Fires

# Loading the dataset:

df=pd.read_csv('Datasets\Forest Fires\\forestfires.csv')

# Dropping the month and day columns:

df=df.drop(['month','day'],axis=1)

# Discretizing our feature attributes:

medians=df.median()
cols=df.columns.tolist()

for i in range(len(cols)-1):
    df[cols[i]+" <="+str(medians[i])]=df[cols[i]].apply(lambda x:"Yes" if x<=medians[i] else "No" )

# Preparing our dataframe:
    
df=df.iloc[:,10:]



# Normalizing the target variable:

normalized_df=df.copy()
means=df.mean()
stds=df.std()
normalized_df.iloc[:,0]=(normalized_df.iloc[:,0]-means[0])/stds[0]
df_1=normalized_df

# Loading our dataframe into the decision tree class:

xinds=np.arange(1,len(df_1.iloc[0,:])).tolist()

print("Dataset 6: Forest Fires")
dt_6=decisiontree.decisiontree(df_1,xinds,0,reg=True,norm=True,means=means,stds=stds)
enes6,ees6,ene6,ee6,tne6,tee6=dt_6.CARTfit(True,med=True)

# enes6: The fold errors without early stopping
# ees6: The fold errors with early stopping
# ene6: The mean of the fold errors without early stopping
# ee6: The mean of the fold errors with early stopping
# tne6: The generated trees without early stopping
# tee6: The generated trees with early stopping

print("No Early Stopping MSE: "+str(np.round(ene6,2)))
if(ee6!=np.nan):
    print("Early Stopping MSE: "+str(np.round(ee6,2)))

