# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:47:38 2020

@author: Christopher El Khouri
        605.649.81
"""
import pandas as pd
import numpy as np
import knn_class
import sys
import knn_reg
import enn_reg
import cnn_reg
import cnn_class
import enn_class


sys.stdout = open('output7.txt','wt')

# Dataset 1: Glass

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Glass\\glass.data',header=None)
df.head()
cols=['id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
df.columns=cols
typesg=df['Type'].unique()
d=df.describe()

# Normalizing the features:

normalized_df=df.copy()
means=df.iloc[:,1:10].mean()
stds=df.iloc[:,1:10].std()
for i in range(1,10):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i-1])/stds[i-1]



print("Dataset 1: Glass")

# K-NN Classifier:
# Loading the data:

knn_glass=knn_class.knn_class(normalized_df,range(1,10),10)

# Tuning the parameters:

bk_k_1=knn_glass.tune()

# Fitting and testing:

accs_k_1,acc_k_1=knn_glass.fit()

print()
print("Dataset 1: Glass")
print("K-NN Classification Accuracy:"+str(acc_k_1))
print()

# E-NN Classifier:
# Loading the data:

enn_glass=enn_class.enn_class(normalized_df,range(1,10),10)

# Tuning the parameters:

bk_ek_1=enn_glass.tune()

# Fitting and testing:

accs_ek_1,acc_ek_1=enn_glass.fit()

print()
print("Dataset 1: Glass")
print("E-NN Classification Accuracy:"+str(acc_ek_1))
print()

# C-NN Classifier:
# Loading the data:

cnn_glass=cnn_class.cnn_class(normalized_df,range(1,10),10)

# Tuning the parameters:

bk_ck_1=cnn_glass.tune()

# Fitting and testing:

accs_ck_1,acc_ck_1=cnn_glass.fit()

print()
print("Dataset 1: Glass")
print("C-NN Classification Accuracy:"+str(acc_ck_1))
print()



# Dataset 2: Image Segmentation

# Loading the dataset:

df=pd.read_csv('Datasets\Image Segmentation\\segmentation.data')
df.head()
df=df.reset_index()
df=df.rename(columns={'index':'Class'})


print("Dataset 2: Image Segmentation")

# K-NN Classifier:
# Loading the data:

knn_image=knn_class.knn_class(df,range(1,20),0)

# Tuning the parameters:

bk_k_2=knn_image.tune()

# Fitting and testing:

accs_k_2,acc_k_2=knn_image.fit()

print()
print("Dataset 2: Image Segmentation")
print("K-NN Classification Accuracy:"+str(acc_k_2))
print()

# E-NN Classifier:
# Loading the data:

enn_image=enn_class.enn_class(df,range(1,20),0)

# Tuning the parameters:

bk_ek_2=enn_image.tune()

# Fitting and testing:

accs_ek_2,acc_ek_2=enn_image.fit()

print()
print("Dataset 2: Image Segmentation")
print("E-NN Classification Accuracy:"+str(acc_ek_2))
print()

# C-NN Classifier:
# Loading the data:

cnn_image=cnn_class.cnn_class(df,range(1,20),0)

# Tuning the parameters:

bk_ck_2=cnn_image.tune()

# Fitting and testing:

accs_ck_2,acc_ck_2=cnn_image.fit()

print()
print("Dataset 2: Image Segmentation")
print("C-NN Classification Accuracy:"+str(acc_ck_2))
print()



# Dataset 3: Vote

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
classv=df['class_name'].unique()

# Converting string values to numerical values

for index,row in df.iterrows():
    for i in range(len(row)):
        if(df.iloc[index,i]=='n'):
            df.iloc[index,i]=0
        elif(df.iloc[index,i]=='y'):
            df.iloc[index,i]=1
        elif(df.iloc[index,i]=='?'):
            df.iloc[index,i]=-1

print("Dataset 3: Vote")

# K-NN Classifier:
# Loading the data:

knn_vote=knn_class.knn_class(df,range(1,17),0)

# Tuning the parameters:

bk_k_3=knn_vote.tune()

# Fitting and testing:

accs_k_3,acc_k_3=knn_vote.fit()

print()
print("Dataset 3: Vote")
print("K-NN Classification Accuracy:"+str(acc_k_3))
print()

# E-NN Classifier:
# Loading the data:

enn_vote=enn_class.enn_class(df,range(1,17),0)

# Tuning the parameters:

bk_ek_3=enn_vote.tune()

# Fitting and testing:

accs_ek_3,acc_ek_3=enn_vote.fit()

print()
print("Dataset 3: Vote")
print("E-NN Classification Accuracy:"+str(acc_ek_3))
print()

# C-NN Classifier:
# Loading the data:

cnn_vote=cnn_class.cnn_class(df,range(1,17),0)

# Tuning the parameters:

bk_ck_3=cnn_vote.tune()

# Fitting and testing:

accs_ck_3,acc_ck_3=cnn_vote.fit()

print()
print("Dataset 3: Vote")
print("C-NN Classification Accuracy:"+str(acc_ck_3))
print()



# Dataset 4: Abalone

# Loading the dataset and columns:
df=pd.read_csv('Datasets\Abalone\\abalone.data',header=None)
cols=['Sex','Length','Diameter','Height','whole_weight',
      'shucked_weight','viscera_weight','shell_weight','rings']
df.columns=cols
df.describe()

# Normalizing:

normalized_df=df.copy()
means=df.iloc[:,1:9].mean()
stds=df.iloc[:,1:9].std()
for i in range(1,9):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i-1])/stds[i-1]

# Creating dummy variables for row 'Sex':
    
normalized_df = pd.concat([normalized_df, pd.get_dummies(normalized_df["Sex"])], axis=1)
df_1=normalized_df.iloc[:,1:-1]

print("Dataset 4: Abalone")

# K-NN Regression:
# Loading the data:

knn_abalone=knn_reg.knn_reg(df_1,[0,1,2,3,4,5,6,8,9],7,means,stds,True)

# Tuning the parameters:

k_4,s_4=knn_abalone.tune()

# Fitting and testing:

mses_k4,mse_k4=knn_abalone.knn_fit()

print()
print("Dataset 4: Abalone")
print("K-NN Regression MSE:"+str(mse_k4))
print()

# E-NN Regression:
# Loading the data:

enn_abalone=enn_reg.enn_reg(df_1,[0,1,2,3,4,5,6,8,9],7,means,stds,True)

# Tuning the parameters:

ek_4,es_4,ee_4=enn_abalone.tune()

# Fitting and testing:

mses_ek4,mse_ek4=enn_abalone.fit()

print()
print("Dataset 4: Abalone")
print("E-NN Regression MSE:"+str(mse_ek4))
print()

# C-NN Regression:
# Loading the data:

#cnn_abalone=cnn_reg.cnn_reg(df_1,[0,1,2,3,4,5,6,8,9],7,means,stds,True)
#
## Tuning the parameters:
#
#ck_4,cs_4,ce_4=cnn_abalone.tune()
#
## Fitting and testing:
#mses_ck4,mse_ck4=cnn_abalone.fit()
#
#print()
#print("Dataset 4: Abalone")
#print("C-NN Regression MSE:"+str(mse_ck4))
#print()




# Dataset 5: Computer Hardware

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Computer Hardware\\machine.data',header=None)
cols=['vendor_name','model_name','MYCT',
      'MMIN','MMAX','CACH','CHMIN','CHMAX',
      'PRP','ERP']
df.columns=cols
df.describe()

# Normalizing:

normalized_df=df.copy()
means=df.iloc[:,2:9].mean()
stds=df.iloc[:,2:9].std()
for i in range(2,9):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i-2])/stds[i-2]
df_1=normalized_df.iloc[:,2:9]

print("Dataset 5: Computer Hardware")

# K-NN Regression:
# Loading the data:

knn_5=knn_reg.knn_reg(df_1,range(6),6,means,stds,True)

# Tuning the parameters:

k_5,s_5=knn_5.tune()

# Fitting and testing:

mses_k5,mse_k5=knn_5.knn_fit()

print()
print("Dataset 5: Computer Hardware")
print("K-NN Regression MSE:"+str(mse_k5))
print()

# E-NN Regression:
# Loading the data:

enn_5=enn_reg.enn_reg(df_1,range(6),6,means,stds,True)

# Tuning the parameters:

ek_5,es_5,ee_5=enn_5.tune()

# Fitting and testing:

mses_ek5,mse_ek5=enn_5.fit()

print()
print("Dataset 5: Computer Hardware")
print("E-NN Regression MSE:"+str(mse_ek5))
print()

# C-NN Regression:
# Loading the data:

cnn_5=cnn_reg.cnn_reg(df_1,range(6),6,means,stds,True)

# Tuning the parameters:

ck_5,cs_5,ce_5=cnn_5.tune()

# Fitting and testing:

mses_ck5,mse_ck5=cnn_5.fit()

print()
print("Dataset 5: Computer Hardware")
print("C-NN Regression MSE:"+str(mse_ck5))
print()


# Dataset 6: Forest Fires

# Loading the dataset:

df=pd.read_csv('Datasets\Forest Fires\\forestfires.csv')

# Creating dummies for columns 'day' and 'month'

df = pd.get_dummies(df,columns=['day','month'],drop_first=True)

# Rearranging dataframe for simplicity:

cols=df.columns
cols1=cols[:10]
cols1=np.concatenate([cols1,cols[11:]])
cols1=np.append(cols1,cols[10])
df=df[cols1]

# Normalizing:

normalized_df=df.copy()
means=df.mean()
stds=df.std()
for i in range(len(df.iloc[0,:])):
    normalized_df.iloc[:,i]=(normalized_df.iloc[:,i]-means[i])/stds[i]
df_1=normalized_df

print("Dataset 6: Forest Fires")

# K-NN Regression:
# Loading the data:

knn_6=knn_reg.knn_reg(df_1,range(27),27,means,stds,True)

# Tuning the parameters:

k_6,s_6=knn_6.tune()

# Fitting and testing:

mses_k6,mse_k6=knn_6.knn_fit()

print()
print("Dataset 6: Forest Fires")
print("K-NN Regression MSE:"+str(mse_k6))
print()

## E-NN Regression:
## Loading the data:
#
#enn_6=enn_reg.enn_reg(df_1,range(27),27,means,stds,True)
#
## Tuning the parameters:
#
#ek_6,es_6,ee_6=enn_6.tune()
#
## Fitting and testing:
#
#mses_ek6,mse_ek6=enn_6.fit()
#
#print()
#print("Dataset 6: Forest Fires")
#print("E-NN Regression MSE:"+str(mse_ek6))
#print()
#
#
## C-NN Regression:
## Loading the data:
#
#cnn_6=cnn_reg.cnn_reg(df_1,range(27),27,means,stds,True)
#
## Tuning the parameters:
#
#ck_6,cs_6,ce_6=cnn_6.tune()
#
## Fitting and testing:
#
#mses_ck6,mse_ck6=cnn_6.fit()
#
#print()
#print("Dataset 6: Forest Fires")
#print("C-NN Regression MSE:"+str(mse_ck6))
#print()