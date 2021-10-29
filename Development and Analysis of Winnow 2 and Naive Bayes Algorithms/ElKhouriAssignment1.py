# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:39:09 2020

@author: Christopher El Khouri
        605.649.81
    
"""
# Importing the necessary libraries:

import pandas as pd
import numpy as np
import winnow2
import nbayes
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


# Initializing the parameters for the summary table:
#   nba: The array that contains the accuracies of the Naive Bayes algorithm
#   w2a: The array that contains the accuracies of the Winnow-2 algorithm
#   acols: The array that contains the column names of the summary table
    
nba=[]
w2a=[]
acols=[]

nba.append('Naive Bays')
w2a.append('Winnow-2')
acols.append('Algorithm')

# Dataset 1: Breast Cancer

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Breast Cancer\\breast-cancer-wisconsin.data',header=None)
cols=['sample_code_number','clump_thickness','uniformity_cell_size',
      'uniformity_cell_shape','marginal_adhesion','single_epithelial_cell_size',
      'bare_nuclei','brand_chromatin','normal_nucleoli','mitosis','class']
df.columns=cols

acols.append('D1: Breast Cancer')
# Handling the missing values:

df_1=df[df['bare_nuclei']!='?']

df_1_2=df_1[df_1['class']==2]
df_1_4=df_1[df_1['class']==4]
d_2=df_1_2['bare_nuclei'].describe()
d_4=df_1_4['bare_nuclei'].describe()

df_1_2['bare_nuclei']=df_1_2['bare_nuclei'].astype(int)


df_1_4['bare_nuclei']=df_1_4['bare_nuclei'].astype(int)


df_2=df[df['bare_nuclei']=='?']
len(df_2[df_2['class']==2])

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
d=df.describe()
medians=np.median(df,axis=0)

# Discretizing the features:

for i in range(1,len(cols)-1):
    df[cols[i]+" >"+str(medians[i])]=df[cols[i]].apply(lambda x:0 if x<=medians[i] else 1 )

# Shuffling the dataframe for use in Winnow-2:
    
df_3=df.sample(frac=1)

# Setting the x and y values for use in Winnow-2:

x=df_3.iloc[:,11:20]
y=df_3.iloc[:,10]
x=np.array(x)
y=np.array(y)

# Winnow-2 algorithm:
# Loading our data into the Winnow-2 algorithm:

w1=winnow2.winnow2(x,y,alpha=2,theta=1)

# Tuning using 10% of the data:

tenp=np.floor(len(x)/10).astype(int)-1

abest,tbest=w1.tune(x[:tenp],y[:tenp])

# Fitting and testing the data:
w1=winnow2.winnow2(x[tenp:],y[tenp:],alpha=abest,theta=tbest)
ws1=w1.fit()
ypredw1,accw1,yw1=w1.test()
df_d1_w2c=pd.DataFrame(data=[ypredw1,yw1]).transpose()
df_d1_w2c.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 1: Breast Cancer')
print('Accuracy: '+str(np.round(accw1,2))+'%')
print('Results of 1 Fold: ')
print(df_d1_w2c)
print('View variable df_d1_w2c for full result')
print('')
w2a.append(accw1)

# Naive Bayes:

# Loading the data into the Naive Bayes algorithm:

nb_breast=nbayes.nbayes(df,range(11,20),10)

# Tuning:

m,p1=nb_breast.tune()

# Fitting and testing:

npsx,npsy=nb_breast.fit()
yp,ys,accs=nb_breast.test()
df_d1_nbc=pd.DataFrame(data=[yp,ys]).transpose()
df_d1_nbc.columns=['Predicted','Actual']

print('Naive Bayes Algorithm for Dataset 1: Breast Cancer')
print('Accuracy: '+str(np.round(accs,2))+'%')
print('Results of 1 Fold: ')
print(df_d1_nbc)
print('View variable df_d1_nbc for full result')
print('')
nba.append(accs)





# Dataset 2: Glass

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Glass\\glass.data',header=None)
df.head()
cols=['id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
df.columns=cols
typesg=df['Type'].unique()
d=df.describe()

# Discretizing the features:

means=np.mean(df,axis=0)
means=np.round(means,2)
for i in range(1,len(cols)-1):
    df[cols[i]+" >"+str(means[i])]=df[cols[i]].apply(lambda x:0 if x<=means[i] else 1 )

df = pd.concat([df, pd.get_dummies(df["Type"], prefix="Type")], axis=1)

# Shuffling the dataframe for use in Winnow-2:

df_3=df.sample(frac=1)
tenp=np.floor(len(df_3)/10).astype(int)-1

# Winnow-2 algorithm:

# Glass Type 1
# Tuning using 10% of the data:
acols.append('D2: Glass: Type 1')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,20]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)

# Fitting and testing the data:
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,20]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()
df_d2_w2c_1=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_1.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 1')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_1)
print('View variable df_d2_w2c_1 for full result')
print('')
w2a.append(b)
nba.append(0)

# Glass Type 2

acols.append('D2: Glass: Type 2')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,21]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,21]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d2_w2c_2=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_2.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 2')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_2)
print('View variable df_d2_w2c_2 for full result')
print('')
w2a.append(b)
nba.append(0)

# Glass Type 3

acols.append('D2: Glass: Type 3')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,22]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,22]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d2_w2c_3=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_3.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 3')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_3)
print('View variable df_d2_w2c_3 for full result')
print('')
w2a.append(b)
nba.append(0)

# Glass Type 5

acols.append('D2: Glass: Type 5')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,23]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,23]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d2_w2c_5=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_5.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 5')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_5)
print('View variable df_d2_w2c_5 for full result')
print('')
w2a.append(b)
nba.append(0)

# Glass Type 6

acols.append('D2: Glass: Type 6')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,24]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,24]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d2_w2c_6=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_6.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 6')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_6)
print('View variable df_d2_w2c_6 for full result')
print('')
w2a.append(b)
nba.append(0)

# Glass Type 7

acols.append('D2: Glass: Type 7')
x=df_3.iloc[:tenp,11:20]
y=df_3.iloc[:tenp,25]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,11:20]
y=df_3.iloc[tenp:,25]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d2_w2c_7=pd.DataFrame(data=[a,c]).transpose()
df_d2_w2c_7.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 2: Glass: Type 7')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_w2c_7)
print('View variable df_d2_w2c_7 for full result')
print('')
w2a.append(b)
nba.append(0)

# Combining Winnow-2 accuracies for the Glass Dataset:

w2_glassaccs=w2a[2]*w2a[3]*w2a[4]*w2a[5]*w2a[6]*w2a[7]
w2_glassaccs=w2_glassaccs*100/((10**2)**6)
w2a.append(w2_glassaccs)
acols.append('D2: Glass')


# Naive Bayes:

# Loading the data into the Naive Bayes algorithm:

nb_glass=nbayes.nbayes(df,range(11,20),10)

# Tuning:

m,p1=nb_glass.tune()

# Fitting and testing:

npsx,npsy=nb_glass.fit()
yp,ys,accs=nb_glass.test()
df_d2_nbc=pd.DataFrame(data=[yp,ys]).transpose()
df_d2_nbc.columns=['Predicted','Actual']

print('Naive Bayes Algorithm for Dataset 2: Glass')
print('Accuracy: '+str(np.round(accs,2))+'%')
print('Results of 1 Fold: ')
print(df_d2_nbc)
print('View variable df_d2_nbc for full result')
print('')
nba.append(accs)




# Dataset 3: Iris

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Iris\\iris.data',header=None)
cols=['sepal_length','sepal_width','petal_length',
      'petal_width','class']
df.columns=cols
classi=df['class'].unique()
d=df.describe()

# Discretizing the features and classes:

means=np.mean(df,axis=0)
means=np.round(means,2)
for i in range(0,len(cols)-1):
    df[cols[i]+" >"+str(means[i])]=df[cols[i]].apply(lambda x:0 if x<=means[i] else 1 )

df = pd.concat([df, pd.get_dummies(df["class"], prefix="class")], axis=1)

# Shuffling the dataframe for use in Winnow-2:

df_3=df.sample(frac=1)
tenp=np.floor(len(df_3)/10).astype(int)-1

# Winnow-2 algorithm:

# Iris Class 1

x=df_3.iloc[:tenp,5:9]
y=df_3.iloc[:tenp,9]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,5:9]
y=df_3.iloc[tenp:,9]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d3_w2c_1=pd.DataFrame(data=[a,c]).transpose()
df_d3_w2c_1.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 3: Iris: Class 1')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d3_w2c_1)
print('View variable df_d3_w2c_1 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D3: Iris: Class 1')

# Iris Class 2

x=df_3.iloc[:tenp,5:9]
y=df_3.iloc[:tenp,10]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,5:9]
y=df_3.iloc[tenp:,10]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d3_w2c_2=pd.DataFrame(data=[a,c]).transpose()
df_d3_w2c_2.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 3: Iris: Class 2')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d3_w2c_2)
print('View variable df_d3_w2c_2 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D3: Iris: Class 2')


# Iris Class 3

x=df_3.iloc[:tenp,5:9]
y=df_3.iloc[:tenp,11]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,5:9]
y=df_3.iloc[tenp:,11]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d3_w2c_3=pd.DataFrame(data=[a,c]).transpose()
df_d3_w2c_3.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 3: Iris: Class 3')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d3_w2c_3)
print('View variable df_d3_w2c_3 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D3: Iris: Class 3')

# Combining Winnow-2 accuracies for the Iris Dataset:

w2_irisaccs=w2a[9]*w2a[10]*w2a[11]
w2_irisaccs=w2_irisaccs*100/((10**2)**3)
w2a.append(w2_irisaccs)
acols.append('D3: Iris')


# Naive Bayes:

# Loading the data into the Naive Bayes algorithm:

nb_iris=nbayes.nbayes(df,range(5,9),4)

# Tuning:

m,p1=nb_iris.tune()

# Fitting and testing:

npsx,npsy=nb_iris.fit()
yp,ys,accs=nb_iris.test()
df_d3_nbc=pd.DataFrame(data=[yp,ys]).transpose()
df_d3_nbc.columns=['Predicted','Actual']

print('Naive Bayes Algorithm for Dataset 3: Iris')
print('Accuracy: '+str(np.round(accs,2))+'%')
print('Results of 1 Fold: ')
print(df_d3_nbc)
print('View variable df_d3_nbc for full result')
print('')
nba.append(accs)





# Dataset 4: Soybean (small)

# Loading the dataset and columns:

df=pd.read_csv('Datasets\Soybean\\soybean-small.data',header=None)
class_s=df.loc[:,35].unique()
d=df.describe()

# Discretizing the classes:

df = pd.concat([df, pd.get_dummies(df.loc[:,35])], axis=1)

# Shuffling the dataframe for use in Winnow-2:

df_3=df.sample(frac=1)
tenp=np.floor(len(df_3)/10).astype(int)-1

# Winnow-2 algorithm:
# Soybean Type D1

x=df_3.iloc[:tenp,:35]
y=df_3.iloc[:tenp,36]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,:35]
y=df_3.iloc[tenp:,36]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d4_w2c_1=pd.DataFrame(data=[a,c]).transpose()
df_d4_w2c_1.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 4: Soybean: Type D1')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d4_w2c_1)
print('View variable df_d4_w2c_1 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D4: Soybean: Type D1')

# Soybean Type D2


x=df_3.iloc[:tenp,:35]
y=df_3.iloc[:tenp,37]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,:35]
y=df_3.iloc[tenp:,37]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d4_w2c_2=pd.DataFrame(data=[a,c]).transpose()
df_d4_w2c_2.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 4: Soybean: Type D2')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d4_w2c_2)
print('View variable df_d4_w2c_2 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D4: Soybean: Type D2')


# Soybean Type D3

x=df_3.iloc[:tenp,:35]
y=df_3.iloc[:tenp,38]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,:35]
y=df_3.iloc[tenp:,38]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d4_w2c_3=pd.DataFrame(data=[a,c]).transpose()
df_d4_w2c_3.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 4: Soybean: Type D3')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d4_w2c_3)
print('View variable df_d4_w2c_3 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D4: Soybean: Type D3')


# Soybean Type D4

x=df_3.iloc[:tenp,:35]
y=df_3.iloc[:tenp,39]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,:35]
y=df_3.iloc[tenp:,39]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d4_w2c_4=pd.DataFrame(data=[a,c]).transpose()
df_d4_w2c_4.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 4: Soybean: Type D4')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d4_w2c_4)
print('View variable df_d4_w2c_4 for full result')
print('')
w2a.append(b)
nba.append(0)
acols.append('D4: Soybean: Type D4')


# Combining Winnow-2 accuracies for the Soybean Dataset:

w2_soyaccs=w2a[13]*w2a[14]*w2a[15]*w2a[16]
w2_soyaccs=w2_soyaccs*100/((10**2)**4)
w2a.append(w2_soyaccs)
acols.append('D4: Soybean')


# Naive Bayes:

nb_soybean=nbayes.nbayes(df,range(0,35),35)
m,p1=nb_soybean.tune()
npsx,npsy=nb_soybean.fit()
yp,ys,accs=nb_soybean.test()

df_d4_nbc=pd.DataFrame(data=[yp,ys]).transpose()
df_d4_nbc.columns=['Predicted','Actual']

print('Naive Bayes Algorithm for Dataset 4: Soybean')
print('Accuracy: '+str(np.round(accs,2))+'%')
print('Results of 1 Fold: ')
print(df_d4_nbc)
print('View variable df_d4_nbc for full result')
print('')
nba.append(accs)





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
classv=df['class_name'].unique()

# Discretizing the features:

for index,row in df.iterrows():
    for i in range(len(row)):
        if(df.iloc[index,i]=='n'):
            df.iloc[index,i]=0
        elif(df.iloc[index,i]=='y'):
            df.iloc[index,i]=1
            
# Handling missing values:
            
df_dem=df[df['class_name']==classv[1]].reset_index(drop=True)
df_rep=df[df['class_name']==classv[0]].reset_index(drop=True)
mode_dem=df_dem.mode()
mode_rep=df_rep.mode()
df_2=df
for index,row in df_2.iterrows():
    for i in range(len(row)):
        if(df_2.iloc[index,i]=='?'):
            if(df_2.iloc[index,0]=='republican'):
                df_2.iloc[index,i]=mode_rep.iloc[0,i]
            else:
                df_2.iloc[index,i]=mode_dem.iloc[0,i]

# Disctretizing the classes:
                
df_2 = pd.concat([df_2, pd.get_dummies(df_2["class_name"])], axis=1)

# Shuffling the dataframe for use in Winnow-2:

df_3=df_2.sample(frac=1)
tenp=np.floor(len(df_3)/10).astype(int)-1


# Winnow-2 algorithm:

x=df_3.iloc[:tenp,1:17]
y=df_3.iloc[:tenp,17]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y)
abest,tbest=w2.tune(x,y)
x=df_3.iloc[tenp:,1:17]
y=df_3.iloc[tenp:,17]
x=np.array(x)
y=np.array(y)
w2=winnow2.winnow2(x,y,abest,tbest)
ws=w2.fit()
a,b,c=w2.test()

df_d5_w2c=pd.DataFrame(data=[a,c]).transpose()
df_d5_w2c.columns=['Predicted','Actual']

print('Winnow-2 Algorithm for Dataset 5: Vote')
print('Accuracy: '+str(np.round(b,2))+'%')
print('Results of 1 Fold: ')
print(df_d5_w2c)
print('View variable df_d5_w2c for full result')
print('')
w2a.append(b)
acols.append('D5: Vote')



# Naive Bayes:

nb_votes=nbayes.nbayes(df_2,range(1,17),17)
m,p1=nb_votes.tune()
npsx,npsy=nb_votes.fit()
yp,ys,accs=nb_votes.test()


df_d5_nbc=pd.DataFrame(data=[yp,ys]).transpose()
df_d5_nbc.columns=['Predicted','Actual']

print('Naive Bayes Algorithm for Dataset 5: Vote')
print('Accuracy: '+str(np.round(accs,2))+'%')
print('Results of 1 Fold: ')
print(df_d5_nbc)
print('View variable df_d5_nbc for full result')
print('')
nba.append(accs)



# Producing Summary Table:
df_summary=pd.DataFrame(data=[w2a,nba],columns=acols)
print(df_summary)