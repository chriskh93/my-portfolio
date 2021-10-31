## -*- coding: utf-8 -*-
#"""
#
#@author: Christopher El Khouri
#"""
import pandas as pd
import numpy as np
import D_N
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump

def backwardElimination(X1, sl,y,X_test=pd.DataFrame()):
    numVars = len(X1.iloc[0])
    if(numVars<3):
        return X1,np.nan,X_test,False
    else:
        regressor_OLS = sm.OLS(y, X1).fit()
        maxVar = max(regressor_OLS.pvalues[:-1])
        if maxVar > sl:
            for j in range(0, numVars-1):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X1 = X1.drop([X1.iloc[:,j].name],axis=1)
                    if(X_test.empty==False):
                        X_test=X_test.drop([X_test.iloc[:,j].name],axis=1)
                    break
            return backwardElimination(X1,sl,y,X_test)
        else:
            print(regressor_OLS.summary())
            return X1,regressor_OLS,X_test,True
        
                   
           
           


# Loading the dataset and columns:

df_u=pd.read_csv('users_271120.csv')
df_u=df_u[pd.isnull(df_u['weight'])!=True].reset_index(drop=True)
df_u=df_u[pd.isnull(df_u['height'])!=True].reset_index(drop=True)
df=pd.read_csv('df_up_1_final_3.csv')
df=df.iloc[:,1:]
df_ar=pd.read_csv('df_fin_1.csv')
df_ar=df_ar.iloc[:,1:]
df['size']=df['size'].apply(lambda x: x[:2] if '/' in x else x)
df['size']=df['size'].apply(lambda x: x[:2] if 'x' in x else x)
users=df_u['user_id'].unique()
df_ar=df_ar.iloc[:,[0,1,2,3,5,6,4]]


# None-Purchases:
scats1=['XS','S','M','L','XL','XXL','XXXL','XXXXL']
df_1=df.copy()
for index,row in df.iterrows():
    if(df.iloc[index,-3]=='no'):
        if(df.iloc[index,-2]=='TOO SMALL'):
            if(df.iloc[index,-1].isnumeric()):
                df.iloc[index,-1]=str(int(df.iloc[index,-1])+1)
            elif df.iloc[index,-1] in scats1:
                df.iloc[index,-1]=scats1[scats1.index(df.iloc[index,-1])+1]
        elif(df.iloc[index,-2]=='TOO BIG'):
            if(df.iloc[index,-1].isnumeric()):
                df.iloc[index,-1]=str(int(df.iloc[index,-1])-1)
            elif df.iloc[index,-1] in scats1:
                df.iloc[index,-1]=scats1[scats1.index(df.iloc[index,-1])-1]

df=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,14]]

# Preprocess User Fits:
df['user_shirts_fit']=df['user_shirts_fit'].apply(lambda x: np.nan if x=='No idea' or x=='FALSE' else x)    
df['user_pants_fit']=df['user_pants_fit'].apply(lambda x: np.nan if x=='No idea' else x)    
df_u['shirts_fit']=df_u['shirts_fit'].apply(lambda x: np.nan if 'Fit' not in str(x) else x)            
df_u['pants_fit']=df_u['pants_fit'].apply(lambda x: np.nan if 'Fit' not in str(x) else x)            


# Prepare Arrays:

uss=[]
for i in range(len(users)):
    h=df_u[df_u['user_id']==users[i]]['height'].iloc[0]
    w=df_u[df_u['user_id']==users[i]]['weight'].iloc[0]
    sb=df_u[df_u['user_id']==users[i]]['waist_size'].iloc[0]
    ss=df_u[df_u['user_id']==users[i]]['shirt_size'].iloc[0]
    sbl=df_u[df_u['user_id']==users[i]]['blazer_size'].iloc[0]
    fb=df_u[df_u['user_id']==users[i]]['pants_fit'].iloc[0]
    ft=df_u[df_u['user_id']==users[i]]['shirts_fit'].iloc[0]
    dn1=D_N.D_N(users[i],h,w,sb,ss,sbl,fb,ft)
    uss.append(dn1.size(df,df_ar))
    uss[i]=uss[i].modeit()



uss1=[]
for i in range(len(uss)):
    uss1.append(uss[i].copy())

        
# Modling s_cat_bottom given s_num_bottom:
        
sizemod=[]
j=0
for i in range(len(uss1)):
    sm1=[]
    if(uss1[i].found_cat_bottom and uss1[i].found_num_bottom):
        sm1.append(uss1[i].s_cat_bottom)
        sm1.append(uss1[i].s_num_bottom)
        sm1.append(uss1[i].w)
        sm1.append(uss1[i].h)
        sizemod.append(sm1)

df_sm=pd.DataFrame(data=sizemod,columns=['s_cat','s','weight','height'])


df_sm['s']=df_sm['s'].astype(int)   
X=df_sm.iloc[:,[1,2,3]]
y=df_sm.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y)


clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
yp=clf.predict(X_test)
y_test=y_test.reset_index(drop=True)
count=0
for ii in range(len(yp)):
    if(yp[ii]!=y_test.iloc[ii]):
        count+=1
pc=count*100/len(yp)

clf_bc_bn = LogisticRegression(max_iter=100000).fit(X, y)
for i in range(len(uss1)):
    if(uss1[i].found_cat_bottom==False and uss1[i].s_num_bottom!=None):
        X1=[]
        X1.append(int(uss1[i].s_num_bottom))
        X1.append(uss1[i].w)
        X1.append(uss1[i].h)
        X1=np.array(X1)
        X1=X1.reshape([1,-1])
        uss1[i].s_cat_bottom=clf_bc_bn.predict(X1)[0]

dump(clf_bc_bn, 'clf_bc_bn.joblib') 

uss2=[]
for i in range(len(uss1)):
    uss2.append(uss1[i].copy())

# Modling s_num_shirt given s_cat_shirt:
sizemod=[]

for i in range(len(uss2)):
    sm1=[]
    if(uss2[i].s_cat_shirt!=None and uss2[i].found_num_shirt):
        sm1.append(uss2[i].s_cat_shirt)
        sm1.append(uss2[i].s_num_shirt)
        sm1.append(uss2[i].w)
        sm1.append(uss2[i].h)
        sizemod.append(sm1)

df_sm=pd.DataFrame(data=sizemod,columns=['s_cat','s','weight','height'])

df_sm['s']=df_sm['s'].astype(int)
df_sm['s_cat']=df_sm['s_cat'].apply(lambda x:scats1.index(x)+1).astype(int)
X=df_sm.iloc[:,[0,2,3]]
X['const']=1
y=df_sm.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

X1,reg12,X_test,xd=backwardElimination(X_train, 0.01,y_train,X_test)


yp=reg12.predict(X_test)
yp_in=np.round(yp,0).astype(int)
y_test=y_test.reset_index(drop=True)
count=0
for ii in range(len(yp_in)):
    if(yp_in.iloc[ii]!=y_test.iloc[ii]):
        count+=1
pc=count*100/len(yp)
pe=np.sum((yp-y_test)**2)/(len(yp))




X1,reg_OLS_st_ct,X_test,xd = backwardElimination(X, 0.01,y)
for i in range(len(uss2)):
    if(uss2[i].found_num_shirt==False and uss2[i].s_cat_shirt!=None):
        X1=[]
        if(" " in uss2[i].s_cat_shirt):
            uss2[i].s_cat_shirt=uss2[i].s_cat_shirt.replace(" ","")
        X1.append(scats1.index(uss2[i].s_cat_shirt)+1)
        X1.append(uss2[i].w)
        X1.append(uss2[i].h)
        X1.append(1)
        X1=np.array(X1)
        X1=X1.reshape([1,-1])
        uss2[i].s_num_shirt=np.round(reg_OLS_st_ct.predict(X1)[0],0)
dump(reg_OLS_st_ct, 'reg_OLS_st_ct.joblib') 


        
        
bandts=[]

for i in range(len(uss2)):
    cid=uss2[i].cid
    sz=uss2[i].sizes.copy()
    sz=sz.reset_index()
    sbn=uss2[i].s_num_bottom
    sbc=uss2[i].s_cat_bottom
    ssn=uss2[i].s_num_shirt
    ssc=uss2[i].s_cat_shirt
    if not(sbn==None and ssc==None):
        if(sbn!=None):
            sbn=int(sbn)
        
        sz1=sz[sz['Found']==True].reset_index(drop=True)
        if(len(sz1)>0):
            for index,row in sz1.iterrows():
                ud=[]
                ud.append(cid)
                ud.append(sz1.iloc[index,0])
                ud.append(sz1.iloc[index,1])
                ud.append(sz1.iloc[index,2])
                ud.append(sz1.iloc[index,3])
                ud.append(sz1.iloc[index,4])
                if(type(sz1.iloc[index,5])==np.float64):
                    ud.append(str(sz1.iloc[index,5]))
                else:
                    ud.append(sz1.iloc[index,5])
                ud.extend([sbn,sbc,ssn,ssc])
                bandts.append(ud)
        else:
            ud=[]
            ud.append(cid)
            ud.extend(['','','','','',''])
            ud.extend([sbn,sbc,ssn,ssc])
            bandts.append(ud)

        


df_btm=pd.DataFrame(bandts)

df_btm.to_csv('df_btm_3.csv',index=False)
df_ar_1=uss2[0].sizes.copy()
df_ar_1['Size']=np.nan
df_ar_1['Found']=np.nan
df_ar_1.to_csv('df_ar_3.csv',index=False)

    
