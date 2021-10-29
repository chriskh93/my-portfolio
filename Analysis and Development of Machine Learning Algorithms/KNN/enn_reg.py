# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:40:05 2020

@author: CHRIS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 00:29:52 2020

@author: CHRIS
"""

# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import math

class enn_reg:
    
    
    def __init__(self,df, xinds, yinds,means=0,stds=0,norm=False):
        self.df=df
        self.xinds=xinds
        self.yinds=yinds
        self.k=0
        self.err=0
        self.means=means
        self.stds=stds
        self.norm=norm
        
    def eucd(self,x1,x2):
        if(type(x1)==np.int64):
            y=np.sqrt((x1-x2)**2)
        else:
                
            if(len(x1)!=len(x2)):
                return False
            else:
                y=0
                for i in range(len(x1)):
                    y=y+np.sqrt((x1[i]-x2[i])**2)
        return y
    
    def kmeans(self,df_1,km):
        cents=[]
        df_11=df_1.sample(frac=1)
        for i in range(km):
            cents.append(df_11.iloc[i,:])  
        df_2=df_1.copy()
        df_2['cluster']=0
        err=100
        count=0
        while((err>10**-3) and count!=100):
            for index,row in df_2.iterrows():
                mind=0
                for i in range(len(cents)):
                    d=self.eucd(cents[i].iloc[self.xinds],df_2.iloc[index,self.xinds])
                    if(i==0):
                        mind=d
                        df_2.iloc[index,len(df_2.iloc[0,:])-1]=i
                    else:
                        if(d<mind):
                            mind=d
                            df_2.iloc[index,len(df_2.iloc[0,:])-1]=i
            means=[]
            for i in range(len(cents)):
                mns=df_2[df_2['cluster']==i].iloc[:,:-1].mean()
                means.append(mns)
            errs=[]
            for i in range(len(means)):
                e=np.abs(means[i].iloc[self.xinds]-cents[i].iloc[self.xinds])/(cents[i].iloc[self.xinds]+0.001)
                errs.append(e)
            err=np.nanmax(errs)
            cents=means
            count+=1
        
        return cents,df_2
    
    def wss(self,df_2,cents):
        sse=0
        for i in range(len(cents)):
            df_3=df_2[df_2['cluster']==i].reset_index(drop=True)
            for index,row in df_3.iterrows():
                d=self.eucd(df_3.iloc[index,:-1],cents[i])
                d=d**2
                sse+=d

        return sse          
     
        
    def rbf(self,df_2,cents,sigma):
        df_3=df_2.copy().reset_index(drop=True)
        y_pred=[]

        for index,row in df_3.iterrows():
            num=0
            den=0
            for i in range(len(cents)):
                kern=np.exp(-((self.eucd(cents[i].iloc[self.xinds],df_3.iloc[index,self.xinds]))**2)/(2*(sigma**2)))
                kern=kern
                num=num+kern*cents[i].iloc[self.yinds]
                den=den+kern
            g=num/den
            y_pred.append(g)
        
        return df_3,y_pred

        
        
            
        
                    
                    
    def tune(self):
        df=self.df.sample(frac=1).reset_index(drop=True)
        yind=self.yinds
        xinds=self.xinds
        tenp=np.floor(len(df)/10).astype(int)
        df_tenp=df.iloc[:tenp,:].reset_index(drop=True)
        wsss=[]
        bestkm=0
        unique_ys=df.iloc[:,yind].unique()
        bestcents=[]
        for km in np.arange(1,len(unique_ys),1):
            cents,df_2=self.kmeans(df_tenp,km)
            wsss.append(self.wss(df_2,cents))
            bestcents.append(cents)
            if(len(wsss)>1):
                if(((np.abs(wsss[len(wsss)-1]-wsss[len(wsss)-2])/wsss[len(wsss)-2])<=0.15)):
                    bestkm=km-1
                    bestcents=bestcents[bestkm-1]
                    break
        sigma=1
        bestmse=0
        bestsigma=0
        for j in np.arange(1,11,1):
            sigma=j
            df_3,y_pred=self.rbf(df_tenp,bestcents,sigma)
            mse=np.sum((np.array(df_3.iloc[:,self.yinds])-y_pred)**2)/len(df_3)
            if(j==1):
                bestmse=mse
                bestsigma=j
            else:
                if(mse<bestmse):
                    bestmse=mse
                    bestsigma=j
        sigma=bestsigma
        
        err_th=0
        best_err_th=0

        bestmsef=0
        for err_th in np.arange(1,15,1):
            df_tenp1=df_tenp.copy()
            for index,row in df_tenp.iterrows():
                er=np.abs(y_pred[index]-df_tenp.iloc[index,self.yinds])/np.abs(df_tenp.iloc[index,self.yinds])
                if(er<0):
                    er=er*-1
                if(er*100>err_th):
                    df_tenp1.drop(index=index,inplace=True)
            df_tenp1=df_tenp1.reset_index(drop=True)
            if(len(df_tenp1)>=bestkm):
                cents,df_2=self.kmeans(df_tenp1,bestkm)
                bestmse=0
                bestsigma=0
                for j in np.arange(1,11,1):
                    sigma=j
                    df_4,y_pred1=self.rbf(df_tenp,cents,sigma)
                    mse=np.sum((np.array(df_4.iloc[:,self.yinds])-y_pred1)**2)/len(df_4)
                    if(j==1):
                        bestmse=mse
                        bestsigma=j
                    else:
                        if(mse<bestmse):
                            bestmse=mse
                            bestsigma=j

                msef=bestmse
                if(bestmsef==10**6):
                    bestmsef=msef
                    best_err_th=err_th
                    bsigma=bestsigma
                else:
                    if(msef<bestmsef):
                        bestmsef=msef
                        best_err_th=err_th
                        bsigma=bestsigma
            else:
                bestmsef=10**6
                best_err_th=100
                bsigma=1
            
                    
                    
                    
        
        self.bestk=bestkm
        self.bestsigma=bsigma
        self.besterr=best_err_th
        self.df=df.iloc[tenp:,:]
        return bestkm,sigma,best_err_th
                
    def fit(self):
        k=self.bestk
        df=self.df.reset_index(drop=True)
        sigma=self.bestsigma
        err_th=self.besterr
        start=0
        tl=len(df)
        mses=[]
        for ii in range(5):
            end=np.ceil((tl*(ii+1)/5))
            end=end.astype(int)
            df_test=df.iloc[start:end,:].reset_index(drop=True)
            if(start==0):
                df_train=df.iloc[end:tl].reset_index(drop=True)
            else:
                df_train=pd.concat([df.iloc[0:start,:],df.iloc[end:tl,:]]).reset_index(drop=True)
            df_test=df_test.sample(frac=1).reset_index(drop=True)
            df_train=df_train.sample(frac=1).reset_index(drop=True)
            cents,df_train=self.kmeans(df_train,k)
            df_train1,y_pred1=self.rbf(df_train,cents,sigma)
            for index,row in df_train1.iterrows():
                er=np.abs((y_pred1[index]-df_train1.iloc[index,self.yinds])/df_train1.iloc[index,self.yinds])
                if(er<0):
                    er=er*-1
                if(er*100>err_th):
                    df_train.drop(index=index,inplace=True)
            df_train=df_train.reset_index(drop=True)
            cents,df_train=self.kmeans(df_train,k)
            df_test,y_pred=self.rbf(df_test,cents,sigma)
            y=df_test.iloc[:,self.yinds]
            mse=np.sum((np.array(df_test.iloc[:,self.yinds])-y_pred)**2)/len(df_test)
            mses.append(mse)
            start=end
            y_pred=np.array(y_pred)
            print("E-NN Regression Fold "+str(ii+1)+":")
            print("k="+str(k))
            print("MSE: "+str(np.round(mse,2)))
            if(self.norm):
                
                y=y*self.stds.iloc[self.yinds]+self.means.iloc[self.yinds]
                y_pred=y_pred*self.stds.iloc[self.yinds]+self.means.iloc[self.yinds]
            df_res=pd.DataFrame({'Actual':y,'Predicted':y_pred})
            print(df_res.to_string())
            print()
        return mses,np.mean(mses)
        
        