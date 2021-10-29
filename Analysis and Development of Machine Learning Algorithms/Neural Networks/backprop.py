# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:24:59 2020

@author: Christopher El Khouri
        605.649.81
"""


# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import random
import math

# class backprop implements the backpropagation feed forward neural net model:

class backprop:
    
    # Constructor takes the follwing parameters:
    #   df: The pandas dataframe that requires modeling
    #   xinds: The indices of the x-variables within df
    #   yinds: The index of the y-variable within df (The target)
    #   reg: Boolean variable for whether or not this is a regression model
    #   norm: Boolean variable for whether or not the target variable is normalized
    #   means: The mean of the target variable
    #   stds: The standard deviation of the target variable
    
    # The constructor further initializes the following variables:
    #   y_cats: The different possible categorical values of 
    #       our y-variable (target)
    
    # Furthemore, constructor appends ones to dataframe for bias and modifies
    #   xinds accordingly
    
    def __init__(self,df, xinds, yind,reg=False,norm=False,means=0,stds=0):
        
        self.yind=yind
        if(reg==False):
            self.ycats=df.iloc[:,self.yind].unique()
        else:
            self.means=means
            self.stds=stds
            self.norm=norm
        self.reg=reg
        self.df=pd.concat([df,pd.DataFrame(np.ones(len(df)))],axis=1)
        xinds.append(len(self.df.iloc[0])-1)
        self.xinds=xinds
        

    # Method fivefoldcv divides a dataframe df into 5 folds for cross validation 
    # The method take the following input parameters:
    #   df: The dataset
    # The method returns:
    #   folddfs: A 5x2 array containing with each row containing a column for
    #       the training dataset and a column for the test dataset
    
    def fivefoldcv(self,df):
        
        folds=[]
        tl=len(df)
        start=0
        for i in range(5):
            end=np.round((tl*(i+1)/5),0).astype(int)
            df_1=df.iloc[start:end,:].copy()
            folds.append(df_1)
            start=end
        folddfs=[]
        for i in range(5):
            folddf=[]
            df_2=pd.DataFrame(columns=df.columns)
            for j in range(5):
                if(j==i):
                    df_3=folds[j]
                else:
                    
                    df_2=pd.concat([df_2,folds[j]])
            folddf.append(df_2.reset_index(drop=True))
            folddf.append(df_3.reset_index(drop=True))
            folddfs.append(folddf)
        return folddfs
    
    
    # Method train performs gradient descent with backprop to generate the 
    #   coefficients to be used in the feedforward neural network model 
    #
    # The method take the following input parameters:
    #   n: The learning rate 
    #   df: The dataset
    #   nhl: The number of hidden layers
    #   nhd: The number of hidden dimensions (array data type incase nhl=2)
    #       nhd[0]: The number of hidden dimensions of the first layer
    #       nhd[1]: The number of hudden dimensions of the second layer
    #   maxc: The maximum number of epochs
    #
    # The method returns:
    #   ws: The converged coefficients of the input layer
    #   err: The error difference between the previous ws and the current ws
    #   count: The number of epochs
    #   v1: The converged coefficients of the 1st hidden layer
    #   v2: The converged coefficients of the 2nd hidden layer

    def train(self,n,nhl,nhd,df=pd.DataFrame(),maxc=1000):
        xinds=self.xinds
        yind=self.yind
        reg=self.reg
        if(df.empty):
            df=self.df
        
        # Initializing our coefficients to values between -0.01 and 0.01
        # If nhl>0 then consider an extra node for the v1 and v2 arrays for the 
        #   bias coefficients
        # If nhl=2 then split dataframe into 'ndfs' dataframe of 100 rows each
        
        if(reg==False):
            ycats=self.ycats
            if(nhl==0):
                v1=[]
                v2=[]
                ws=np.zeros([len(ycats),len(xinds)])
            elif(nhl==1):
                ws=np.zeros([nhd,len(xinds)])
                v1=np.zeros([len(ycats),nhd+1])
                v2=[]
                for i in range(len(v1)):
                    for j in range(len(v1[0])):
                        v1[i,j]=random.uniform(-0.01,0.01)
            elif(nhl==2):
                ws=np.zeros([nhd[0],len(xinds)])
                v1=np.zeros([nhd[1],nhd[0]+1])
                v2=np.zeros([len(ycats),nhd[1]+1])
                for i in range(len(v1)):
                    for j in range(len(v1[0])):
                        v1[i,j]=random.uniform(-0.01,0.01)
                for i in range(len(v2)):
                    for j in range(len(v2[0])):
                        v2[i,j]=random.uniform(-0.01,0.01)
                dfs=[]
                ndfs=np.round(len(df)/100,0)
                if(ndfs==0):
                    ndfs+=1
                start=0
                end=np.round(len(df)/ndfs,0).astype(int)
                ndfs=ndfs.astype(int)
                for i in range(ndfs):
                    d=df.iloc[start:end,:].copy().reset_index(drop=True)
                    dfs.append(d)
                    start=end
                    end+=np.round(len(df)/ndfs,0).astype(int)
                
            for i in range(len(ws)):
                for j in range(len(ws[0])):
                    ws[i,j]=random.uniform(-0.01,0.01)
        else:
            if(nhl==0):
                v1=[]
                v2=[]
                ws=np.zeros(len(xinds))
            elif(nhl==1):
                ws=np.zeros([nhd,len(xinds)])
                v1=np.zeros(nhd+1)
                v2=[]
                for i in range(len(v1)):
                    v1[i]=random.uniform(-0.01,0.01)
            elif(nhl==2):
                ws=np.zeros([nhd[0],len(xinds)])
                v1=np.zeros([nhd[1],nhd[0]+1])
                v2=np.zeros(nhd[1]+1)
                for i in range(len(v1)):
                    for j in range(len(v1[0])):
                        v1[i,j]=random.uniform(-0.01,0.01)
                for i in range(len(v2)):
                    v2[i]=random.uniform(-0.01,0.01)
                dfs=[]
                ndfs=np.round(len(df)/100,0)
                if(ndfs==0):
                    ndfs+=1
                start=0
                end=np.round(len(df)/ndfs,0).astype(int)
                ndfs=ndfs.astype(int)
                for i in range(ndfs):
                    d=df.iloc[start:end,:].copy().reset_index(drop=True)
                    dfs.append(d)
                    start=end
                    end+=np.round(len(df)/ndfs,0).astype(int)
            for i in range(len(ws)):
                if(nhl==0):
                    ws[i]=random.uniform(-0.01,0.01)
                else:
                    for j in range(len(ws[0])):
                        ws[i,j]=random.uniform(-0.01,0.01)        
        err=10**6
        count=0
        minc=0
        # dws initialized to ws for array size purposes:
        dw=ws.copy()
        
        # 10**-1 is considered as the convergance threshold
        while((((err>10**-1) or count<minc) and count<maxc)):
            
            # oldws is set as the ws from the last iteration:
            oldws=ws.copy()
            
            # 0 Hidden Layers:
            
            if(nhl==0):
                
                # Classification:
                
                if(reg==False):
                    y=np.zeros([len(df),len(ycats)])
                    
                    # Calculating y using softmax:
                    
                    for i in range(len(ycats)):
                        y[:,i]=self.softmax(df,ws,i,xinds)
                    
                    # Calculating dw:
                    
                    for i in range(len(ycats)):
                        rs=df.iloc[:,yind].apply(lambda x: 1 if x==ycats[i] else 0)
                        summer=rs-y[:,i]
                        prod1=np.zeros([len(df),len(ws[0])])
                        for j in range(len(ws)):
                            prod1[:,j]=df.iloc[:,xinds[j]]*summer
                        dw[i]=np.mean(n*prod1,axis=0)
                    
                    # Updating the ws values:
                    
                    for i in range(len(ws)):
                        ws[i]=ws[i]+dw[i]
                
                # Regression:
                
                else:
                    y=np.zeros(len(df))
                    
                    # Calculating y linearly:
                    
                    for j in range(len(ws)):
                        y=y+df.iloc[:,xinds[j]]*ws[j]
                    
                    # Calculating dw:
                    rs=df.iloc[:,yind]
                    summer=rs-y
                    prod1=np.zeros([len(df),len(ws)])
                    for j in range(len(ws)):
                        prod1[:,j]=df.iloc[:,xinds[j]]*summer
                    dw=np.mean(n*prod1,axis=0)
                    
                    # Updating the ws values:
                    
                    ws=ws+dw
            
            # 1 Hidden Layer:
            
            elif(nhl==1):
                
                # oldv1 is set as the v1 from the last iteration:
                
                oldv1=v1.copy()
                
                # z coefficients initialized to np.ones with nhd+1 columns to
                #   account for the bias:
                
                z=np.ones([len(df),nhd+1])
                
                # Calculating the z coefficients using the logsitic sigmoid:
                
                for h in range(nhd):
                    z[:,h+1]=self.sigmoid_log(df,ws,h,xinds)
                zdf=pd.DataFrame(data=z)
                
                # Classification:
                
                if(reg==False):
                    dv1=np.zeros([len(v1),len(v1[0])])
                    y=np.zeros([len(df),len(ycats)])
                    
                    # Calculating the y values using softmax:
                    for i in range(len(ycats)):
                        y[:,i]=self.softmax(zdf,v1,i,np.arange(len(zdf.iloc[0])))
                    
                    # Calculating dv1:
                    
                    for i in range(len(ycats)):
                        prod=np.zeros([len(df),nhd+1])
                        rs=df.iloc[:,yind].apply(lambda x: 1 if x==ycats[i] else 0)
                        for j in range(len(z[0])):
                            prod[:,j]=(rs-y[:,i])*z[:,j]
                        dv1[i]=np.mean(n*prod,axis=0)
                    
                    # Calculating dw:
                    
                    for h in range(nhd):
                        summer=np.zeros(len(df))
                        for i in range(len(ycats)):
                            rs=df.iloc[:,yind].apply(lambda x: 1 if x==ycats[i] else 0)
                            summer=summer+(rs-y[:,i])*v1[i][h]
                        prod0=z[:,h+1]*(1-z[:,h+1])
                        prod1=np.zeros(len(df))
                        prod1=df.iloc[:,xinds].apply(lambda x: x*prod0)
                        prod2=np.zeros([len(df),len(ws[0])])
                        for j in range(len(ws[0])):
                            prod2[:,j]=prod1.iloc[:,j]*summer
                        dw[h]=np.mean(n*prod2,axis=0)
                    
                    # Updating v1 values:
                    
                    for i in range(len(ycats)):
                        v1[i]=v1[i]+dv1[i]
                        
                # Regression:
                
                else:
                    # Calculating the z coefficients using the logsitic sigmoid:
                    
                    dv=np.zeros([len(v1)])
                    y=np.zeros([len(df)])
                    
                    # Calculating the y values linearly
                    for j in range(len(z[0])):
                        y=y+v1[j]*z[:,j]
                    
                    # Calculating dv:
                    
                    prod=np.zeros([len(df),nhd+1])
                    rs=df.iloc[:,yind]
                    for j in range(len(z[0])):
                        prod[:,j]=(rs-y)*z[:,j]
                    dv=np.mean(n*prod,axis=0)
                    
                    # Calculating dw:
                    for h in range(nhd):
                        summer=np.zeros(len(df))
                        rs=df.iloc[:,yind]
                        summer=summer+(rs-y)*v1[h]
                        prod0=z[:,h+1]*(1-z[:,h+1])
                        prod1=np.zeros(len(df))
                        prod1=df.iloc[:,xinds].apply(lambda x: x*prod0)
                        prod2=np.zeros([len(df),len(ws[0])])
                        for j in range(len(ws[0])):
                            prod2[:,j]=prod1.iloc[:,j]*summer
                        dw[h]=np.mean(n*prod2,axis=0)
                    
                    # Updating v1:
                    
                    v1=v1+dv
                
                # Updating ws:
                
                for h in range(nhd):
                    ws[h]=ws[h]+dw[h]
            
            # 2 Hidden Layers:
            
            else:
                oldv1=v1.copy()
                oldv2=v2.copy()
                z1=np.ones([nhd[0]+1])
                z2=np.ones([nhd[1]+1])

                # Implementation of bath gradient descent utilizing splitted 
                #   dataframes:
                
                for dfc in range(ndfs):
                    dv1=np.zeros([len(v1),len(v1[0])])
                    dw=np.zeros([len(ws),len(ws[0])])
                    if(reg==False):
                        dv2=np.zeros([len(v2),len(v2[0])])
                    else:
                        dv2=np.zeros(len(v2))
                    
                    # Iterating through each splitted dataframe:
                    
                    for index,row in dfs[dfc].iterrows():
                        
                        # Calculating the z coefficients using logistic sigmoid:
                        
                        for h in range(nhd[0]):
                            z1[h+1]=self.sigmoid_log1(row,ws,h,xinds)
                        z1d=pd.DataFrame(data=z1)
                        for h2 in range(nhd[1]):
                            z2[h2+1]=self.sigmoid_log1(z1d,v1,h2,np.arange(len(z1)))
                        
                        # Classification:
                        
                        if(reg==False):   
                            y=np.zeros([len(ycats)])
                            
                            # Calculating the ys using softmax:
                            
                            for i in range(len(ycats)):
                                    y[i]=self.softmax1(z2,v2,i,np.arange(len(z2)))
                            
                            # Calculating dv2:

                            err0=np.zeros([len(ycats)])
                            for i in range(len(ycats)):
                                if(row.iloc[yind]==ycats[i]):
                                    rs=1
                                else:
                                    rs=0
                                err0[i]=(rs-y[i])
                                prod=np.zeros([nhd[1]+1])
                                for l in range(nhd[1]+1):
                                    prod[l]=err0[i]*z2[l]
                                    
                                dv2[i]+=n*prod
                            
                            # Calculating dv1:
                            
                            err2=np.zeros([nhd[1]])
                            for l in range(nhd[1]):
                                sume1=0
                                for i in range(len(ycats)):
                                    sume1=sume1+(err0[i]*v2[i,l])
                                z2m=z2[l+1]*(1-z2[l+1])
                                
                                err2[l]=sume1*z2m
                                
                                prod=np.zeros([nhd[0]+1])
                                for h in range(nhd[0]+1):
                                    prod[h]=err2[l]*z1[h]
                                dv1[l]+=n*prod
                            
                            # Calculating dw:
                            
                            err1=np.zeros([nhd[0]])
                            for h in range(nhd[0]):
                                sume2=0
                                for l in range(nhd[1]):
                                    sume2=sume2+(err2[l]*v1[l,h])
                                z1m=z1[h+1]*(1-z1[h+1])
                                
                                err1[h]=sume2*z1m
                                prod=np.zeros([len(xinds)])
                                for j in range(len(xinds)):
                                    prod[j]=err1[h]*row.iloc[xinds[j]]
                                dw[h]+=n*prod
                        
                        # Regression:
                        
                        else:
                            
                            # Calculating y linearly:
                            
                            y=0
                            for j in range(len(z2)):
                                    y=y+v2[j]*z2[j]
                                    
                            # Calculating dv2:
                            
                            err0=0
                            rs=row.iloc[yind]
                            err0=(rs-y)
                            prod=np.zeros([nhd[1]+1])
                            for l in range(nhd[1]+1):
                                prod[l]=err0*z2[l]  
                            dv2+=n*prod
                            
                            # Calculating dv1:
                            
                            err2=np.zeros([nhd[1]])
                            for l in range(nhd[1]):
                                sume1=err0*v2[l]
                                z2m=z2[l+1]*(1-z2[l+1])
                                err2[l]=sume1*z2m
                                prod=np.zeros([nhd[0]+1])
                                for h in range(nhd[0]+1):
                                    prod[h]=err2[l]*z1[h]
                                dv1[l]+=n*prod
                            
                            # Calculating dw:
                            
                            err1=np.zeros([nhd[0]])
                            for h in range(nhd[0]):
                                sume2=0
                                for l in range(nhd[1]):
                                    sume2=sume2+(err2[l]*v1[l,h])
                                z1m=z1[h+1]*(1-z1[h+1])
                                
                                err1[h]=sume2*z1m
                                prod=np.zeros([len(xinds)])
                                for j in range(len(xinds)):
                                    prod[j]=err1[h]*row.iloc[xinds[j]]
                                dw[h]+=n*prod
                           
                            
                    # Updating our coefficients at the end of each splitted df:
                    
                    if(reg==False):
                        for i in range(len(ycats)):
                            v2[i]=v2[i]+dv2[i]
                    else:
                        v2=v2+dv2
                    if(len(v1)==1):
                        v1=v1+dv1
                    else:
                        for l in range(nhd[1]):
                            v1[l]=v1[l]+dv1[l]   
                    if(len(ws)==1):
                        ws=ws+dw
                    else:
                        for h in range(nhd[0]):
                            ws[h]=ws[h]+dw[h] 

            # Calculating the maximum error difference between the previous and current
            #   coefficients:
            
            count+=1
            errsw=ws.copy()
            for i in range(len(errsw)):
                if(nhl==0 and reg):
                    errsw[i]=np.abs(ws[i]-oldws[i])/np.abs(oldws[i])
                else:
                    for j in range(len(ws[0])):
                        errsw[i,j]=np.abs(ws[i,j]-oldws[i,j])/np.abs(oldws[i,j])
            err=np.max(errsw)
            if(nhl>=1):
                errsv1=v1.copy()
                for i in range(len(errsv1)):
                    if(reg and nhl==1):
                        errsv1[i]=np.abs(v1[i]-oldv1[i])/np.abs(oldv1[i])
                    else:
                        for j in range(len(errsv1[0])):
                            errsv1[i,j]=np.abs(v1[i,j]-oldv1[i,j])/np.abs(oldv1[i,j])
                
                if(nhl==2):
                    errsv2=v2.copy()
                    for i in range(len(errsv2)):
                        if(reg):
                            errsv2[i]=np.abs(v2[i]-oldv2[i])/np.abs(oldv2[i])
                        else:
                            for j in range(len(errsv2[0])):
                                errsv2[i,j]=np.abs(v2[i,j]-oldv2[i,j])/np.abs(oldv2[i,j])
                    err=np.max([np.max(errsw),np.max(errsv1),np.max(errsv2)])
                else:
                    err=np.max([np.max(errsw),np.max(errsv1)])

        return ws,err,count,v1,v2

    
    # Method softmax1 performs the softmax function P(C_i|x) on a single row
    # The method take the following input parameters:
    #   row: The x values
    #   ws: The weighted coefficients
    #   i: The i value in P(C_i|x)
    #   inds: The indicies in row upon which to consider for our linear 
    #       sum product
    # The method returns:
    #   num/den: The calculated softmax value  

    def softmax1(self,row,ws,i,inds):
        num=0
        den=0
        for ind in range(len(inds)):
            num=num+ws[i][ind]*row[inds[ind]]
        num=np.exp(num)
        for j in range(len(ws)):
            den1=0
            for ind in range(len(inds)):
                den1=den1+ws[j][ind]*row[inds[ind]]
            den1=np.exp(den1)
            den=den+den1
        if(math.isnan(num/den)):
            return 0
        else:
            return num/den
    
    # Method softmax performs the softmax function P(C_i|x) on an entire dataframe
    # The method take the following input parameters:
    #   df: The dataframe
    #   ws: The weighted coefficients
    #   i: The i value in P(C_i|x)
    #   inds: The columns in df upon which to consider for our linear 
    #       sum product
    # The method returns:
    #   num/den: A dataframe with the calculated softmax value of every row of df  
        
    def softmax(self,df,ws,i,inds):
        num=0
        den=0
        for ind in range(len(inds)):
            num=num+ws[i][ind]*df.iloc[:,inds[ind]]
        num=np.exp(num.astype(float))
        for j in range(len(ws)):
            den1=0
            for ind in range(len(inds)):
                den1=den1+ws[j][ind]*df.iloc[:,inds[ind]]
            den1=np.exp(den1.astype(float))
            den=den+den1
        return num/den
    
    # Method sigmoid_log performs the logistic sigmoid function on an entire df
    # The method take the following input parameters:
    #   df: The dataframe
    #   ws: The weighted coefficients
    #   i: The i value in P(C_i|x)
    #   inds: The columns in df upon which to consider for our linear 
    #       sum product
    # The method returns: A dataframe with the calculated sigmoid value of every
    #   row of df
    
    def sigmoid_log(self,df,ws,i,inds):
        summer=0
        for ind in range(len(inds)):
            summer=summer+ws[i][ind]*df.iloc[:,inds[ind]]
        sig=np.exp(-summer.astype(float))
        return (1/(1+sig))
    
    # Method sigmoid_log1 performs the logistic sigmoid function on a single row
    # The method take the following input parameters:
    #   row: The x values
    #   ws: The weighted coefficients
    #   i: The i value in P(C_i|x)
    #   inds: The indicies in row upon which to consider for our linear 
    #       sum product
    # The method returns: The calculated sigmoid value
    
    def sigmoid_log1(self,row,ws,i,inds):
        summer=0
        for ind in range(len(inds)):
            summer=summer+ws[i][ind]*row.iloc[inds[ind]]
        sig=np.exp(-summer.astype(float))
        return (1/(1+sig))
    
    
    # Method validate performs the backward propagation gradient descent on 
    #   the coefficients (w,v1,v2) required for the Feedforward Neural Network
    #   then proceeds to test these values on a validation set.
    #
    # The method proceeds to return the converged coefficients, the number of 
    #   hidden dimensions per hidden layer, the learning rate that led to the 
    #   convergence, and the number of epochs that were required
    #
    #  
    # The method functions as follows:
    #   Initializes n (learning rate) to 1
    #   Initializes runs (maximum number of epochs) to 1000
    #   Initializes the number of hidden dimensions (nhd) to 1
    #   Iteratively performs gradient on the inputted dataframe (df_train)
    #   If it did not converge, divide n by 10
    #   If it did converge: 
    #       predict the values of our validation set using the converged 
    #       coefficients
    #       If validation error <= best error:
    #           If validation error< best error:
    #               set best error (bestpe), best coefficients(bestws,bestv1,bestv2),
    #               best number of hidden dimensions(bestnhd), best learning rate(bestn),
    #               best epochs (bestrun) to current status
    #           Elif validation error==best error:
    #               If current error== past 2 best errors then break
    #       Elif validation error> besterror:
    #           break
    #   If nhd < number of inputs of that layer-1, increment nhd
    #   Else If n<=0.1 and still did not converge, break
    #
    # The method take the following input parameters:
    #   df_train: The training set upon which to perform backprop gradient descent
    #   df_v: The validation set upon which to asses our coefficients performance
    #   nhl: The number of hidden layers
    #
    # The method returns the following:
    #   bestws,bestv1,bestv2: The converged coefficients
    #   bestnhd: The number of hidden dimensions per hidden layer that led to 
    #       convergance
    #   bestn: The learning rate (n) that led to convergance
    #   bestrun: The number of epochs required to converge
    
    def validate(self,df_train,df_v,nhl):
        
        # Initializing the required variables:
        
        n=1
        e=10**6
        bestn=n
        runs=1000
        if(nhl==0):
            nhd=0
        elif(nhl==1):
            nhd=1
        else:
            nhd=[1,1]
        bestpe=10**6
        bestps=[]
        bestnhd=0
        bestv2=[]
        bestv1=[]
        bestrun=0
        bestws=[]
        
        # Begining our iterative loop with a percentage error difference of less
        #   than 10**-1 required to indicate convergance:
        
        while(n):
            
            # Running backprop gradient descent on the training set:
            
            ws,e,c,v1,v2=self.train(n,nhl,nhd,df_train,runs)
            if(e<10**-1):
                
                # If converges, predict validation set values:
                
                yp,pe=self.predict(df_v,ws,nhl,v1,v2)
                
                # Classification:
                
                if(self.reg==False):
                    
                    # pe: current validation error
                    # bestpe: best validation error
                    
                    if(pe<=bestpe):
                        
                        # Append current validation error for comparison purposes:
                        
                        bestps.append(pe)
                        
                        if(pe<bestpe):
                            bestpe=pe
                            bestws=ws.copy()
                            if(nhl>0):
                                bestv1=v1.copy()
                                if(nhl==2):
                                    bestv2=v2.copy()
                                    bestnhd=nhd.copy()
                                else:
                                    bestnhd=nhd
                            bestn=n
                            bestrun=c
                        
                        # if pe==bestpe, compare with previous bestpes
                        else:
                            if(len(bestps)>3):
                                if(pe==bestps[-2]):
                                    break
                    elif(pe>bestpe and (len(bestps)>3)):
                        break
                
                # Regression:
                
                else:
                    
                    # 10**-3 is set as a threshold for comparing MSE values:
                    
                    if(((np.abs(pe-bestpe)/(bestpe))<=10**-3)or pe<bestpe):
                        bestps.append(pe)
                        if(((pe-bestpe)/bestpe)<-10**-3):
                            bestpe=pe
                            bestws=ws
                            if(nhl>0):
                                bestv1=v1
                                if(nhl==2):
                                    bestv2=v2
                                    bestnhd=nhd.copy()
                                else:
                                    bestnhd=nhd
                            bestn=n
                            bestrun=c
                        else:
                            if(len(bestps)>3):
                                if((np.abs(pe-bestps[-2])/bestps[-2])<=10**-3):
                                    break
                    elif(pe>bestpe and (len(bestps)>2)):
                        break
            else:
                
                # If did not converge, divide n by 10
                
                if(n>0.01):
                    n=n/10
                    continue
            
            # Tuning nhds:
            
            if(nhl==1):
                if(nhd<len(self.xinds)-2):
                                nhd+=1
                else:
                    nhd=1
                    if(n<0.1):
                        break
                    else:
                        n=n/10
            elif(nhl==2):
                if(nhd[1]<nhd[0]-1):
                    nhd[1]+=1
                else:
                    nhd[1]=1
                    if(nhd[0]<len(self.xinds)-1):
                        nhd[0]+=1
                    else:
                        nhd[0]=1
                        if(n<0.1):
                            break
                        else:
                            n=n/10
            
            # If number of hidden dimensions == number of inputs of that
            #   dimension-1 and no converges happened, then break
            else:
                break
                
        return bestws,bestv1,bestv2,bestnhd,bestn,bestrun
        
        
    # Method fitTest does the following:
    #   Extracts 10% of our dataframe as a validation set
    #   Performs 5-fold cross validation on the remaining 90%
    #   Takes 4 of the 5 folds as training data and runs the validate method on
    #       training set and validation set
    #   Predicts the values of the remaining fold as the test set and records
    #       the error
    #
    # The method returns the following:
    #   ws: The converged coefficients
    #   ys: An array of the predicted values of the test set
    #   errs: The errors of the folds
    #   np.mean(errs): The mean of the errors
    #
    # Furthermore, the method prints the results in a user friendly way 
       
    def fitTest(self,nhl=0):
        
        df=self.df
        
        # Shuffling our dataframe:
        
        df=df.sample(frac=1).reset_index(drop=True)
        
        # Determining the size of the 10% for our validation set:
        
        tenp=np.round(len(df)/10,0).astype(int)
        
        # df_v: validation set
        # df_t: test/train sets
        
        df_v=df.iloc[:tenp,:]
        df_t=df.iloc[tenp:,:]
        
        # dfs: our 5 folds
        
        dfs=self.fivefoldcv(df_t)
        

        
        errs=[]
        ys=[]
        ws=[]
        

    
    
        df_train=dfs[0][0]
        df_test=dfs[0][1]
        

        # Performing the validate method on the validation set:
        w,v1,v2,nhd,n,c=self.validate(df_train,df_v,nhl)
        
        # Predicting our test set values using the coefficient returned from our
        #   validate method:
        
        if(w==[]):
            print("Feedforward Neural Network Model for Fold "+str(0+1)+":")
            print("Did not converge")
        else:
        
            y,err=self.predict(df_test,w,nhl,v1,v2)
            ys.append(y)
            errs.append(err)
            ws.append(w)
            print("Feedforward Neural Network Model for Fold "+str(0+1)+":")
            print("Number of Hidden Layers: "+str(nhl))
            print("Number of Hidden Dimensions: "+str(nhd))
            print("n: "+str(n))
            print("Epochs: "+str(c))
            if(self.reg==False):
                df_res=pd.DataFrame({'Actual':df_test.iloc[:,self.yind],'Predicted':y})
            else:
                if(self.norm==True):
                    ya=df_test.iloc[:,self.yind]*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                    ypred=np.array(y)*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                    df_res=pd.DataFrame({'Actual':ya,'Predicted':ypred})

            print(df_res.to_string())
            print("Error: "+str(np.round(err,2)))
            print()
    
        return ws,ys,errs,np.mean(errs)  

    
    
    
   
        
    

    # Method predict predicts the target variable values of a dataframe "df" 
    #   using coefficients "ws","v1","v2"
    # The method take the following input parameters:
    #   df: The dataframe who's values to predict
    #   ws,v1,v2: The coefficients to use
    #   nhl: The number of hidden layers
    # The method returns:
    #   ypred: The predicted target values
    #   cer: The error
    
    
    def predict(self,df,ws,nhl,v1=[],v2=[]):
        xinds=self.xinds
        yind=self.yind
        
        ypred=[]
        if(nhl==2):
            nhd=[len(ws),len(v1)]
        elif(nhl==1):
            nhd=len(ws)


        if(self.reg==False):
            ycats=self.ycats
            for index,row in df.iterrows():
                ps=[]
                if(nhl==0):
                    for i in range(len(ycats)):
                        sump=0
                        for j in range(len(ws[0])):
                            sump=sump+ws[i][j]*row.iloc[xinds[j]]
                        ps.append(sump)
                    
                elif(nhl==1):
                    z1=np.ones(len(ws)+1)
                    for h in range(nhd):
                        z1[h+1]=self.sigmoid_log1(row,ws,h,xinds)
                    for i in range(len(ycats)):
                        ps.append(self.softmax1(z1,v1,i,np.arange(len(z1)).tolist()))
                else:

                    z1=np.ones(len(v1[0]))
                    for l in range(nhd[0]):
                        z1[l+1]=self.sigmoid_log1(row,ws,l,xinds)
                    z2=np.ones(len(v2[0]))
                    z1d=pd.DataFrame(data=z1)
                    for h in range(nhd[1]):
                        z2[h+1]=self.sigmoid_log1(z1d,v1,h,np.arange(nhd[0]+1).tolist())
                    for i in range(len(ycats)):
                        ps.append(self.softmax1(z2,v2,i,np.arange(len(z2)).tolist()))
                maxpi=-1
                maxp=-1
                for i in range(len(ps)):
                    if(ps[i]>maxp):
                        maxp=ps[i]
                        maxpi=i
                
                ypred.append(ycats[maxpi])
            
            countin=0
            for t in range(len(ypred)):
                if(ypred[t]!=df.iloc[t,yind]):
                    countin+=1
            
            cer=countin*100/len(df)
        else:
            for index,row in df.iterrows():
                if(nhl==0):
                    sump=0
                    for j in range(len(ws)):
                        sump=sump+ws[j]*row.iloc[xinds[j]]
                    y=sump
                    
                elif(nhl==1):
                    z1=np.ones(len(ws)+1)
                    for h in range(nhd):
                        z1[h+1]=self.sigmoid_log1(row,ws,h,xinds)
                    y=np.dot(v1,z1)
                else:
                    z1=np.ones(len(v1[0]))
                    for l in range(nhd[0]):
                        z1[l+1]=self.sigmoid_log1(row,ws,l,xinds)
                    z2=np.ones(len(v2))
                    z1d=pd.DataFrame(data=z1)
                    for h in range(nhd[1]):
                        z2[h+1]=self.sigmoid_log1(z1d,v1,h,np.arange(nhd[0]+1).tolist())
                    y=np.dot(v2,z2)
                
                ypred.append(y)
            

            
            cer=self.MSError(ypred,df,yind)
            
            
        
        return ypred,cer
                
    
    
    # Method MSError returns the mean squared error of a regression prediction
    # The method take the following input parameters:
    #   ypred: The predicted values
    #   df: The dataframe who's target values were predicted
    #   yind: The index of the target variable in "df"
    # The method returns:
    #   mse: The mean squared error associated with the prediction
                  
    def MSError(self,ypred,df=pd.DataFrame(),yind=None):
            
            if(df.empty):
                df=self.df
            
            if(yind==None):
                yind=self.yind
                
            mse=np.sum((np.array(df.iloc[:,yind])-ypred)**2)/len(df)
            return mse           
                
                    
                        
                


