# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:06:49 2020

@author: Christopher El Khouri
        605.649.81
"""


# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import random
import math

# class logReg implements the Logistic Regression modeling algorithm

class logReg:
    
    # Constructor takes the follwing parameters:
    #   df: The pandas dataframe that requires modeling
    #   xinds: The indices of the x-variables within df
    #   yind: The index of the y-variable within df (The target)
    
    # The constructor further initializes the following variable:
    #   y_cats: The different possible values of our y-variable (target)

    
    def __init__(self,df, xinds, yind):
        self.df=df
        self.xinds=xinds
        self.yind=yind
        self.ycats=df.iloc[:,self.yind].unique()

    
        
    # Method validate performs gradient descent on the coefficients (w) required
    #   for the logistic regression algorithm and returns the converged
    #   coefficients, the learning rate that led to the convergence, and the
    #   number of epochs that were required
    #  
    # The method functions as follows:
    #   Initializes n (learning rate) to 1
    #   Initializes runs (maximum number of epochs) to 1000
    #   Iteratively performs gradient descent on the inputted dataframe (df_train)
    #   If w did not converge, divide n by 10
    #   If n<=0.01 and w still did not converge, multiply runs by 10
    #
    # The method take the following input parameters:
    #   df_train: The dataframe upon which to perform gradient descent on
    #
    # The method returns the following:
    #   ws: The converged coefficients
    #   bestn: The learning rate (n) that led to convergance
    #   bestrun: The number of epochs required to converge


    
    def validate(self,df_train):
        
        # Initializing the required variables:
        
        n=1
        runs=1000
        e=10**6
        bestn=n
        bestrun=runs
        w=[]
        nes=[]
        
        # Begining our iterative loop with a percentage error difference of less
        #   than 10**-2 required to indicate convergance:
        
        while(e>10**-2):
            
            # If runs>1000, this means that we have iteratively looped through
            #   the different ns without convergance, therefore, we can pick up
            #   where the gradient descent function left off at 1000 runs:
            
            if(runs>1000):
                for i in range(len(nes)):
                    if(nes[i][0]==n and nes[i][2]<runs):
                        w=nes[i][1]
            
            # Performing the gradient descent method:
                
            ws,e,c=self.gradDesc(df_train,n,runs,w)
            
            w=[]
            
            # Recording the different n,ws, and runs values:
            
            ne=[]
            ne.append(n)
            ne.append(ws)
            ne.append(runs)
            nes.append(ne)
            
            # If convergance occured, store n and epoch values:
            
            if(e<10**-2):
                bestn=n
                bestrun=c
                if(runs>1000):
                    bestrun+=runs/10
            
            
            # If the percentage error returned from the gradient descent function
            #   is nan, then change the error value to 100 so as to keep the loop
            #   running:
                    
                    
            elif(math.isnan(e)):
                e=100
                
            
            # if n<=0.01 and convergance has not occured then increase the
            #   maximum number of epochs and reset n to 1:
            
            if(n<=0.01):
                runs=runs*10
                n=1
            
            # else, decrease n:
            
            else:
                n=n/10
                
                
        return ws,bestn,bestrun
        
            
    
    
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
    
    
    
    # Method fitTest does the following:
    #   Extracts 10% of our dataframe as a validation set
    #   Performs the method validate on the validation set and returns the ideal
    #       learning rate to use with our training set
    #   Performs 5-fold cross validation on the remaining 90%
    #   Takes 4 of the 5 folds as training data and performs gradient descent
    #       using the ideal learning rate to produce the coefficients 
    #   Predicts the values of the remaining fold as the test set and records
    #       the classification error
    #
    # The method returns the following:
    #   ws: The converged coefficients used for logistic regression
    #   ys: An array of the predicted values of the test set
    #   errs: The classification errors of the folds
    #   np.mean(errs): The mean of the errors
    #
    # Furthermore, the method prints the pruned and unpruned results in a user
    #   friendly way
    
    def fitTest(self):
        
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
        
        # Performing the validate method on the validation set:
        
        w,n,r=self.validate(df_v)
        for i in range(len(dfs)):
            df_train=dfs[i][0]
            df_test=dfs[i][1]
            
            # Performing gradient descent using the n returned from the
            #   validation:
            
            w,e,r=self.gradDesc(df_train,n,10000)
            
            # Predicting our test set values using the w returned from our
            #   gradient descent method:
            
            y,err=self.predict(df_test,w)
            ys.append(y)
            errs.append(err)
            ws.append(w)
            print("Logistic Regression Model for Fold "+str(i+1)+":")
            print("n: "+str(n))
            print("Epochs: "+str(r))
            df_res=pd.DataFrame({'Actual':df_test.iloc[:,self.yind],'Predicted':y})
            print(df_res.to_string())
            print("Error: "+str(np.round(err,2)))
            print()
        
        return ws,ys,errs,np.mean(errs)
        
            
    
    # Method gradDesc performs gradient descent to generate the coefficients to 
    #   be used in logistic regression modeling
    #
    # The method take the following input parameters:
    #   df: The dataset
    #   n: The learning rate 
    #   maxc: The maximum number of epochs
    #   ws: Initial coefficients to consider, defaulted as empty
    #
    # The method returns:
    #   ws: The converged coefficients
    #   err: The error difference between the previous ws and the current ws
    #   count: The number of epochs
    
    
    
    def gradDesc(self,df,n,maxc=1000,ws=[]):
        xinds=self.xinds
        yind=self.yind
        ycats=self.ycats
        
        # If ws is empty, then initialize the coefficients to be random numbers
        #   between -0.01 and 0.01:
        
        if(ws==[]):    
            ws=np.zeros([len(ycats),len(xinds)+1])
            
            for i in range(len(ycats)-1):
                for j in range(len(xinds)+1):
                    ws[i,j]=random.uniform(-0.01,0.01)
            
        err=10**6
        count=0
        
        # 10**-2 is considered as the convergance threshold
        
        while((err>10**-2) and count<maxc):
            
            # oldws is set as the ws from the last iteration:
            
            oldws=ws.copy()
            
            # dws and os are initialized to 0s:
            
            dws=np.zeros([len(ycats)-1,len(xinds)+1])
            os=np.zeros([len(df),len(ycats)])

            
            # Calculating the os:
            
            for i in range(len(ycats)-1):
                for j in range(len(xinds)+1):
                    if(j==0):
                        os[:,i]=os[:,i]+ws[i][j]
                    else:
                        os[:,i]=os[:,i]+ws[i][j]*df.iloc[:,xinds[j-1]]
                        
            # Calculating the ys:
            
            ys=[]
            for i in range(len(ycats)):
                calcn=np.exp(os[:,i])
                calcd=np.sum(np.exp(os),axis=1)
                calc=calcn/calcd
                for cal in range(len(calc)):
                    if math.isnan(calc[cal]):
                        calc[cal]=1
                ys.append(calc)
            
            # Calculating the deltas:
            
            deltas=[]    
            for i in range(len(ycats)-1):
                rs=df.iloc[:,yind].apply(lambda x: 1 if x==ycats[i] else 0)
            
                deltas.append(rs-ys[i])
                for j in range(len(xinds)+1):
                    if(j==0):
                        dws[i,j]=dws[i,j]+np.mean(deltas[i])
                    else:
                        xd=deltas[i]*df.iloc[:,xinds[j-1]]
                        dws[i,j]=dws[i,j]+np.mean(xd)
                            
                
            # Calculating the updated ws values:
            
            for i in range(len(ycats)-1):
                for j in range(len(xinds)+1):
                    ws[i,j]=ws[i,j]+n*dws[i,j]
            
            # Calculating the error difference between the last ws and the
            #   current ws:
            
            errs=ws[:-1].copy()
            for i in range(len(errs)):
                for j in range(len(ws[0])):
                    errs[i,j]=np.abs(ws[i,j]-oldws[i,j])/np.abs(oldws[i,j])
            
            # Calculating the maximum error difference:
                    
            err=np.max(errs)
            count+=1
        
        return ws,err,count
        
    

    # Method predict predicts the target variable values of a dataframe "df" 
    #   using coefficients "ws"
    # The method take the following input parameters:
    #   df: The dataframe who's values to predict
    #   ws: The coefficients to use
    # The method returns:
    #   ypred: The predicted target values
    #   cer: The classification error
    
    
    def predict(self,df,ws):
        xinds=self.xinds
        yind=self.yind
        ycats=self.ycats
        
        ypred=[]
        
        for index,row in df.iterrows():
            ps=[]
            for i in range(len(ycats)):

                numcal=ws[i,0]
                numsum=0
                for j in range(len(xinds)):
                    numsum=numsum+ws[i,j+1]*df.iloc[index,xinds[j]]
                num=np.exp(numcal+numsum)
                
                
                den=0
                for i1 in range(len(ycats)):
                    dencal=ws[i1,0]
                    densum=0
                    for j in range(len(xinds)):
                        densum=densum+ws[i1,j+1]*df.iloc[index,xinds[j]]
                    dencal=np.exp(dencal+densum)
                    den=den+dencal
                
                p=num/den
                if(math.isnan(p)):
                    p=1
                ps.append(p)
            
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
            
        
        return ypred,cer
                
                    
                
                
                    
                        
                
