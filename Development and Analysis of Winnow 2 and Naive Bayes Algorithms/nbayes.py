# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:29:34 2020

@author: Christopher El Khouri
        605.649.81
    
"""

# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import random

# class nbayes implements a Naive Bayes classifier algorithm
class nbayes:

    # Constructor takes the follwing parameters:
    #   df: The pandas dataframe that requires classification
    #   xinds: The indices of the x-variables within df
    #   yinds: The index of the y-variable within df (The class attributes)
    #   m: The variable m taken for smoothing purposes
    #   p: The variable p taken for smoothing purposes
    
    # The constructor further initializes the following variables:
    #   y_cats: The different possible values of our y-variable
    #   x_cats: The different possible values of our x-variables
    #   npsx: An array containing the probabilities of our x-variables occuring 
    #         given a y-value
    #   npsy: An array containing the probabilites of our y-variables occuring
    
    def __init__(self,df, xinds, yinds,m=0,p=0):
        self.df=df
        self.xinds=xinds
        self.yinds=yinds
        self.y_cats=df.iloc[:,yinds].unique()
        x_cats=[]
        for i in xinds:
            xcat1=df.iloc[:,i].unique()
            x_cats.append(xcat1)
        self.x_cats=x_cats
        self.m=m
        self.p=p
        self.npsx=0
        self.npsy=0
    
    # Method tune tunes the values of m and p based on the following:
    #   10% of the dataframe is taken
    #   For every m in range (1,10) with increments of 1
    #   For every p in range (0.001,0.01) with increments of 0.001
    #   tune calculates the accuracy of our model using the m and p variables
    #   The m and p values resulting in the highest accuracy are returned
    #   The private variables m and p are replaced accordingly
    #   The private variable df is replaced with the dataframe containing the 
    #       remaining 90% of the data
       
    def tune(self):
        df=self.df.sample(frac=1)
        tenp=np.floor(len(df)/10).astype(int)
        df_tenp=df.iloc[:tenp,:]
        x_cats=self.x_cats
        y_cats=self.y_cats
        m=1
        bestacc=0
        bestm=m
        p=0.001
        bestp=p
        tl=len(df_tenp)
        ypred=[]
        xinds=self.xinds
        yind=self.yinds
        for m in np.arange(1,10,1):
            p=0.001
            for p in np.arange(0.001,0.01,0.001):
                proby=[]
                probxall=[]
                for i in range (len(y_cats)):
                    probx=[]
                    df_1=df_tenp[df_tenp.iloc[:,yind]==y_cats[i]]
                    proby.append(len(df_1)/len(df_tenp))
                    for j in range(len(x_cats)):
                        probx1=[]
                        for k in range(len(x_cats[j])):
                            if(len(df_1.iloc[:,j])==0):
                                pb=(len(df_1[df_1.iloc[:,j]==x_cats[j][k]])+p)/(len(df_1.iloc[:,j])+m)
                            else: 
                                pb=len(df_1[df_1.iloc[:,xinds[j]]==x_cats[j][k]])/len(df_1.iloc[:,j])
                            if(pb==0):
                                 pb=(len(df_1[df_1.iloc[:,j]==x_cats[j][k]])+p)/(len(df_1.iloc[:,j])+m)
                            probx1.append(pb)
                        probx.append(probx1)
                    probxall.append(probx)
                
                x=df_tenp.iloc[:,xinds]
                x=np.array(x)
                ypred=[]
                y=df_tenp.iloc[:,yind]
                y=np.array(y)
                for i in range(len(x)):
                    pmax=0
                    ycat=-1
                    for j in range(len(y_cats)):
                        p1=proby[j]
                        for k in range(len(x_cats)):
                            for l in range(len(x_cats[k])):
                                if(x[i][k]==x_cats[k][l]):
                                    p1=p1*probxall[j][k][l]
                            
                        if(p1>pmax):
                            pmax=p1
                            ycat=j
                    
                    ypred.append(y_cats[ycat])
                
                count=0
                
                for ii in range(len(ypred)):
                    if(ypred[ii]!=y[ii]):
                        count=count+1
                
                accuracy=1-count/tl
                accuracy=accuracy*100
                if accuracy>bestacc:
                    bestacc=accuracy
                    bestm=m
                    bestp=p
          
        self.m=bestm
        self.p=bestp
        self.df=df.iloc[tenp:,:]
        return bestm,bestp
    
    # Method fit calculates the Bayesian probabilites of the x and y variables
    #   to be used in the final model
    # 5-fold cross validation is utilized for the fitting
    # The private variables npsx and npsy are replaced accordingly with the final
    # Bayesian probabilities
    # The method returns the final Bayesian probabilities 

    def fit(self):
        m=self.m
        p=self.p
        df=self.df
        xinds=self.xinds
        yinds=self.yinds
        x_cats=self.x_cats
        y_cats=self.y_cats
        npsx=[]
        npsy=[]
        tl=len(df)
        start=0
        for ii in range(5):
            end=np.floor(tl*(ii+1)/5)
            end=end.astype(int)
            if(end>tl):
                end=tl-1
            df_1=df.iloc[start:end,:]
            proby=[]
            probxall=[]
            for i in range (len(y_cats)):
                probx=[]
                df_2=df_1[df_1.iloc[:,yinds]==y_cats[i]]
                proby.append(len(df_2)/len(df_1))
                for j in range(len(x_cats)):
                    probx1=[]
                    for k in range(len(x_cats[j])):
                        if(len(df_2.iloc[:,j])==0):
                            pb=(len(df_2[df_2.iloc[:,j]==x_cats[j][k]])+p)/(len(df_2.iloc[:,j])+m)
                        else:                            
                            pb=len(df_2[df_2.iloc[:,xinds[j]]==x_cats[j][k]])/len(df_2.iloc[:,j])
                        if(pb==0):
                             pb=(len(df_2[df_2.iloc[:,j]==x_cats[j][k]])+p)/(len(df_2.iloc[:,j])+m)
                        probx1.append(pb)
                    probx.append(probx1)
                probxall.append(probx)
                start=end
            npsx.append(probxall)
            npsy.append(proby)
        
        final_np_x=[]
        final_np_y=[]
        
        for i in range(len(y_cats)):
            pre_final_np_x=[]
            for j in range(len(x_cats)):
                pre_final_np_x_1=[]
                for k in range(len(x_cats[j])):
                    m_x=np.mean([npsx[0][i][j][k],npsx[1][i][j][k],npsx[2][i][j][k],
                            npsx[3][i][j][k],npsx[4][i][j][k]])
                    pre_final_np_x_1.append(m_x)
                pre_final_np_x.append(pre_final_np_x_1)
            final_np_x.append(pre_final_np_x)
            m_y=np.mean([npsy[0][i],npsy[1][i],npsy[2][i],npsy[3][i],npsy[4][i]])
            final_np_y.append(m_y)
        
        self.npsx=final_np_x
        self.npsy=final_np_y
        return final_np_x,final_np_y
    
    
    # Method test tests our fitted model
    # Since we utilized 5-fold cross validation , we will be returning the 
    #   results of one of the folds selected at random
    # The following variables are returned:
    #   ypred: The results of the classification performed on one of the folds
    #   df_1.iloc[:,yinds]: The actual values of y-index of the dataframe
    #   accuracy: The accuracy of our model

    def test(self):
        df=self.df
        xinds=self.xinds
        yinds=self.yinds
        x_cats=self.x_cats
        y_cats=self.y_cats
        npsx=self.npsx
        npsy=self.npsy
        tl=len(df)
        fold=random.randrange(0,5,1)
        end=np.floor(tl*(fold+1)/5)
        end=end.astype(int)
        start=(end-(np.floor(tl/5).astype(int)))
        start=start.astype(int)
        ypred=[]
        df_1=df.iloc[start:end]
        x=df_1.iloc[:,xinds]
        x=np.array(x)
        ypred=[]
        y=df_1.iloc[:,yinds]
        y=np.array(y)
        for i in range(len(x)):
            pmax=0
            ycat=-1
            for j in range(len(y_cats)):
                p1=npsy[j]
                for k in range(len(x_cats)):
                    for l in range(len(x_cats[k])):
                        if(x[i][k]==x_cats[k][l]):
                            p1=p1*npsx[j][k][l]
                    
                if(p1>pmax):
                    pmax=p1
                    ycat=j
            
            ypred.append(y_cats[ycat])
                
            count=0
            
            for ii in range(len(ypred)):
                if(ypred[ii]!=y[ii]):
                    count=count+1
            
            accuracy=1-count/len(y)
            accuracy=accuracy*100
            
        
        return ypred,df_1.iloc[:,yinds],accuracy
        
        

