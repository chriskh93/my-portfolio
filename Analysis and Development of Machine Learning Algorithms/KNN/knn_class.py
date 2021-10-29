# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:54:32 2020

@author: Christopher El Khouri
        605.649.81
    
"""

# Importing the necesarry libraries:

import pandas as pd
import numpy as np

# class knn_class implements a k-nearest neighbors classifier algorithm

class knn_class:
    
    # Constructor takes the follwing parameters:
    #   df: The pandas dataframe that requires classification
    #   xinds: The indices of the x-variables within df
    #   yinds: The index of the y-variable within df (The class attributes)
    
    # The constructor further initializes the following variables:
    #   y_cats: The different possible values of our y-variable
    #   k: The ideal value of k to be used in training our k-nn classifier
    #   n_ycats: An array used to perform startified cross validation
    
    def __init__(self,df, xinds, yinds):
        self.df=df
        self.xinds=xinds
        self.yinds=yinds
        self.y_cats=df.iloc[:,yinds].unique()
        self.k=0
        self.n_ycats=0
    
    # Method eucd calculates the euclidean distance between two variables
    # The method take the following input parameters:
    #   x1: Numerical array of data
    #   x2: Numerical array of data
    # The method returns false if the sizes of the input variables are not equal
    # The method returns the euclidean distance between the two variables if the 
    #   sizes are equal
        
    def eucd(self,x1,x2):
        if(len(x1)!=len(x2)):
            return False
        else:
            y=0
            for i in range(len(x1)):
                y=y+np.sqrt((x1[i]-x2[i])**2)
            return y
    
    # Method tune tunes the value of k based on the following:
    #   10% of the dataframe is taken and stratified based on the proportion
    #       of different classes within the entire dataset
    #   For every k in range (1,16) with increments of 2
    #   tune calculates the accuracy of our model using k
    #   The k value resulting in the highest accuracy is returned
    #   The private variable k is replaced accordingly
    #   The private variable df is replaced with the dataframe containing the 
    #       remaining 90% of the data
    
    def tune(self):
        df=self.df.sample(frac=1)
        y_cats=self.y_cats
        n_ycats=[]
        yind=self.yinds
        
        # Stratifying the tuning dataset:
        
        for i in range(len(y_cats)):
            n_ycats.append(len(df[df.iloc[:,yind]==y_cats[i]]))
        n_ycats=np.divide(n_ycats,len(df))
        self.n_ycats=n_ycats
        tenp=np.floor(len(df)/10).astype(int)
        n_ycats=np.round(n_ycats*tenp,0).astype(int)
        df_tenp_0=pd.DataFrame(columns=df.columns)
        for i in range(len(y_cats)):
            df_tenp_0=df_tenp_0.append(df[df.iloc[:,yind]==y_cats[i]].iloc[:n_ycats[i],:])
        df_tenp_0=df_tenp_0.sample(frac=1).reset_index(drop=True)
        
        k=1
        bestacc=0
        bestk=k
        tl=len(df_tenp_0)
        ypred=[]
        xinds=self.xinds

        # Tuning k:
        
        for k in np.arange(1,16,2):
            df_tenp=df_tenp_0.copy()
            ypred=[]
            
            # Iterating through the tuning dataset:
            
            for index,row in df_tenp.iterrows():
                df_tenp_wd=df_tenp
                df_tenp_wd['Distances']=0
                endind=len(df_tenp_wd.iloc[0,:])-1
                
                # Calculating the euclidean distances:
                
                for index1,row1 in df_tenp_wd.iterrows():
                    x1=np.array(df_tenp_wd.iloc[index,xinds])
                    x2=np.array(df_tenp_wd.iloc[index1,xinds])
                    dist=self.eucd(x1,x2)
                    df_tenp_wd.iloc[index1,endind]=dist
                srted=df_tenp_wd.sort_values('Distances').reset_index(drop=True)
                
                # Exempting the variable itself from the nearest neighbors:
                
                srted=srted.iloc[1:1+k,:]
                
                # Returning the most common class:
                
                cs=srted.iloc[:,yind].mode()
                ypred.append(cs[0])
            
            # Calculating the accuracy:
            
            count=0
            y=df_tenp.iloc[:,yind]
            for ii in range(len(ypred)):
                if(ypred[ii]!=y[ii]):
                    count=count+1
            
            accuracy=1-count/tl
            accuracy=accuracy*100
            
            # Determining the performance in comparison with the other ks:
            
            if accuracy>bestacc:
                bestacc=accuracy
                bestk=k
          
        self.k=bestk
        self.df=df.iloc[tenp:,:]
        return bestk
    
    
    # Method fit performs the k-nearest neighbor classification on the dataset
    # The k value previously tuned is used for the classification
    # Stratified 5-fold cross validation is utilized for the fitting and testing
    # df_train represents the training set
    # df_test represents the test set
    # The method prints df_res which represents a dataframe that compares the 
    #   predicted values with the actual values
    # The method returns the following:
    #   accs: The accuracy of each test fold of the 5 fold cross validation
    #   np.mean(accs): The mean of the accuracies of all 5 test folds
    
    def fit(self):
        k=self.k
        df_1=self.df.reset_index(drop=True)
        xinds=self.xinds
        yinds=self.yinds
        y_cats=self.y_cats
        n_ycats=self.n_ycats
        accs=[]
        tl=len(df_1)
        n_ycats=np.round(n_ycats*tl,0).astype(int)
        starts=np.zeros(len(n_ycats)).astype(int)
        ends=np.ceil(n_ycats/5).astype(int)
        df=pd.DataFrame(columns=df_1.columns)
        
        # Stratifying the folds:
        
        for i in range(5):
            for j in range(len(n_ycats)):
                if(i!=4):
                    df=df.append(df_1[df_1.iloc[:,yinds]==y_cats[j]].iloc[starts[j]:ends[j]])
                else:
                    df=df.append(df_1[df_1.iloc[:,yinds]==y_cats[j]].iloc[starts[j]:])
            starts=ends
            ends=np.ceil(ends+n_ycats/5).astype(int)
        
        # Performing stratified 5-fold cross validation:
        
        start=0
        for ii in range(5):
            end=np.ceil((tl*(ii+1)/5))
            end=end.astype(int)
            df_test=df.iloc[start:end,:].reset_index(drop=True)
            if(start==0):
                df_train=df.iloc[end:tl].reset_index(drop=True)
            else:
                df_train=pd.concat([df.iloc[0:start,:],df.iloc[end:tl,:]]).reset_index(drop=True)
            ypred=[]
            df_test=df_test.sample(frac=1).reset_index(drop=True)
            df_train=df_train.sample(frac=1).reset_index(drop=True)
            
            # Iterating through the test dataset:
            
            for index,row in df_test.iterrows():
                df_train_wd=df_train.copy()
                df_train_wd['Distances']=0
                endind=len(df_train_wd.iloc[0,:])-1
                
            # Calculating the euclidean distances between the test and train datasets:
            
                for index1,row1 in df_train_wd.iterrows():
                    x1=np.array(df_test.iloc[index,xinds])
                    x2=np.array(df_train_wd.iloc[index1,xinds])
                    dist=self.eucd(x1,x2)
                    df_train_wd.iloc[index1,endind]=dist
                
                # Finding the k nearest neighbors:
                
                srted=df_train_wd.sort_values('Distances').reset_index(drop=True)
                srted=srted.iloc[:k,:]
                
                # Returning the most common class:
                
                cs=srted.iloc[:,yinds].mode()
                ypred.append(cs[0])
            
            # Calculating the accuracy:
            
            count=0
            y=df_test.iloc[:,yinds]
            for i in range(len(ypred)):
                if(ypred[i]!=y[i]):
                    count=count+1
            
            accuracy=1-count/len(y)
            accuracy=accuracy*100
            accs.append(accuracy)
            start=end
            print("K-NN Classifier Fold "+str(ii+1)+":")
            print("k="+str(k))
            print("Accuracy: "+str(np.round(accuracy,2)))
            df_res=pd.DataFrame({'Actual':y,'Predicted':ypred})
            print(df_res.to_string())
            print()
        return accs,np.mean(accs)
                

                

                
                                    