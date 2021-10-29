# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:02:22 2020

@author: Christopher El Khouri
        605.649.81
    
"""

# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import random

# class winnow2 implements a Winnow-2 classifier algorithm

class winnow2:
    
    # Constructor takes the follwing parameters:
    #   x: The array of x-values (features)
    #   y: The array of y-values (classes)
    #   alpha: The variable alpha taken for promotion/demotion
    #   theta: The variable theta taken for promotion/demotion
    
    # The constructor further initializes the following variables:
    #   finalws: The array containing the coefficients for Winnow-2
    
    def __init__(self, x, y,alpha=2,theta=0.5):
        self.x=x
        self.y=y
        self.alpha=alpha
        self.theta=theta
        self.finalws=np.ones((5,len(x[0])))
    
    # Method tune tunes the values of alpha and theta based on the following:
    #   10% of the dataframe is taken
    #   For every alpha in range (1.1,10) with increments of 0.1
    #   For every theta in range (0.5,5) with increments of 0.1
    #   tune calculates the accuracy of our model using the alpha and theta variables
    #   The alpha and theta values resulting in the highest accuracy are returned
    #   The variables alpha and theta are replaced accordingly

    def tune(self,xt,yt):
        alpha=2
        bestacc=0
        bestalpha=alpha
        theta=0.5
        besttheta=theta
        for alpha in np.arange(1.1,10,0.1):
            for theta in np.arange(0.5,5,0.1):
                x=xt
                y=yt
                tl=len(x)
                ws=np.ones(len(x[0]))
                y_cats=np.unique(y)
                x1=x
                ypred=[]
                for j in range(len(x1)):
                    f=0
                    for k in range(len(x1[j])):
                        f=f+ws[k]*x1[j][k]
                    if(f>theta and y[j]<np.max(y_cats)):
                        for l in range(len(x1[j])):
                            if(x1[j][l]>0):
                                ws[l]=ws[l]/alpha
                    elif(f<theta and y[j]==np.max(y_cats)):
                        for l in range(len(x1[j])):
                            if(x1[j][l]>0):
                                ws[l]=ws[l]*alpha
                ypred=[]
                for i in range(len(x1)):
                    f=0
                    for j in range(len(x1[i])):
                        f=ws[j]*x1[i][j]
                    if(f>theta):
                        ypred.append(np.max(y_cats))
                    else:
                        ypred.append(np.min(y_cats))
                count=0
                for ii in range(len(ypred)):
                    if(ypred[ii]!=y[ii]):
                        count=count+1
                
                accuracy=1-count/tl
                accuracy=accuracy*100
                if accuracy>bestacc:
                    bestacc=accuracy
                    bestalpha=alpha
                    besttheta=theta
          
        self.alpha=bestalpha
        self.theta=besttheta
        return bestalpha,besttheta
        
        
    # Method fit calculates the coefficients of the x variables to be used 
    #   in the final model
    # 5-fold cross validation is utilized for the fitting
    # The private variables finalws are replaced accordingly
    # The method returns the final coefficients to be used
    
    def fit(self):
        x=self.x
        y=self.y
        alpha=self.alpha
        theta=self.theta
        tl=len(x)
        ws=np.ones((5,len(x[0])))
        start=0
        end=0
        y_cats=np.unique(y)
        
        for i in range(5):
            end=np.floor(tl*(i+1)/5)
            end=end.astype(int)
            if(end>tl):
                end=tl-1
            x1=x[start:end]
            y1=y[start:end]
            for j in range(len(x1)):
                f=0
                for k in range(len(x1[j])):
                    f=f+ws[i][k]*x1[j][k]
                if(f>theta and y[j]<np.max(y_cats)):
                    # Demotion
                    for l in range(len(x1[j])):
                        if(x1[j][l]>0):
                            ws[i][l]=ws[i][l]/alpha
                elif(f<theta and y[j]==np.max(y_cats)):
                    # Promotion
                    for l in range(len(x1[j])):
                        if(x1[j][l]>0):
                            ws[i][l]=ws[i][l]*alpha
            start=end+1
        finalws=np.average(ws,axis=0)
        self.finalws=finalws
        return finalws
    
    # Method test tests our fitted model
    # Since we utilized 5-fold cross validation , we will be returning the 
    #   results of one of the folds selected at random
    # The following variables are returned:
    #   ypred: The results of the classification performed on one of the folds
    #   accuracy: The accuracy of our model
    #   y1: The actual values of y-index of the dataframe
    
    def test(self):
        x=self.x
        y=self.y
        tl=len(x)
        alpha=self.alpha
        theta=self.theta
        finalws=self.finalws
        fold=random.randrange(0,5,1)
        end=np.floor(tl*(fold+1)/5)
        end=end.astype(int)
        start=(end-(np.floor(tl/5).astype(int)))
        start=start.astype(int)
        ypred=[]
        x1=x[start:end]
        y_cats=np.unique(y)
        y1=y[start:end]
        for i in range(len(x1)):
            f=0
            for j in range(len(x1[i])):
                f=finalws[j]*x1[i][j]
            if(f>theta):
                ypred.append(np.max(y_cats))
            else:
                ypred.append(np.min(y_cats))
        count=0
        for i in range(len(y1)):
            if(y1[i]!=ypred[i]):
                count=count+1
        
        accuracy=1-count/len(y1)
        accuracy=accuracy*100
        return ypred,accuracy,y1               
            
                
                    
                    