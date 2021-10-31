# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:44:20 2021

@author: CHRIS
"""
import pandas as pd
import numpy as np
import math

class D_N:
     
    def __init__(self,cid,h,w,sb=None,ss=None,sbl=None,fb=None,ft=None):
        self.cid=cid
        self.h=h
        self.w=w
        self.s_num_bottom=None
        self.s_cat_bottom=None
        self.s_num_shirt=None
        self.s_cat_shirt=None
        self.s_num_blazer=None
        self.s_cat_blazer=None
        self.fit_pants=None
        self.fit_shirt=None
        
        if(type(sb)!=str):
            if(sb==None):
                self.s_num_bottom=None
                self.s_cat_bottom=None
        elif(sb.isnumeric()):
            
            self.s_num_bottom=sb
            self.s_cat_bottom=None
        else:
            if(sb!=np.nan and sb!='No idea' and sb!='No Idea'):
                self.s_cat_bottom=sb
                self.s_num_bottom=None
            else:
                self.s_num_bottom=None
                self.s_cat_bottom=None
        
        if(type(ss)!=str):
            if(ss==None):
                self.s_num_shirt=None
                self.s_cat_shirt=None
        elif(ss.isnumeric()):
            
            self.s_num_shirt=ss
            self.s_cat_shirt=None
        else:
            if(ss!=np.nan and ss!='No idea' and ss!='No Idea'):
                self.s_cat_shirt=ss
                self.s_num_shirt=None
            else:
                self.s_num_shirt=None
                self.s_cat_shirt=None
        
        if(type(sbl)!=str):
            if(sbl==None):
                self.s_cat_blazer=None
                self.s_num_blazer=None
        elif(sbl.isnumeric()):
            
            self.s_num_blazer=sbl
            self.s_cat_blazer=None
        else:
            if(sbl!=np.nan and sbl!='No idea' and sbl!='No Idea'):
                self.s_cat_blazer=sbl
                self.s_num_blazer=None
            else:
                self.s_cat_blazer=None
                self.s_num_blazer=None

        
        if(fb!=None):
            if(str(fb)=='nan'):
                self.fit_pants=None
            else:
                self.fit_pants=fb

        if(ft!=None):
            if(str(ft)=='nan'):
                self.fit_shirt=None
            else:
                self.fit_shirt=ft
            
        self.sizes=None
        
        
        self.found_num_bottom=False
        self.found_cat_bottom=False
        self.found_num_shirt=False
        self.found_cat_shirt=False
        self.found_num_blazer=False
        self.found_cat_blazer=False
        
    def size(self,df,df_ar):
        df_1=df[df['user_id']==self.cid]
        sizes=df_ar.copy()
        sizes.iloc[:,5]=False
        if(len(df_1)>0):
            rem=len(df_1)
            for index,row in sizes.iterrows():
                df_2=df_1[df_1['type']==sizes.iloc[index,0]]
                if(len(df_2)>0):
                    df_2=df_2[df_2['brand']==sizes.iloc[index,1]]
                    if(len(df_2)>0):
                        df_2=df_2[df_2['category']==sizes.iloc[index,2]]
                        if(len(df_2)>0):
                            df_2=df_2[df_2['fit']==sizes.iloc[index,3]]
                            if(len(df_2)>0):
                                sizes.iloc[index,4]=df_2['size'].mode()[0]
                                sizes.iloc[index,5]=True
                                rem=rem-len(df_2)
                                if(rem==0):
                                    break
                            else:
                                sizes.iloc[index,4]=np.nan
                        else:
                            sizes.iloc[index,4]=np.nan
                    else:
                        sizes.iloc[index,4]=np.nan
                else:
                    sizes.iloc[index,4]=np.nan
        self.sizes=sizes
        return self
    
    def modeit(self):
        df_s=self.sizes.copy()
        df_s=df_s[df_s['Found']==True]
        if(len(df_s)>0):
        
            df_s_bottom=df_s[df_s['Type']=='Bottoms']
            
            df_s_num_bottom=df_s_bottom[df_s_bottom['Size_type']=='Numeric']
            ms=df_s_num_bottom['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric()):
                    self.s_num_bottom=ms.iloc[0]
                    self.found_num_bottom=True
                
            df_s_cat_bottom=df_s_bottom[df_s_bottom['Size_type']=='Categorical']
            ms=df_s_cat_bottom['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric()==False and str(ms.iloc[0])[0].isnumeric()==False):
                    self.s_cat_bottom=ms.iloc[0]
                    self.found_cat_bottom=True

    
            df_s_tops=df_s[df_s['Type']=='Tops']
            
            df_s_shirts=df_s_tops[df_s_tops['Category']!='Blazer']
            
            df_s_num_shirts=df_s_shirts[df_s_shirts['Size_type']=='Numeric']
            ms=df_s_num_shirts['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric() and int(ms.iloc[0])>35):
                    self.s_num_shirt=ms.iloc[0]
                    self.found_num_shirt=True
                
            df_s_cat_shirts=df_s_shirts[df_s_shirts['Size_type']=='Categorical']
            ms=df_s_cat_shirts['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric()==False and str(ms.iloc[0])[0].isnumeric()==False):
                    self.s_cat_shirt=ms.iloc[0]
                    self.found_cat_shirt=True
            
            df_s_blazer=df_s_tops[df_s_tops['Category']=='Blazer']
            
            df_s_num_blazer=df_s_blazer[df_s_blazer['Size_type']=='Numeric']
            ms=df_s_num_blazer['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric()):
                    self.s_num_blazer=ms.iloc[0]
                    self.found_num_blazer=True
                
            df_s_cat_blazer=df_s_blazer[df_s_blazer['Size_type']=='Categorical']
            ms=df_s_cat_blazer['Size'].mode()
            if(len(ms)>0):
                if(str(ms.iloc[0]).isnumeric()==False and str(ms.iloc[0])[0].isnumeric()==False):
                    self.s_cat_blazer=ms.iloc[0]
                    self.found_cat_blazer=True
        
        return self
    



    def copy(self):
        ndn=D_N(self.cid,self.h,self.w)
        ndn.sizes=self.sizes.copy()
        ndn.found_num_bottom=self.found_num_bottom
        ndn.found_cat_bottom=self.found_cat_bottom
        ndn.found_num_shirt=self.found_num_shirt
        ndn.found_cat_shirt=self.found_cat_shirt
        ndn.found_num_blazer=self.found_num_blazer
        ndn.found_cat_blazer=self.found_cat_blazer
        ndn.s_num_bottom=self.s_num_bottom
        ndn.s_cat_bottom=self.s_cat_bottom
        ndn.s_num_shirt=self.s_num_shirt
        ndn.s_cat_shirt=self.s_cat_shirt
        ndn.s_num_blazer=self.s_num_blazer
        ndn.s_cat_blazer=self.s_cat_blazer
        ndn.fit_pants=self.fit_pants
        ndn.fit_shirt=self.fit_shirt
        return ndn
        
        
        
            
                        
        
    
    
    