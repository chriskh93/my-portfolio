# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:43:21 2021

@author: CHRIS
"""

import flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import math
from joblib import load

app=flask.Flask(__name__)



#app.config["DEBUG"] = True


df_btm=pd.read_csv('df_btm_3.csv')
df_ar=pd.read_csv('df_ar_3.csv')


users=df_btm.iloc[:,0].unique().tolist()



def getSize(cid=None,dfb=pd.DataFrame()):
    sz=df_ar.copy()
    if(dfb.empty):
        dfb=df_btm[df_btm.iloc[:,0]==cid].reset_index(drop=True)
    if(cid==None):
        cid="None"
    bts=[]
    if(len(dfb)==1):
        if(dfb.iloc[0,1]=='nan'):
            dfb.iloc[0,1]=np.nan
        if(math.isnan(dfb.iloc[0,1])):
            for index,row in sz.iterrows():
                ud={}
                ud['user_id']=str(cid)
                ud['type']=sz.iloc[index,0]
                ud['brand']=sz.iloc[index,1]
                ud['category']=sz.iloc[index,2]
                ud['fit']=sz.iloc[index,3]


                if sz.iloc[index,0]=='Bottoms' and dfb.iloc[0,7]!=None:
                    if sz.iloc[index,-1]=='Numeric':
                        ud['size']=str(dfb.iloc[0,7])
                    else:
                        ud['size']=str(dfb.iloc[0,8])
                elif sz.iloc[index,0]=='Tops' and dfb.iloc[0,10]!=None:
                    if sz.iloc[index,-1]=='Numeric':
                        ud['size']=str(dfb.iloc[0,9])
                    else:
                        ud['size']=str(dfb.iloc[0,10])
                if('size' in ud):
                    bts.append(ud)
        else:
            sz.iloc[int(dfb.iloc[0,1]),5]=True
            sz.iloc[int(dfb.iloc[0,1]),4]=str(dfb.iloc[0,6])
            for index,row in sz.iterrows():
                ud={}
                ud['user_id']=str(cid)
                ud['type']=sz.iloc[index,0]
                ud['brand']=sz.iloc[index,1]
                ud['category']=sz.iloc[index,2]
                ud['fit']=sz.iloc[index,3]
                if(sz.iloc[index,5]==True):
                    ud['size']=str(sz.iloc[index,4])
                else:
                    if sz.iloc[index,0]=='Bottoms' and dfb.iloc[0,7]!=None:
                        if sz.iloc[index,-1]=='Numeric':
                            ud['size']=str(dfb.iloc[0,7])
                        else:
                            ud['size']=str(dfb.iloc[0,8])
                    elif sz.iloc[index,0]=='Tops' and dfb.iloc[0,10]!=None:
                        if sz.iloc[index,-1]=='Numeric':
                            ud['size']=str(dfb.iloc[0,9])
                        else:
                            ud['size']=str(dfb.iloc[0,10])
                bts.append(ud)
    else:
        for index,row in dfb.iterrows():
            sz.iloc[int(dfb.iloc[index,1]),5]=True
            sz.iloc[int(dfb.iloc[index,1]),4]=str(dfb.iloc[index,6])
        for index,row in sz.iterrows():
                ud={}
                ud['user_id']=str(cid)
                ud['type']=sz.iloc[index,0]
                ud['brand']=sz.iloc[index,1]
                ud['category']=sz.iloc[index,2]
                ud['fit']=sz.iloc[index,3]
                if(sz.iloc[index,5]==True):
                    ud['size']=str(sz.iloc[index,4])
                else:
                    if sz.iloc[index,0]=='Bottoms' and dfb.iloc[0,7]!=None:
                        if sz.iloc[index,-1]=='Numeric':
                            ud['size']=str(dfb.iloc[0,7])
                        else:
                            ud['size']=str(dfb.iloc[0,8])
                    elif sz.iloc[index,0]=='Tops' and dfb.iloc[0,10]!=None:
                        if sz.iloc[index,-1]=='Numeric':
                            ud['size']=str(dfb.iloc[0,9])
                        else:
                            ud['size']=str(dfb.iloc[0,10])

                bts.append(ud)
    return bts



@app.route('/', methods=['GET'])
def home():
    return "<h1>Mr. Draper Size Recommender</h1><p>This site is a prototype API for a Size Recommender for Mr.Draper</p>"



@app.route('/sizes', methods=['POST'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.json:
        cid = int(request.json['id'])
        bts=getSize(cid=cid)
    elif ('weight' in request.json and 'height' in request.json):
        weight=int(request.json['weight'])
        height=int(request.json['height'])
        if ('size_b' not in request.json and 'size_t' not in request.json):
            return "Error: Not enough inputs provided."
        elif('size_b' in request.json and 'size_t' not in request.json):
            size_b=int(request.json['size_b'])
            size_t=None
        elif('size_t' in request.json and 'size_b' not in request.json):
            size_t=str(request.json['size_t'])
            size_b=None
        else:
            size_b=int(request.json['size_b'])
            size_t=str(request.json['size_t'])
        clf_b = load('clf_bc_bn.joblib')
        r_ss_c=load('reg_OLS_st_ct.joblib')
        if(size_b!=None):
            X1=[]
            X1.append(size_b)
            X1.append(weight)
            X1.append(height)
            X1=np.array(X1)
            X1=X1.reshape([1,-1])
            sbc=clf_b.predict(X1)[0]
        else:
            sbc=None


        if(size_t!=None):
            scats1=['XS','S','M','L','XL','XXL','XXXL','XXXXL']
            X1=[]
            X1.append(scats1.index(size_t.upper())+1)
            X1.append(weight)
            X1.append(height)
            X1.append(1)
            X1=np.array(X1)
            X1=X1.reshape([1,-1])
            ssn=np.round(r_ss_c.predict(X1)[0],0)
        else:
            ssn=None


        dfb=pd.DataFrame(data=np.array([ float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"),size_b,sbc,ssn,size_t]).reshape([1,-1]))
        bts=getSize(dfb=dfb)
    else:

        return "Error: Not enough inputs provided."


    typ=''
    brand=''
    fit=''
    category=''
    if 'type' in request.json:
        typ=str(request.json['type'])
    if 'brand' in request.json:
        brand=str(request.json['brand'])

    if 'category' in request.json:
        category=str(request.json['category'])

    if 'fit' in request.json:
        fit=str(request.json['fit'])
    results1 = bts.copy()

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
#    for bottom in bottoms:
#        if bottom['user_id'] == id:
#            results1.append(bottom)

    results2=[]
    if(typ==''):
        results2=results1.copy()
    else:
        for r in results1:
            if(r['type']==typ):
                results2.append(r)
    results1=results2.copy()
    results2=[]
    if(brand==''):
        results2=results1.copy()
    else:
        for r in results1:
            if(r['brand']==brand):
                results2.append(r)
    results1=results2.copy()
    results2=[]
    if(category==''):
        results2=results1.copy()
    else:
        for r in results1:
            if(r['category']==category):
                results2.append(r)
    results1=results2.copy()
    results2=[]
    if(fit==''):
        results2=results1.copy()
    else:
        for r in results1:
            if(r['fit']==fit):
                results2.append(r)



    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results2)

if __name__=='__main__':
        app.run()
