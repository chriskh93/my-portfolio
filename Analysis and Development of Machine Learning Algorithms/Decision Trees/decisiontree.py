# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:39:08 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necesarry libraries:

import pandas as pd
import numpy as np
import Node
import math

# class decisiontree implements both the ID3 and CART modeling algorithms

class decisiontree:
    
    # Constructor takes the follwing parameters:
    #   df: The pandas dataframe that requires modeling
    #   xinds: The indices of the x-variables within df
    #   yinds: The index of the y-variable within df (The target)
    #   reg: Boolean variable for whether or not this is a regression model
    #   norm: Boolean variable for whether or not the target variable is normalized
    #   means: The mean of the target variable
    #   stds: The standard deviation of the target variable
    
    # The constructor further initializes the following variables:
    #   y_cats: The different possible values of our y-variable (target)
    #   tree: The decision tree

    def __init__(self,df, xinds, yinds,reg=False,norm=False,means=0,stds=0):
        self.df=df
        self.xinds=xinds
        self.yind=yinds
        self.norm=norm
        if(reg==False):
            self.ycats=df.iloc[:,self.yind].unique()
        else:
            if(norm):
                self.means=means
                self.stds=stds


        self.tree=None



    # Method I calculates the Entropy of a dataset given a target variable
    #   y having classes y_cats
    # The method take the following input parameters:
    #   df: The dataset
    #   yind: The index of the target variable within df
    # The method returns the calculated value of I

    
    def I(self,df,yind=None):
        
        if(yind==None):
            yind=self.yind
        
        ycats=self.ycats
        I=0
        for i in range(len(ycats)):
            cal=len(df[df.iloc[:,yind]==ycats[i]])/len(df)
            cal1=cal*np.log2(cal)
            
            # The method replaces the logarithmic calculated values 
            #   of nan with 0:
            
            if(math.isnan(cal1)):
                I=I-0
            else:
                I=I-cal1
                
        return I
  
    
    # Method E calculates the Expected Entropy of a feature xind 
    #   in dataset df
    # The method take the following input parameters:
    #   df: The dataset
    #   xind: The index of the feature x within df
    # The method returns the calculated value of E
    
    def E(self,df,xind):
        
        xcats=df.iloc[:,xind].unique()
        E=0
        for i in range(len(xcats)):
            df_1=df[df.iloc[:,xind]==xcats[i]].copy()
            cal=len(df_1)/len(df)
            E=E+cal*self.I(df_1)
        return E
    
    
    
    # Method IV calculates the Information Value of a feature xind 
    #   in dataset df
    # The method take the following input parameters:
    #   df: The dataset
    #   xind: The index of the feature x within df
    # The method returns the calculated value of IV
    
    def IV(self,df,xind):
        xcats=df.iloc[:,xind].unique()
        IV=0
        for i in range(len(xcats)):
            df_1=df[df.iloc[:,xind]==xcats[i]].copy()
            cal=len(df_1)/len(df)
            IV=IV-cal*np.log2(cal)
        return IV
    
    
    # Method buildID3 builds a decision tree using the ID3 algorithm
    # The method take the following input parameters:
    #   df: The dataset upon which to build the tree
    #   xinds: The indices of the feature values
    #   yind: The index of target variable
    #   parent: The parent of the current tree node that is being constructed 
    #       Initially None

    def buildID3 (self,df=pd.DataFrame(),xinds=None,yind=None,parent=None):
        
        # Initializing the input parameters:
        
        if(df.empty):
            df=self.df
        if(xinds==None):
            xinds=self.xinds.copy()
        
        # Making copies of xinds so as not to overwrite and lose data:
        
        else:
            xinds=xinds.copy()
            
        # Initializing yind to the stored private variable:
        
        if(yind==None):
            yind=self.yind
        root=parent
        
        # Initializing the gain ratio array
        
        gainrs=[]
        
        # Calculating the Entropy of the current dataset:
        
        I=self.I(df,yind)

        # I!=0 indicates that there is more than 1 class in the dataset
        # len(xinds)>0 indicates that there are feature attributes to split by:
        
        if((I!=0) and len(xinds)>0):
            
            # For each feature attribute, the gain ratio is calculated:
            
            for i in range(len(xinds)):
                g=I
                g=g-self.E(df,xinds[i])
                gr=g/self.IV(df,xinds[i])
                
                # nan is replaced with 0 for calculation purposes:
                
                if(math.isnan(gr)):
                    gr=0
                gainrs.append(gr)
            
            # The maximum gain ratio is determined:
            
            maxgainr=np.max(gainrs)
            
            # The feature corresponding to the maximum gain ratio is determined:
            
            rt=-1
            for i in range(len(gainrs)):
                if(gainrs[i]==maxgainr):
                    rt=i
                    break
            
            # Initializing the data variable to create our tree node:
            
            data=[]
            indrt=xinds[rt]
            
            # A maxgainr of 0 indicates no added value of splitting by a certain
            #   feature x
            
            if(maxgainr>0):
                
                # Setting up the Node variables:
                
                data.append(df.columns[xinds[rt]])
                x_cats=df.iloc[:,xinds[rt]].unique()
                for i in range(len(x_cats)):
                    data.append(x_cats[i])
                        
                if("<" in data[0]):
                    typ="Numeric"
                    cond="<"
                else:
                    typ="Categorical"
                    cond="CC"
                    
                declf="Decision"
                    
                # Removing this feature from our list of feature attributes:
                
                xinds.remove(indrt)
            
            # If maxgainr==0 but I!=1, that means there is more of 1 class than 
            #   another, therefore we can take the most occuring class (mode) as 
            #   a leaf:
            
            elif I!=1:
                y=df.iloc[:,yind].mode()
                data=y[0]
                typ="Class"
                cond=None    
                declf="Leaf"
            
            # If maxgainr==0 and I==1, that means that there are an equal number
            #   of classes and therefore unsplittable, the method returns None
            
            else:
                return

            
                
        
        else:
            
            # A value of I==0 indicates that there is only 1 class within the
            #   current dataset, therefore a leaf of that class can be formed:
            
            if(I==0):
                y=df.iloc[:,yind].unique()
                data=y[0]
                typ="Class"
                cond=None    
                declf="Leaf"
            
            # This condition indicates that there are no more feature attributes 
            #   to split by, therefore if I!=1, this mean that there is an 
            #   unequal number of classes within the current dataset and a leaf
            #   can be generated with the mode:
            
            elif (I!=1):
                y=df.iloc[:,yind].mode()
                data=y[0]
                typ="Class"
                cond=None    
                declf="Leaf"
            
            # This condition indicates that there are no more feature attributes 
            #   to split by and I==1 which indicates that there are an equal
            #   number of classes in the current dataset which makes it
            #   unsplittable, and therefore returns None:
            
            else:

                return
            
                
                    
                    
        # If the parent of the current Node is = None, therefore this must be the
        #   root of the tree, the root is therefore established:
        if(parent==None):
            root=Node.Node(data,cond,typ,declf)
            
            # The tree begins construction with the root node being a decision:
            
            if(declf=="Decision"):
                
                # Looping through the different feature values of the splitting 
                #   feature:
                
                for i in range(len(x_cats)):
                    
                    # Splitting the dataset accordingly:
                    
                    df_1=df[df.iloc[:,indrt]==x_cats[i]]
                    
                    # Recursively running the ID3 algorithm on the splitted
                    #   datset, the modified feature list, and root of the tree
                    #   established:
                    
                    self.buildID3(df=df_1,xinds=xinds,parent=root)
            
            # dic is defined for debugging and visualization purposes:
            
            dic=root
        
        # If the parent of the current Node is != None, therefore this a child 
        #   Node:
        
        else:
            
            if(declf=="Decision"):
                
                # The new node is inserted into the child of the parent:
                
                nn=parent.insert(data,cond,typ,declf,parent)
                for i in range(len(x_cats)):
                    # Splitting the dataset accordingly:
                    
                    df_1=df[df.iloc[:,indrt]==x_cats[i]]
                    
                    # Recursively running the ID3 algorithm on the splitted
                    #   datset, the modified feature list, and the new parent
                    
                    self.buildID3(df=df_1,xinds=xinds,parent=nn)
                    
                
                # Ensuring that no parent node has only 1 child:
                
                if(i==1 and len(nn.children)==1):
                    
                        # If a parent node has only 1 child, then the child
                        #   replaces the parent:
                        
                        newn=nn.children[0]
                        newch=nn.children[0].children
                        newp=parent
                        nn=newn
                        nn.children=newch
                        nn.parent=newp                        
                        nn.parent.children[len(nn.parent.children)-1]=nn
                
                
                # Ensuring that no parent node has only identical leaves:
                
                if(len(nn.parent.children)>1):
                    
                    # Determining if the children are all leaves:
                    
                    declfs=[]
                    for i in range(len(nn.parent.children)):
                        declfs.append(nn.parent.children[i].declf)
                    leafs=True
                    for i in range(len(declfs)):
                        if(declfs[i]!="Leaf"):
                            leafs=False
                            break
                    if(leafs):
                        
                        # Ensuring that not all the leaves are the same:
                        
                        same=True
                        comp=nn.parent.children[0].data
                        for i in range(len(nn.parent.children)):
                            if(same==False):
                                break
                            for j in range(len(nn.parent.children)):
                                if(i!=j):
                                    if(nn.parent.children[i].data!=nn.parent.children[j]):
                                        same=False
                                        break
                                    
                        # If the leaves are all the same, then the parent 
                        #   becomes the leaf:
                        
                        if(same):
                            nn.parent.data=comp
                            nn.parent.typ="Class"
                            nn.parent.cond=None    
                            nn.parent.declf="Leaf"
                            nn.parent.children=[]
                            
                    
                        

                # dic defined for debugging and visualization purposes:
                
                dic=nn
                
                    
            else:
                
                # This condition is if the current tree Node is a leaf
                # Ensuring that the parent does not have only identical leaves
                #   as children:
                
                if(len(parent.children)>1):
                    same=True
                    for i in range(len(parent.children)):
                        if(parent.children[i].data!=data):
                            same=False
                            break
                    if(same):
                        
                        # If the leaves are indentical, then the parent becomes
                        #   the leaf:
                        
                        parent.data=data
                        parent.typ="Class"
                        parent.cond=None    
                        parent.declf="Leaf"
                        parent.children=[]
                        dic=parent
                    
                    else:
                        
                        # Else, insert the leaf into the child of the current
                        #   parent:
                        nn=parent.insert(data,cond,typ,declf,parent)
                        dic=nn

                else:
                    
                    # If the parent only has less than one child, 
                    #   then insert this Node:
                    
                    nn=parent.insert(data,cond,typ,declf,parent)

                    dic=nn
                        
        
        # treedict is defined using previously defined dic variable for 
        #   debugging and visualization purposes:
        
        treedict=self.getTreeDict(dic,self.getTreeDict(dic.parent))
        
        # Set the value of our tree to root:
        
        self.tree=root

    
    
    
    # Method ID3fit does the following:
    #   Extracts 10% of our dataframe as a validation set
    #   Performs 5-fold cross validation on the remaining 90%
    #   Takes 4 of the 5 folds as training data and builds an ID3 Tree upon it
    #   Predicts the values of the remaining fold as the test set and records
    #       the error
    #   Predicts the values of the validation set using the constructed tree of
    #       the training set
    #   Performs post-pruning by pruning the constructed tree and measuring
    #       the error on the validation set
    #   Determines the optimum post-pruned tree by finding the post-pruned tree
    #       that results in the lowest validation set error
    # The method take the following input parameters:
    #   pruning: A boolean variable for whether or not to perform post-pruning
    #
    # The method returns the following:
    #   errorsnp: The fold classification errros without pruning
    #   errorsp: The fold classification errors with pruning
    #   np.mean(errorsnp): The mean of the unpruned errors
    #   np.mean(errorsp): The mean of the pruned errors
    #   treesnp: The generated unpruned trees
    #   treesp: The generated pruned trees
    # Furthermore, the method prints the pruned and unpruned results in a user
    #   friendly way
    
    
    def ID3fit(self,pruning=False):
        
        # Initializing the required variables:
        
        df=self.df
        yind=self.yind
        
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
        
        # errorsnp: The errors of the unpruned trees
        # errorsp: The errors of the pruned trees
        # treesnp: The generated unpruned trees
        # treesp: The generated pruned trees
        errorsnp=[]
        errorsp=[]
        treesnp=[]
        treesp=[]
        for i in range(len(dfs)):
            xinds=self.xinds.copy()
            df_train=dfs[i][0]
            df_test=dfs[i][1]
            treec=decisiontree(df_train,xinds,yind)
            treec.buildID3()
            
            # tree is the Node variable of the unpruned tree
            
            tree=treec.getTree()
            treesnp.append(tree.copy())
            
            # treedict is a dictionary created in order to visualize the tree
            #   on the variable explorer pane
            
            treedict=treec.getTreeDict()
            
            # ypred is the predicted y-values of the test set
            
            ypred=treec.predict(df=df_test)
            
            # er is the classification error of our unpruned tree
            
            er=treec.ID3Error(ypred,df=df_test)
            
            # treesize is the number of vertices (branches and leaves) in our
            #   tree
            treesize=self.getTreeSize(tree)
            errorsnp.append(er)
            
            # ypredv is the predicted y-values of the validation set
            
            ypredv=treec.predict(df=df_v)
            
            # erv is the classification error of our validation set:
            
            erv=treec.ID3Error(ypredv,df=df_v)
            
            # Printing the unpruned results in a user friendly way:
            
            print("ID3 Decision Tree Model for Fold "+str(i+1)+":")
            print("Pruning: No")
            print("Tree Size: "+str(treesize))
            df_res=pd.DataFrame({'Actual':df_test.iloc[:,yind],'Predicted':ypred})
            print(df_res.to_string())
            print("Error: "+str(np.round(er,2)))
            print("Validation Set Error: "+str(np.round(erv,2)))
            print()
            
            if(pruning):
                
                # treel is the depth of our unpruned tree
                
                treel=treec.getTreeDepth()
                
                # Copying our tree in order not to lose data, and setting it
                #   up for pruning:
                
                og=tree.copy().setPrune()
                
                # Running the pruning algorithm:
                
                treep=self.prune(treel,df_train,df_v,erv,og,df_train,treel,0,og)
                
                # treep is the Node variable of our pruned tree
                # treepd is the dictionary variable of treep for visualization
                #   purposes
                
                treepd=treec.getTreeDict(root=treep)
                
                # treesizen is the size of our pruned tree
                
                treesizen=self.getTreeSize(treep)
                
                # yp1 is the predicted validation set values using the pruned
                #   tree
                
                yp1=treec.predict(df=df_v,tree=treepd)
                
                # er1 is the classification error of the validation set using
                #   the pruned tree
                
                er1=treec.ID3Error(yp1,df=df_v)
                
                # treepl is the depth of the pruned tree
                
                treepl=treec.getTreeDepth(treepd)
                treesp.append(treep.copy())
                
                # ypredp is the predicted values of the test set using the pruned
                #   tree
                
                ypredp=treec.predict(df=df_test,tree=treepd)
                
                # erp is the classification error of the test set using the 
                #   pruned tree:
                
                erp=treec.ID3Error(ypredp,df=df_test)
                errorsp.append(erp)
                
                # Printing the results in a user friendly way:
                
                print("ID3 Decision Tree Model for Fold "+str(i+1)+":")
                print("Pruning: Yes")
                print("Tree Size: "+str(treesizen))
                df_res=pd.DataFrame({'Actual':df_test.iloc[:,yind],'Predicted':ypredp})
                print(df_res.to_string())
                print("Error: "+str(np.round(erp,2)))
                print("Validation Set Error: "+str(np.round(er1,2)))
                print()
                
            
        return errorsnp,errorsp,np.mean(errorsnp),np.mean(errorsp),treesnp,treesp
           
            
    


    # Method prune does the following:
    #   The algorithm begins by traversing all the way to the bottom of the tree
    #   Then starts evaluating if a vertex can be pruned by evaluating the 
    #       classification error on the validation set
    #   The optimum post-pruned tree is determined by finding the post-pruned 
    #       tree that results in the lowest validation set error
    #       
    # The method take the following input parameters:
    #   level: The tree level to evaluate (initialized as the depth of the tree)
    #   df_t: The current dataset (initialzed to the training set)
    #   df_v: The validation set
    #   er: The validation set error to compare with
    #   og: The tree to compare with
    #   df_tog: The original training dataset
    #   ogl: The original depth of the tree
    #   yind: The index of the target variable
    #   root: The current Node of the tree
    #
    # The method returns the following:
    #   og: The optimum post-pruned tree
    
    
    def prune(self,level,df_t,df_v,er,og,df_tog,ogl,yind=None,root=None):
        ogd=self.getTreeDict(og)
        if(root==None):
            root=self.tree.copy()
        if(yind==None):
            yind=self.yind
        if(level==1):
            
            # if level==1 , this indicates to prune this level and this node of
            #   of the tree
            # erv: classification error of the pruned tree on the validation set
            # nt: The pruned tree
            
            erv,nt=self.validate(df_v,root)
            
            
            if(erv<=er):
                
                # if the new classification error is less than or equal to our
                #   previous error er
                # ap (after pruning) and bp (before pruning) tree dictionaries
                #   before and after pruning
                # They are defined for visualization and debugging purposes
                
                ap=self.getTreeDict(nt)
                bp=self.getTreeDict(og)
                
                # Recursively run this algorithm with the new error and new tree
                #   as the tree and error to compare with
                
                nwt=self.prune(self.getTreeDepth(root=self.getTreeDict(nt)),df_tog,df_v,erv,nt.copy(),df_tog,self.getTreeDepth(root=self.getTreeDict(nt)),yind,nt)
                
                # nwt: New tree after re-running pruning algorithm
                # sn: Size of new tree
                # so: Size of current tree
                
                sn=self.getTreeSize(nwt)
                so=self.getTreeSize(og)
                if(sn<so):
                    
                    # if the tree got smaller, i.e. pruned, replace current tree
                    #   with pruned tree
                    
                    og=nwt
                return og
            else:
                
                # if the error is not less then check if other vertices can be
                #   pruned
                
                if(ogl>2):
                    
                    # If the initial tree level is greater than 2, i.e. there
                    #   is a decision node:
                    # Maintain our unpruned tree but search for vertices in
                    #   higher tree levels
                    
                    nwt=self.prune(ogl-1,df_tog,df_v,er,og,df_tog,ogl-1,yind,og.copy())
                    sn=self.getTreeSize(nwt)
                    so=self.getTreeSize(og)
                    if(sn<so):
                        
                        # if the tree got smaller, i.e. pruned, replace current tree
                        #   with pruned tree
                        og=nwt
                    return og
                else:
                    return og
         
        else:
            if(level==2 and len(root.children)>0):
                
                # level==2 and len(root.children)>0 refers to the parent of a
                #   potentially pruned branch 
                if(root.checked==False):
                    
                    # if(root.checked==False) ensures that this vertex hasnt been
                    #   previously checked
                    
                    # og.setCheck(root,True) sets the checked value of this
                    #   vertex in og to True
                    og.setCheck(root,True)
                    
                    # Replace the parent node with the mode:
                    
                    y=df_t.iloc[:,yind].mode()
                    data=y[0]
                    typ="Class"
                    cond=None    
                    declf="Leaf"
                    root1=Node.Node(data,cond,typ,declf)                    
                    ogd=self.getTreeDict(og)
                    og05=self.replaceNode(og,root,root1)
                    if(og05!=None):
                        og1d=self.getTreeDict(root=og05)
                        
                        # nwt: New tree after re-running pruning algorithm with
                        #   decremented level
                        # sn: Size of new tree
                        # so: Size of current tree
                        
                        nwt=self.prune(level-1,df_t,df_v,er,og,df_tog,ogl,yind,root1)
                        sn=self.getTreeSize(nwt)
                        so=self.getTreeSize(og)
                        if(sn<so):
                            
                            # if the tree got smaller, i.e. pruned, replace current tree
                            #   with pruned tree
                            og=nwt
                    return og
                else:
                    return og
                    
                
            elif(len(root.children)>0 and root.declf=="Decision"):
                
                # if the current level is not at the defined pruning level then
                #   traverse the tree
                
                for i in range(len(root.children)):
                    
                    # Split the dataset accordingly:
                    
                    df_n=df_t[df_t[root.data[0]]==root.data[1+i]].copy()
                    
                    # Recursively run the algorithm with level, dataset, and root
                    #   updated accordingly
                    
                    # nwt: New tree after re-running pruning algorithm
                    # sn: Size of new tree
                    # so: Size of current tree
                    
                    nwt=self.prune(level-1,df_n,df_v,er,og,df_tog,ogl,yind,root=root.children[i])
                    
                    
                    sn=self.getTreeSize(nwt)
                    so=self.getTreeSize(og)
                    if(sn<so):
                        
                        # if the tree got smaller, i.e. pruned, replace current tree
                        #   with pruned tree
                        
                        og=nwt
                return og
            else:
                return og
            
                    
        
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
                
            
            
    # Method getTreeDict converts a tree from a Node object into a Dictionary
    # This is useful for visualization and debugging purposes
    # Each Decision Node has the following keys:
    #   Value: The value of the Node
    #   Decision/Leaf: Whether the Node is a decision or a leaf
    #   Type: The type of the Node
    #   Options: If the Node is a Decision Node, then the discrete attribute 
    #       values are listed
    #   Children: An array containing dictionaries of the children of the Node
    #   Parent: The dictionary of the parent of the Node
    #   Checked: The "checked" value of the Node, used for pruning purposes
    #   IDN: The Node unique identifier
    #
    # The method take the following input parameters:
    #   root: The root Node of the tree to convert
    #   parent: The parent of the Node
    # The method returns:
    #   showTree: The tree dictionary    
        
        
    def getTreeDict(self,root=None,parent=None):
        if(root==None):
            root=self.tree
        if(parent==None and root.parent==None):
                            
            if(root.declf=="Leaf"):
                showTree={"Value":root.data,
                      "Decision/Leaf":root.declf,"Type":root.typ}
            else:    
                showTree={"Value":root.data[0],"Options":root.data[1:],
                      "Decision/Leaf":root.declf,"Type":root.typ}
                childs=[]
                if(len(root.children)!=0):
                    for i in range(len(root.children)):
                        child=self.getTreeDict(root=root.children[i],parent=showTree)
                        childs.append(child)
                if(len(childs)==0):
                    childs=None
                showTree["Children"]=childs
            
        else:
            
            if(parent==None):
                par=self.getTreeDict(root.parent)
                parent=par
            if(root.declf=="Leaf"):
                showTree={"Value":root.data,
                      "Decision/Leaf":root.declf,"Type":root.typ}
                showTree['Parent']=parent
            else:    
                showTree={"Value":root.data[0],"Options":root.data[1:],
                      "Decision/Leaf":root.declf,"Type":root.typ}
                childs=[]
                showTree['Parent']=parent
                if(len(root.children)!=0):
                    for i in range(len(root.children)):
                        child=self.getTreeDict(root=root.children[i],parent=showTree)
                        childs.append(child)
                if(len(childs)==0):
                    childs=None
                showTree["Children"]=childs
            
        showTree["Checked"]=root.checked
        showTree["IDN"]=root.idn
        return showTree
    
    
    
    # Method getTreeDepth returns the depth of a TreeDict root
    # The method take the following input parameters:
    #   root: The tree dictionary of the root Node of the tree
    # The method returns:
    #   l: The tree depth 
    
    
    def getTreeDepth(self,root=None):
        if(root==None):
            root=self.getTreeDict()
        l=0
        
        if("Value" in root):
            l+=1
            if("Children" in root):
                ls=[]
                for i in range(len(root['Children'])):
                    ls.append(self.getTreeDepth(root=root['Children'][i]))
                l+=np.max(ls)
        return l
    
    
    # Method predict predicts the target variable values of a dataframe "df" using
    #   a decision tree dictionary "tree"
    # The method take the following input parameters:
    #   df: The dataframe who's values to predict
    #   tree: The decision tree dictionary
    # The method returns:
    #   ypred: The predicted target values 
    
    def predict(self,df=pd.DataFrame(),tree=None):
        
        if(tree==None):
            tree=self.getTreeDict()
        if(df.empty):
            df=self.df
        
        ypred=[]
        
        for index,row in df.iterrows():
            trav=tree
            while(trav['Decision/Leaf']=="Decision"):
                dec=row[trav['Value']]
                for i in range(len(trav['Options'])):
                    if(dec==trav['Options'][i]):
                        break
                trav=trav['Children'][i]
            ypred.append(trav['Value'])
        
        return ypred
    
    
    
    # Method validate predicts the target variable values of a dataframe "df" 
    #   using the tree corresponding to Node "root" and returns the error "er"
    #   and corresponding decision tree "trr"
    # The method take the following input parameters:
    #   df: The dataframe who's values to predict
    #   root: A Node within the tree used to predict the values
    # The method returns:
    #   er: The classification error associated with the prediction
    #   trr: The tree used for the prediction
    
    
    def validate(self,df,root):
        trr=root.root()
        yp1=self.predict(df,self.getTreeDict(trr))
        er=self.ID3Error(yp1,df)
        return er,trr
        
    


    # Method ID3Error returns the classification error of a prediction
    # The method take the following input parameters:
    #   ypred: The predicted values
    #   df: The dataframe who's target values were predicted
    #   yind: The index of the target variable in "df"
    # The method returns:
    #   error: The classification error associated with the prediction

    
    def ID3Error(self,ypred,df=pd.DataFrame(),yind=None):
        
        if(df.empty):
            df=self.df
        
        if(yind==None):
            yind=self.yind
            
        count=0
        for index,row in df.iterrows():
            if(df.iloc[index,yind]!=ypred[index]):
                count+=1
        
        error=count*100/len(df)
        return error
    
    
    
    # Method getSubTree returns the subtree "value" within the tree "root"
    # The method take the following input parameters:
    #   value: A Node variable representing the subtree to find
    #   root: A Node variable representing the tree to look in
    # The method returns:
    #   rt: The SubTree "value" in tree "root"
    #   None: If the subtree is not found
        
    def getSubTree(self,value,root=None):
        
        if(root==None):
            root=self.tree
        
        
        if(root.idn==value.idn):
            return root
        else:
            if(len(root.children)>0):
                for i in range(len(root.children)):
                    rt=self.getSubTree(value,root=root.children[i])
                    if(rt!=None):
                        return rt
                return None
            else:
                return None
    



    # Method getTree returns the tree associated with this class
       
    def getTree(self):
        return self.tree
    
    
    
    # Method replaceNode replaces Node "rootold" with "rootnew" in tree "og"
    # The method take the following input parameters:
    #   og: A Node variable representing the tree
    #   rootold: A Node variable representing the Node to replace
    #   rootnew: A Node variable representing the Node to replace with
    # The method returns:
    #   og1: A Node variable representing the tree with the Node replaced

    def replaceNode(self,og,rootold,rootnew):
        og1=og.copy()
        og1d=self.getTreeDict(og1)
        vd=self.getTreeDict(rootold)
        
        e=self.getSubTree(rootold,og1)

        if(e!=None):
            if(e.parent!=None):
                p1=e.parent
                for i in range(len(p1.children)):
                    if(p1.children[i].idn==rootold.idn):
                        break
                p1.children[i]=rootnew
                rootnew.parent=p1
                og1=rootnew.root()
                declfs=[]
                leaves=True
                for i in range(len(p1.children)):
                    declfs.append(p1.children[i].declf)
                for i in range(len(declfs)):
                    if declfs[i]!="Leaf":
                        leaves=False
                        break
                if(leaves):
                    if(len(p1.children)>1):
                        same=True
    
                        for i in range(len(p1.children)):
                            if(same==False):
                                break
                            for j in range(len(p1.children)):
                                if(j!=i):
                                    if(p1.children[i].data!=p1.children[j].data):
                                        same=False
                                        break
                        if(same):
                            og1=self.replaceNode(og1,p1,rootnew)
                        
                    
                    

                return og1
            else:
                og1=rootnew.root()
                return og1

    
    
    # Method getTreeSize returns the number of branches and leaves in tree "og"
    # The method take the following input parameters:
    #   og: A Node variable representing the tree
    # The method returns:
    #   count: The size of the tree
    
    def getTreeSize(self,og):
        count=1
        if(len(og.children)>0):
            for i in range(len(og.children)):
                count=count+self.getTreeSize(og.children[i])
        return count
     
        
    
    # Method gmj calculates gmj value required for CART regression modeling
    # The method take the following input parameters:
    #   df: The dataset
    #   med: A boolean variable indicating whether or not to calculate gmj using
    #       the mean or median
    # The method returns:
    #   g:
    #       If med==True: then g is the median of the y values in "df"
    #       If med==False: then g is the mean of the y values in "df"    
    
    
    def gmj(self,df,med=False):
        if(med==False):
            g=np.sum(df.iloc[:,self.yind])/len(df)
        else:
            g=np.median(df.iloc[:,self.yind])
        return g
    
    
    
    # Method EM calculates the MSE of splitting by feature "xind" in dataset "df"
    # The method take the following input parameters:
    #   df: The dataset
    #   xind: The index of the feature within df
    #   med: A boolean variable indicating whether or not to calculate gmj using
    #       the mean or median
    # The method returns:
    # em: The MSE
    
    
    def EM(self,df,xind,med=False):
        nm=len(df)
        em=0
        xcats=df.iloc[:,xind].unique()
        for i in range(len(xcats)):
            df_1=df[df.iloc[:,xind]==xcats[i]]
            ein=0
            for index,row in df_1.iterrows():
                g=self.gmj(df_1,med)
                calc=row[self.yind]-g
                calc=calc**2
                ein=ein+calc
            em=em+ein
        
        em=em*1/nm
        
        return em
                
                
            
        
    # Method buildCART builds a decision tree using the CART algorithm
    # The method take the following input parameters:
    #   df: The dataset upon which to build the tree
    #   xinds: The indices of the feature values
    #   yind: The index of target variable
    #   parent: The parent of the current tree node that is being constructed 
    #       Initially None
    #   es: Early Stopping threshold
    #   med: A boolean variable indicating whether or not to calculate gmj using
    #       the mean or median       
            
    def buildCART (self,df=pd.DataFrame(),xinds=None,yind=None,parent=None,es=0,med=False):
            
        # Initializing the input parameters:
            if(df.empty):
                df=self.df
            if(xinds==None):
                xinds=self.xinds.copy()
                
            # Making copies of xinds so as not to overwrite and lose data:
            
            else:
                xinds=xinds.copy()
            
            # Initializing yind to the stored private variable:
            
            if(yind==None):
                yind=self.yind
            root=parent
            
            # len(xinds)>0 indicates that there are features to check if we can
            #   split by:
            
            if(len(xinds)>0):
                
                # Initializing the mses array
                
                mses=[]
                
                # For each feature attribute, the MSE of splitting by that
                #   feature is calculated
                
                for i in range(len(xinds)):
                    mse=self.EM(df,xinds[i],med)
                    mses.append(mse)
                
                # The minimum calculated MSE is determined
                
                minmse=np.min(mses)
                
                # The feature corresponding to that minium MSE is determined:
                
                for rt in range(len(mses)):
                    if(mses[rt]==minmse):
                        break
                
                data=[]
                indrt=xinds[rt]
                x_cats=df.iloc[:,xinds[rt]].unique()
                
                # Ensuring that the selected feature has more than 1 unique value
                #   and that the minimum MSE (minmse) > the Early Stopping
                #   threshold
                
                if(len(x_cats)>1 and minmse>es):
                    
                    # Setting up the tree Node:
                    
                    data.append(df.columns[xinds[rt]])
    
                    for i in range(len(x_cats)):
                        data.append(x_cats[i])
                            
                    if("<" in data[0]):
                        typ="Numeric"
                        cond="<"
                    else:
                        typ="Categorical"
                        cond="CC"
                        
                    declf="Decision"
                        
                    # Updating the list of features:
                    
                    xinds.remove(indrt)
                else:
                    
                    # Do not split, return a leaf based either by the mean or 
                    #   median
                    
                    if(med==False):
                        y=df.iloc[:,yind].mean()
                    else:
                        y=df.iloc[:,yind].median()
                    data=y
                    typ="Value"
                    cond=None    
                    declf="Leaf"
                    

                
                
            else:
                
                    # This condition refers to no more features to split by
                    # Therefore, we should generate a leaf by either using
                    #   the mean or median
                    
                    if(med==False):
                        y=df.iloc[:,yind].mean()
                    else:
                        y=df.iloc[:,yind].median()
                    data=y
                    typ="Value"
                    cond=None    
                    declf="Leaf"
            
            
            # parent==None indicates that this is the root of the tree
            # Therefore requires setting up:
            
            if(parent==None):
                root=Node.Node(data,cond,typ,declf)
                if(declf=="Decision"):
                    
                    # If the root is a decision, then we recursively run the 
                    #   algorithm with the updated split dataset, feature list,
                    #   and parent Node.
                    
                    for i in range(len(x_cats)):
                        df_1=df[df.iloc[:,indrt]==x_cats[i]]
                        self.buildCART(df=df_1,xinds=xinds,parent=root,es=es,med=med)
                dic=root
            else:
                
                # This condition refers to a parent Node existing and therefore,
                #   the current Node is a child of that parent Node.
                
                if(declf=="Decision"):
                    
                    # The new Node is then inserted into the parent Node
                    
                    
                    nn=parent.insert(data,cond,typ,declf,parent)
                    
                    # If the root is a decision, then we recursively run the 
                    #   algorithm with the updated split dataset, feature list,
                    #   and parent Node
                    
                    for i in range(len(x_cats)):
                        df_1=df[df.iloc[:,indrt]==x_cats[i]]
                        
                        self.buildCART(df=df_1,xinds=xinds,parent=nn,es=es,med=med)
                    
                    # Ensuring that no parent node has only 1 child:
                    
                    if(i==1 and len(nn.children)==1):
                        
                        # If a parent node has only 1 child, then the child
                        #   replaces the parent:
                        
                            newn=nn.children[0]
                            newch=nn.children[0].children
                            newp=parent
                            nn=newn
                            nn.children=newch
                            nn.parent=newp                        
                            nn.parent.children[len(nn.parent.children)-1]=nn
                    
                    # Ensuring that no parent node has only identical leaves:
                    
                    if(len(nn.parent.children)>1):
                        
                        # Determining if the children are all leaves:
                        
                        declfs=[]
                        for i in range(len(nn.parent.children)):
                            declfs.append(nn.parent.children[i].declf)
                        leafs=True
                        for i in range(len(declfs)):
                            if(declfs[i]!="Leaf"):
                                leafs=False
                                break
                        if(leafs):
                            
                            # Ensuring that not all the leaves are the same:
                            
                            same=False
                            comp=nn.parent.children[0].data
                            for i in range(1,len(nn.parent.children)):
                                if(nn.parent.children[i].data==comp):
                                    same=True
                                    break
                            
                            # If the leaves are all the same, then the parent 
                            #   becomes the leaf:
                            if(same):
                                nn.parent.data=comp
                                nn.parent.typ="Value"
                                nn.parent.cond=None    
                                nn.parent.declf="Leaf"
                                nn.parent.children=[]
                    dic=nn
                else:
                    
                    # This condition is if the current tree Node is a leaf
                    # Ensuring that the parent does not have only identical leaves
                    #   as children:
                    same=False
                    for i in range(len(parent.children)):
                        if(parent.children[i].data==data):
                            same=True
                            break
                    if(same):
                        
                        # If the leaves are indentical, then the parent becomes
                        #   the leaf:
                        
                        parent.data=data
                        parent.typ="Value"
                        parent.cond=None    
                        parent.declf="Leaf"
                        parent.children=[]
                        dic=parent
                        
    
                    else:
                        
                        # Else, insert the leaf into the child of the current
                        #   parent:
                        
                        nn=parent.insert(data,cond,typ,declf,parent)
                        dic=nn
            
            # dic is created for debugging and tree visualization purposes:
            
            if(dic.parent==None):
                treedict=self.getTreeDict(dic)
            else:
                
                treedict=self.getTreeDict(dic,self.getTreeDict(dic.parent))
                
                
            # Set the value of our tree to root:
            
            self.tree=root
            
            
            
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
    


    # Method CARTfit does the following:
    #   Extracts 10% of our dataframe as a validation set
    #   Performs 5-fold cross validation on the remaining 90%
    #   Takes 4 of the 5 folds as training data and builds CART upon it
    #   Predicts the values of the remaining fold as the test set and records
    #       the error
    #   Predicts the values of the validation set using the constructed tree of
    #       the training set
    #   Performs early stopping by rebuilding the CART with different values
    #       of the early stopping parameter "es" ranging from 
    #       0.01* the standard deviation of the target variable in the training
    #       set up(std) until 1*std in increments of 0.01*std
    #   Determines the optimum early stopped tree by finding the early stopped 
    #       tree that results in the lowest validation set MSE
    #
    # The method take the following input parameters:
    #   ES: A boolean variable for whether or not to perform Early Stopping
    #   med: A boolean variable indicating whether or not to calculate gmj using
    #       the mean or median
    #
    # The method returns the following:
    #   errorsnes: The fold MSE without Early Stopping
    #   errorses: The fold MSE with Early Stopping
    #   np.mean(errorsnes): The mean of the Non Early Stopped Errors
    #   np.mean(errorses): The mean of the Early Stopped Errors
    #   tnes: The generated Non Early Stopped trees
    #   tes: The generated Early Stopped trees
    # Furthermore, the method prints the pruned and unpruned results in a user
    #   friendly way

            
    def CARTfit(self,ES=False,med=False):
        
        # Initializing the required variables:
        
        df=self.df
        np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})
        yind=self.yind
        
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
        
        errorsnes=[]
        errorses=[]
        tnes=[]
        tes=[]
        for i in range(len(dfs)):
            xinds=self.xinds.copy()
            df_train=dfs[i][0]
            df_test=dfs[i][1]
            treec=decisiontree(df_train,xinds,yind)
            treec.buildCART(med=med)
            
            # tree is the Node variable of the Non-Early Stopped tree
            
            tree=treec.getTree()
            
            # treedict is a dictionary created in order to visualize the tree
            #   on the variable explorer pane
            
            treedict=treec.getTreeDict()
            
            # ypred is the predicted y-values of the test set
            
            ypred=treec.predict(df=df_test)
            
            # er is the MSE of our Non-Early Stopped Tree
            
            er=treec.MSError(ypred,df=df_test)
            
            # treesize is the size of our tree
            
            treesize=self.getTreeSize(tree)
            errorsnes.append(er)
            
            # ypredv is the predicted y-values of the validation set
            
            ypredv=treec.predict(df=df_v)
            
            # erv is the MSE of our validation set:
            
            erv=treec.MSError(ypredv,df=df_v)
            tnes.append(tree)
            
            # If the target variable was normalized, then un-normalize for 
            #   printing and comparison purposes:
            
            if(self.norm):
                
                y=df_test.iloc[:,yind]*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                ypred=np.array(ypred)*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                df_res=pd.DataFrame({'Actual':y,'Predicted':ypred})
            else:
                df_res=pd.DataFrame({'Actual':df_test.iloc[:,yind],'Predicted':ypred})
                
                
            # Printing the Non-Early Stopped results in a user friendly way:
            
            print("CART Decision Tree Model for Fold "+str(i+1)+":")
            print("Early Stopping: No")
            print("Tree Size: "+str(treesize))
            print(df_res.to_string())
            print("MSE: "+str(np.round(er,2)))
            print("Validation Set MSE: "+str(np.round(erv,2)))
            print()
            
            if(ES):
                
                # std is the standard deviation of our target variable in our 
                #   training set
                
                std=df_train.iloc[:,yind].std()
                
                # best is a variable representing the "best" tree, i.e. the 
                #   tree with the lowest validation set MSE, this is initialized
                #   to our Non-Early Stopped Tree
                
                best=tree.copy()
                
                # beste is a variable representing the "best" validation set
                #   error, this is initialzed to the validation set error of
                #   our non-early stopped tree
                
                beste=erv
                
                # Iterating through different Early Stopping Parameters from
                #   0.01std->std in increments of 0.01std
                
                for j in np.arange(0.01*std,1*std,0.01*std):
                    tree1c=decisiontree(df_train,xinds,yind)
                    tree1c.buildCART(es=j)
                    t1=tree1c.getTree()
                    t1d=tree1c.getTreeDict(t1)
                    tree1s=tree1c.getTreeSize(t1)
                    ypvt1=tree1c.predict(df=df_v,tree=t1d)
                    er1=tree1c.MSError(ypvt1,df=df_v)
                    if(er1<=beste):
                        
                        # if the new error is less than or equal to our old error
                        #   then save the values
                        beste=er1
                        best=t1.copy()
                    elif(self.getTreeSize(best)!=treesize):
                        
                        # else if the errors is greater and our best tree is
                        #   smaller than our non-early stopped tree, then break
                        
                        break
                    
                treesizen=self.getTreeSize(best)
                bestdic=self.getTreeDict(best)                
                ypredp=self.predict(df=df_test,tree=bestdic)
                eres=self.MSError(ypredp,df=df_test)
                errorses.append(eres)
                tes.append(t1)
            if(self.norm):
                
                y=df_test.iloc[:,yind]*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                ypredp=np.array(ypredp)*self.stds.iloc[self.yind]+self.means.iloc[self.yind]
                df_res=pd.DataFrame({'Actual':y,'Predicted':ypredp})
            else:
                df_res=pd.DataFrame({'Actual':df_test.iloc[:,yind],'Predicted':ypredp})
            print("CART Decision Tree Model for Fold "+str(i+1)+":")
            print("Early Stopping: Yes")
            print("Tree Size: "+str(treesizen))
            print(df_res.to_string())
            print("MSE: "+str(np.round(eres,2)))
            print("Validation Set MSE: "+str(np.round(beste,2)))
            print()
                
            
        return errorsnes,errorses,np.mean(errorsnes),np.mean(errorses),tnes,tes                    
                    