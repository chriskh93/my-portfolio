# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:37:57 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necessary libraries:

import pandas as pd
import numpy as np
import random
import string

# class Node defines the node objects that make up the decision tree

class Node:
    
    # Constructor takes the follwing parameters:
    #   data: The value of the Node, typically a decision Node has the following
    #       template : [feature,value_1,value_2,...value_m]
    #       A leaf node is not an array and just a value
    #   cond: The condition of the Node
    #       cond could take a value of: 
    #           CC: Class Chooser
    #           <: Less than
    #   typ: The type of the Node
    #       typ could take a value of:
    #           Categorical
    #           Numerical
    #
    #   declf: Whether the Node is a Decision or a Leaf
    #   parent: The parent of the Node
    #   idn: The unique identifier
    
    # The constructor further initializes the following variables:
    #   children: The children of the Node
    #       Typically children[0] corresponds to value_1 of the data array
    #                 children[m-1] corresponds to value_m of the data array
    #   idn: Initialized to be a single letter with a random integer
    #   checked: An attribute for pruning purposes to indicate whether or not a 
    #            Node within a tree has been checked

    def __init__ (self, data,cond,typ=None,declf="Decision",parent=None,idn=None,):

        self.children=[]
        self.data = data
        self.cond=cond
        self.declf=declf
        self.typ=typ
        self.parent=parent
        if(idn==None):
            idnum=np.random.randint(0,1000000)
            idl=random.choice(string.ascii_letters)
            idn1=idl+str(idnum)
            self.idn=idn1
        else:
            self.idn=idn
        self.checked=None
        
    
    
    # Method insert inserts a Node into the children of this Node
    # The method take the following input parameters which are described above:
    #   data
    #   cond
    #   typ
    #   declf
    #   parent
    #   idn
    # The method appends a new Node with the above characteristics into the
    #   children array of this Node
    # The method returns the new child Node


    def insert(self,data,cond,typ,declf,parent,idn=None):
        
        c=Node(data,cond,typ,declf,parent,idn)
        self.children.append(c)
        return c
    
    
    # Method copy makes a copy of this Node
    # Important to note that copy only copies this Node and its children but not
    #   the parents
    # Therefore, it is used appropriately by copying the root of the tree
    # The method creates a new Node with the same characteristics of this Node
    # The method checks if this Node has any children and recursively copies the 
    #   child nodes as well
    # The copied Node is returned
    
    def copy(self):
        
        if(type(self.data)==list):
        
            n= Node(self.data.copy(),self.cond,self.typ,self.declf,self.parent,self.idn)
            n.checked=self.checked
        else:
            n= Node(self.data,self.cond,self.typ,self.declf,self.parent,self.idn)
            n.checked=self.checked
        
        if(len(self.children))>0:
            childs=[]
            for i in range(len(self.children)):
                
                childs.append(self.children[i].copy())
            n.children=childs
        
        return n
    
    
    # Method root returns the root of this Node
    # The method loops through the parents until there are no more parents
    # The root is returned
    
    def root(self):
        this=self
        while(this.parent!=None):
            this=this.parent
        
        return this
    
    
    # Method setPrune sets a tree up for pruning
    # The method sets the value of 'checked' to False
    # The method recursively runs setPrune on the child nodes and their
    #   respective parent nodes
    # The setPrune'd tree is returned
    
    def setPrune(self):
        self.checked=False
        for i in range(len(self.children)):
            self.children[i].setPrune()
        if(self.parent!=None):
            self.parent.checked=False
        return self
    
    
    # Method setCheck sets the 'checked' value of a node within a tree
    # The method takes the following input parameters:
    #   root: The node to be set
    #   tf: The value of 'checked'
    # The method proceeds to traverse the tree for the value of root recursively
    # Once root is found, it sets its value of 'checked' to tf

    
    def setCheck(self,root,tf):
        if(self.idn!=root.idn):
                
            if(len(self.children)>0):
                for i in range(len(self.children)):
                    self.children[i].setCheck(root,tf)
        else:
            self.checked=tf
                

        
        

                
                
            
            
            
            
            
            
        
        
        
        
            
        
        
        
                
        

