# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:10:16 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necesarry libraries:

import numpy as np
import pandas as pd
import time
import copy

# class RaceTrack implements Value Iteration, Q-Learning, and SARSA on the 
# Racetrack problem:

class RaceTrack:
    
    # Constructor takes the follwing parameters:
    #   a: A list of strings representing the racetrack text file.
    
    # The constructor initializes the following variables:
    #   track: Our race track in array form
    #   rows: The number of rows in our track (y)
    #   columns: The number of columns in our track (x)
    
    # Furthemore, the constructor runs the method getPoints()
    
    def __init__(self,a):
        rows=int(a[0].replace('\n','').split(',')[0])
        columns=int(a[0].replace('\n','').split(',')[1])
        track = [[0 for x in range(columns)] for y in range(rows)] 
        for i in range(rows):
            track[i]=list(a[i+1].replace('\n',''))
        track.reverse()
        self.track=track
        self.rows=rows
        self.columns=columns
        self.getPoints()
    
    # Method getPoints splits the track points into starts,fins,walls,road where
    #   starts: The coordinates of the starting line
    #   fins: The coordinates of the finish line
    #   walls: The coordinates of the walls
    #   road: The coordinates of the road points
    # The method returns the arrays starts,fins,walls,road and proceeds to run
    #   the method statesactions.
    
    def getPoints(self):
        starts=[]
        fins=[]
        walls=[]
        road=[]
        for i in range(len(self.track)):
            for j in range(len(self.track[0])):
                pos=[]
                pos.append(i)
                pos.append(j)
                if(self.track[i][j]=="S"):
                    starts.append(pos)
                elif(self.track[i][j]=="F"):
                    fins.append(pos)
                elif(self.track[i][j]=="#"):
                     walls.append(pos)
                elif(self.track[i][j]=="."):
                    road.append(pos)


        self.starts=starts
        self.fins=fins
        self.walls=walls
        self.road=road
        self.statesactions()
        return starts,fins,walls,road
    
    
    # Method statesactions defines the states and actions of the racecar and 
    #   racetrack.
    # The method returns and sets the following variables:
    #   S: The states
    #   A: The actions
    
    def statesactions(self):
        
        # A is the list of actions our racecar can take, which is the combination
        #   of ax and ay accelerations between [-1,0,1]:
        
        A=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        S=[]
        maintrack=[]
        for i in range(len(self.starts)):
            maintrack.append(self.starts[i])
        for i in range(len(self.fins)):
            maintrack.append(self.fins[i])
        for i in range(len(self.road)):
            maintrack.append(self.road[i])
        for i in range(len(maintrack)):
                # Appending the different combination of x and y velocities to 
                #   our position coordinates:
                for k in np.arange(-5,6,1):
                    for l in np.arange(-5,6,1):
                        si=[]
                        si.append(maintrack[i][0])
                        si.append(maintrack[i][1])
                        si.append(k)
                        si.append(l)
                        
                        # Appending the tuple of x,y,x_dot,y_dot to our state 
                        #   array S:
                        
                        S.append(si)
        self.S=S
        self.A=A
        return S,A
    
    
    # Method R returns the reward value of a state s 
    # The method takes the following inputs:
    #   s: The state
    # The method returns a value of 0 is the state is in the finish line and 
    #   -1 otherwise
    
    def R(self,s):
        if s[:2] in self.fins:
            return 0
        else:
            return -1
    
    # Method getTf returns value of the summation of the state transition function
    #   for s' in S multiplied by V_t-1(s')
    # The method takes the following inputs:
    #   V: The policy value function
    #   restart: Whether or not to restart the car incase of a crash
    # The method proceeds to run the method TV and sets and returns getT
    
    def getTf(self,V,restart=False):
        S=self.S
        A=self.A
        df_s=pd.DataFrame(data=S) 
        df_s1=df_s.copy()
        self.getT=self.TV(df_s,A,df_s1,V,restart)
        return self.getT
        
        
    # Method ValueIteration performs the ValueIteration algorithm
    # The method takes the following inputs:
    #   thresh: The error threshold upon which to decide convergance
    #   gamma: The discount factor
    #   restart: Whether or not to restart the car incase of a crash
    #   altime: The allowable runtime in minutes
    # The method returns the following:
    #   err: The error difference between the previous and current values of V
    #   Q: The Q function resulting from the algorithm
    #   t: The number of iterations 
    #   (end-start)/60: The elapsed time in minutes of running the algorithm
    
    
    def ValueIteration(self,thresh,gamma,restart=False,altime=0):
        t=0
        err=10**6
        S=self.S
        A=self.A
        
        # Initializing V to an array of 0s:
        
        V=np.zeros(len(S))
        
        # Recording the start time of the algorithm:
        
        start=time.time()
        
        while(err>thresh):
            
            # If the algorithm has exceeded the allowable run time, break:
            
            end=time.time()
            if(altime!=0):
                if((end-start)/60>altime):
                    break
            
            # At t=0, the value of V_t-1 is 0, therefore there is no need to run
            #   the getTf method
            
            if(t>0):
                TV=self.getTf(V,restart)
            
            Vold=copy.deepcopy(V)
            t+=1
            
            # Initializing Q and pi:
            
            Q=np.zeros([len(S),len(A)])
            pi=np.zeros(len(S))
            for s in range(len(S)):
                
                # Getting the reward value of the current state s:
                
                q=self.R(S[s])
                
                # Iterating over the actions:
                
                for a in range(len(A)):
                    if(t>1):
                        q=q+gamma*TV[a][s]
                    
                    # Updating the Q function:
                    
                    Q[s,a]=q
                
                # Setting the pi and V values:
                
                pi[s]=np.max(Q[s])
                V[s]=pi[s]
            
            # Calculating the error:
            
            err=np.max(np.abs(V-Vold))
            
        return err,Q,t,(end-start)/60
        

    

        
    # Method LineGen generates a straight line between points [x1,y1] and [x2,y2]
    # The method takes the following as inputs:
    #   x1: x coordinate of first point
    #   y1: y coordinate of first point
    #   x2: x coordinate of second point
    #   y2: y coordinate of second point
    # The method returns the following:
    #   ps: A list of the points that the straight line between points 1 and 2
    #       consist of.
    
    def LineGen(self,x1,y1,x2, y2):
        
        # The code below decides the slope and the starting point of the straight
        #   line to consider:
        
        if(x1<x2 and y1<y2):
            xs=x1
            xb=x2
            m = (y2 - y1)/(xb-xs)
            y=y1
        elif(x1>x2 and y1>y2):
            xs=x2
            xb=x1
            m = (y1 - y2)/(xb-xs)
            y=y2
        elif(x1<x2 and y2<y1):
            xs=x1
            xb=x2
            m = (y2 - y1)/(xb-xs)
            y=y1
        elif(x1>x2 and y1<y2):
            xs=x2
            xb=x1
            m = (y1 - y2)/(xb-xs)
            y=y2
        elif(x1<x2 and y1==y2):
            xs=x1
            xb=x2
            m = (y2 - y1)/(xb-xs)
            y=y1
        elif(x1>x2 and y1==y2):
            xs=x2
            xb=x1
            m = (y2 - y1)/(xb-xs)
            y=y1
        elif(x1==x2 and y1<y2):
            xb=x1
            xs=x1
            y=y1
            yb=y2
            m=0
        elif(x1==x2 and y1>y2):
            xb=x1
            xs=x1
            y=y2
            yb=y1
            m=0
        else:
            m=0
            xb=x1
            xs=x2
            y=y1
            yb=y2
        
        # b is the calculated y-intercept:
        b=y1-m*(x1)
        if(y1<y2):
            yb=y2
            ys=y1
        else:
            yb=y1
            ys=y2
        ps=[]
        x=xs
        
        # The code below is for the case that x1 and x2 are not equal:
        
        if(x1!=x2):
            while x <xb+1:
                
                # The algorithm works by calculating the coordinates on the 
                #   generated line by the straight line formula y=mx+b
                # However, since the points generated could be decimals and our
                #   track coordinates are only in integers, this has to be
                #   accomdated.
                
                # The y of the corresponding x is calculated:
                
                y=m*x+b
                
                # The y value is rounded to the nearest integer:
                
                y=int(np.round(y,0))
                
                # The code below ensures that no point is missed:
                
                if(len(ps)>=1):
                    if(y>ps[-1][1]+1):
                        xn=((ps[-1][1]+1)-b)/m
                        xn=int(np.round(xn,0))
                        ps.append([xn,ps[-1][1]+1])
                    elif(y<ps[-1][1]-1):
                        xn=((ps[-1][1]-1)-b)/m
                        xn=int(np.round(xn,0))
                        ps.append([xn,ps[-1][1]-1])
                    else:
                        ps.append([x,y])
                        x+=1
                else:
                    ps.append([x,y])
                    x+=1
        
        # The code below is for the case that x1 and x2 are equal: 
           
        else:
            while(y!=yb+1):
                ps.append([xs,y])
                y+=1
        return ps
    


    # Method TV accompanies method getTf by performing the calculations required
    # in returning the value of the summation of the state transition function
    #   for s' in S multiplied by V_t-1(s')
    #
    # The method takes the following inputs:
    #   s1: The current state to consider s
    #   A: The list of actions
    #   s2: The list of possible following states s'
    #   V: The policy value function
    #   restart: Whether or not to restart the car incase of a crash
    #
    # The method applies each of the actions a in A to s1, checks whether or not
    #   a crash occurs, defines the resulting states s2 and produces the values
    #   for the summation of the state transition function for s' in 
    #   S multiplied by V_t-1(s')
    #
    # The method returns:
    #   T: a list containing the values of the summation of the state transition
    #       function for s' in S multiplied by V_t-1(s') for 
    #       each a in A and each s1 in S
    
    def TV(self,s1,A,s2,V,restart=False):
        
        # Initializing T:
        
        T=[]

        # For each action:
        
        for a in A:
            
            # Setting the current state position and velocities:
            
            v1x=s1.iloc[:,2]
            v1y=s1.iloc[:,3]       
            p1x=s1.iloc[:,0]
            p1y=s1.iloc[:,1]
            
            # Calculating the velocities resulting from a:
            
            vx=v1x+a[0]
            vy=v1y+a[1]
            
            # Ensuring that our resulting velocities are within our limits:
            
            vx=vx.apply(lambda x:5 if x>5 else x)
            vx=vx.apply(lambda x:-5 if x<-5 else x)
            vy=vy.apply(lambda x:5 if x>5 else x)
            vy=vy.apply(lambda x:-5 if x<-5 else x)
            
            # Calculating our new positions:
            
            px=p1x+vx
            py=p1y+vy
            
            # Setting up a pandas dataframe pcon containing the following columns:
            #   p1x: Our current x coordinate
            #   p1y: Our current y coordinate
            #   px: Our new x coordinate
            #   py: Our new y coordinate
            
            pcon=pd.concat([p1x,p1y,px,py],axis=1)
            
            # Applying the LineGen method on pcon:
            
            ps=pcon.apply(lambda x:self.LineGen(x.iloc[0],x.iloc[1],x.iloc[2],x.iloc[3]),axis=1)
            
            
            Ts1=[]
            
            # The code below checks each of the LineGen paths for crashes:
            
            for i in range(len(ps)):
                walled=False
                
                # Checking whether our current state location is in the start or
                #   end of the line:
                
                if(ps[i][0]==pcon.iloc[i,:2].values.tolist()):
                    
                    # Checking if any of the coordinates are walls:
                    
                    for j in range(len(ps[i])):
                        
                        # If a wall is found:
                        if(ps[i][j] in self.walls):
                            
                            # Determining which crash variant to follow:
                            
                            if(restart==False):
                                
                                # Incase a restart is not required,set the car 
                                #   location to the nearest crash point on track:
                                
                                if(j>0):
                                    pcon.iloc[i,2]=ps[i][j-1][0]
                                    pcon.iloc[i,3]=ps[i][j-1][1]
                                    vx.iloc[i]=0
                                    vy.iloc[i]=0
                                else:
                                    pcon.iloc[i,2]=pcon.iloc[i,0]
                                    pcon.iloc[i,3]=pcon.iloc[i,1]
                                    vx.iloc[i]=0
                                    vy.iloc[i]=0
                                walled=True
                                break
                            else:
                                
                                # Incase a restart is required when crashing,
                                #   set the car location to a random point on the
                                #   starting line:
                                
                                newps=self.starts[np.random.randint(0,len(self.starts))]
                                pcon.iloc[i,2]=newps[0]
                                pcon.iloc[i,3]=newps[1]
                                vx.iloc[i]==0
                                vy.iloc[i]=0
                    
                    # Ensuring that none of our points are a wall incase LineGen
                    #   missed one:
                    
                    if(walled==False):
                        if(pcon.iloc[i,2:].values.tolist() in self.walls):
                            if(restart==False):
                                pcon.iloc[i,2]=ps[i][j-1][0]
                                pcon.iloc[i,3]=ps[i][j-1][1]
                                vx.iloc[i]=0
                                vy.iloc[i]=0
                                walled=True
                            else:
                                newps=self.starts[np.random.randint(0,len(self.starts))]
                                pcon.iloc[i,2]=newps[0]
                                pcon.iloc[i,3]=newps[1]
                                vx.iloc[i]==0
                                vy.iloc[i]=0
                else:
                    for j in np.arange(len(ps[i])-1,-1,-1):
                        if(ps[i][j] in self.walls):
                            if(restart==False):
                                if(j<len(ps[i])-1):
                                    pcon.iloc[i,2]=ps[i][j+1][0]
                                    pcon.iloc[i,3]=ps[i][j+1][1]
                                    vx.iloc[i]=0
                                    vy.iloc[i]=0
                                else:
                                    pcon.iloc[i,2]=pcon.iloc[i,0]
                                    pcon.iloc[i,3]=pcon.iloc[i,1]
                                    vx.iloc[i]=0
                                    vy.iloc[i]=0
                                walled=True
                                break
                            else:
                                newps=self.starts[np.random.randint(0,len(self.starts))]
                                pcon.iloc[i,2]=newps[0]
                                pcon.iloc[i,3]=newps[1]
                                vx.iloc[i]==0
                                vy.iloc[i]=0
                    if(walled==False):
                        if(pcon.iloc[i,2:].values.tolist() in self.walls):
                            if(restart==False):
                                pcon.iloc[i,2]=ps[i][j+1][0]
                                pcon.iloc[i,3]=ps[i][j+1][1]
                                vx.iloc[i]=0
                                vy.iloc[i]=0
                                walled=True
                            else:
                                newps=self.starts[np.random.randint(0,len(self.starts))]
                                pcon.iloc[i,2]=newps[0]
                                pcon.iloc[i,3]=newps[1]
                                vx.iloc[i]==0
                                vy.iloc[i]=0
                
                # Finding the state index that matches our resulting state s':
                
                df_ch=s2[s2.iloc[:,0]==pcon.iloc[i,2]]
                df_ch=df_ch[df_ch.iloc[:,1]==pcon.iloc[i,3]]
                df_ch=df_ch[df_ch.iloc[:,2]==vx.iloc[i]]
                df_ch=df_ch[df_ch.iloc[:,3]==vy.iloc[i]]
                Ts=0
                
                # 0.8 is considered due to the non-determinism of 20%:
                
                if(len(df_ch)>=1):
                    for index,row in df_ch.iterrows():
                        Ts+=0.8*V[index]

                Ts1.append(Ts)
            T.append(Ts1)
        return T
    
    
    # Method Race performs the race with the racecar on the track using the
    #   converged Q values.
    # The method takes the following as input:
    #   Q: The Q function to race with
    # The method returns:
    #   t: The number of steps it took to finish the race
    
    def Race(self,Q,restart=False):
        
        # Setting the states and actions:
        
        S=self.S
        A=self.A
        
        # Choosing a random position on the starting line to begin the race:
        
        s=np.random.randint(0,len(self.starts))
        ss=[self.starts[s][0],self.starts[s][1],0,0]
        t=0
        s=S.index(ss)
        
        trk1=list(self.track)
        
        # moved is a boolean variable checking whether the car has moved:
        
        moved=False
        
#        print("* Indicates Location of Car on Track")
        
        # Beginning the race:
        
        while(ss[:2] not in self.fins):
            
            # Setting moved to True if the car has moved:
            
            if(ss[:2] not in self.starts):
                moved=True
            
            # Checking if the car has moved at all within 5 steps:
            
            if(t>5 and moved==False):
#                print("Car hasn't moved from starting line")
                t=-1
                break
            
            # Checking if the car has completed the race within 500 steps:
            
            if(t>500):
                t=-1
                break
            crt=copy.deepcopy(trk1)
            xp=ss[0]
            yp=ss[1]
            crt[xp][yp]='*'
#            print("t="+str(t))
#            print("State:"+str(ss))
#            if(t==0):
##                print("Action: None")
#            else:
##                print("Action:"+str(a))
            for rt in np.arange(len(crt)-1,-1,-1):
                s1=str(crt[rt]).replace('[','')
                s1=s1.replace(']','')
#                print(s1)
            
            # Selecting the action while applying non-determinism:
            
            ai=np.argmax(Q[s])
            a=np.multiply(A[ai],np.random.binomial(1,0.8)).tolist()
            
            # Checking for crashes using method wallchecker:
            
            px,py,vx,vy,walled=self.wallchecker(ss,a,restart)
            if([px,py,vx,vy] in S):
                sspi=S.index([px,py,vx,vy])
            else:
                sspi=s
            s=sspi
            ss=S[s]
            t+=1
            if(ss[:2] in self.fins):
                xp=ss[0]
                yp=ss[1]
                crt[xp][yp]='*'
#                print("t="+str(t))
#                print("State:"+str(ss))
#                if(t==0):
##                    print("Action: None")
#                else:
##                    print("Action:"+str(a))
                for rt in np.arange(len(crt)-1,-1,-1):
                    s1=str(crt[rt]).replace('[','')
                    s1=s1.replace(']','')
#                    print(s1)
        return t
        
    

                
    # Method Qsegmentor splits the track for Q-learning and SARSA training in an
    #   effort to improve convergance
    #
    # The method takes the following input parameters:
    #   trk: A string representing the track type ('L','O','R') to implement 
    #       the appropriate splitting method
    #   srsa: A boolean variable of whetehr or not to train with SARSA
    #   gamma: The discount factor
    #   n: The learning rate
    #   maxe: The maximum number of epochs 
    #   restart: Whether or not to restart the car in case of a crash
    #   maxer: The error threshold to consider for convergance
    #   altime: The allowable time in minutes for the algorithm to run
    #
    # The method returns the following:
    #   Q: The converged Q values
    #   df_res: A dataframe containing the following columns:
    #       x: The x coordinate of the car
    #       y: The y coordinate of the car
    #       vx: The x velocity of the car
    #       vy: The y velocity of the car
    #       e: The difference between the current Q values and the previous ones
    #               on convergance
    #       c1: The number of epochs done
    #       c2: A list of the number of epochs required in each epoch
    #   time.time(): The current time of the system

                                            
    def Qsegmentor(self,trk,srsa=False,gamma=0.5,n=1,maxe=20000,restart=False,maxer=10**-2,altime=0):
        
        # Setting the states and actions:
        
        S=self.S
        A=self.A
        
        # Initializing Q randomly:
        
        Q=np.random.normal(loc=0.0, scale=1.0, size=[len(S),len(A)])
        
        
        minx=np.min(np.array(S)[:,0])
        maxx=np.max(np.array(S)[:,0])
        miny=np.min(np.array(S)[:,1])
        maxy=np.max(np.array(S)[:,1])
        fins=np.array(self.fins)
        finsx=np.unique(fins[:,0])
        finsy=np.unique(fins[:,1])
        Sd=pd.DataFrame(data=S)
        
        # Since the tracks will be split, our state space to be used in our
        #   learning algorithms will have to be split as well
        # The modified state space is S1:
        
        
        S1=[]
        for index,row in Sd.iterrows():
            if row.iloc[:2].tolist() in self.fins:
                S1.append(row.tolist())
        curl=fins[0].tolist()
        df_res=[]
        
        # Below is our track choosing code:
        
        if(trk=="L"):
            
            # In case of the L track:
            #   The segmentor begins by taking the row under the finish line
            #       running our learning algorithms on all 4 columns of the rows
            #       with velocities at 0 and goes downwards in decrements of 2
            #   After it reaches the bottom of the track, the segmentor proceeds
            #       to go horizontally leftward in decrements of 10 columns
            
            # For the L-track, since the further the starting point of the
            #   learning algorithms, the larger the iterations required to
            #   converge. Therefore the allocated time considers this by giving
            #   each split 2**count1/1020 * altime with count1 increasing as
            #   we approach the start line.
            #
            # ext is a measure of extra time, incase the algorithms converged Q
            #   in less than the allowable time, the extra time can be carried 
            #   onto the next state:
            
            ext=0
            count1=0

            # Begining our split from one row below the finish line and
            #   decrementing by 2:
                                      
            for ri in np.arange(finsx-1,minx-1,-2):
                
                # Updating the list S1 based on the current environment to
                #   consider:
                
                Sd1=Sd[Sd.iloc[:,0]==ri]
                Sd1=Sd1[Sd1.iloc[:,1]>31]
                S1.extend(Sd1.values.tolist())
                
                # Running our learning algorithms on each road column 
                #   at 0 velocities:
                
                for rj in np.arange(35,31,-1):
                    if([ri,rj] in np.array(S)[:,:2].tolist()):
                        if([ri,rj,0,0] in S):
                            s=S.index([ri,rj,0,0])
                            curl=S[s]
                            
                            # Below is the chooser for whetehr to use Q-learning
                            #   or SARSA based on the variable srsa:
                            
                            if(srsa):
                                Q,e,c1,c2,eltime=self.SARSA(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/1020)+ext)
                            else:
                                Q,e,c1,c2,eltime=self.Qlearning(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/1020)+ext)
                            
                            # Determining whether extra time will be allocated to
                            #   the next run:
                            
                            if(eltime<(altime*(2**count1)/1020)+ext):
                                ext=(altime*(2**count1)/1020)+ext-eltime
                            else:
                                ext=0
                
                # Incrementing count1:
                
                count1+=1
                
                # Updating S1:
                
                Sd1=Sd[Sd.iloc[:,0]>ri-2]
                Sd1=Sd1[Sd1.iloc[:,0]<ri]
                Sd1=Sd1[Sd1.iloc[:,1]>31]
                S1.extend(Sd1.values.tolist())
            for rj in np.arange(31,0,-10):
                Sd1=Sd[Sd.iloc[:,1]==rj]
                Sd1=Sd1[Sd1.iloc[:,0]<5]
                S1.extend(Sd1.values.tolist())
                for ri in np.arange(4,0,-1):
                    vi=0
                    vj=0
                    if([ri,rj,vi,vj] in S):
                        s=S.index([ri,rj,vi,vj])
                        curl=S[s]
                        if(srsa):
                            Q,e,c1,c2,eltime=self.SARSA(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/1020)+ext)
                        else:
                            Q,e,c1,c2,eltime=self.Qlearning(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/1020)+ext)
                        if(eltime<(altime*(2**count1)/1020)+ext):
                            ext=(altime*(2**count1)/1020)+ext-eltime
                        else:
                            ext=0
                        if([ri,rj] in self.starts):
                            dt=[ri,rj,vi,vj]
                            dt.append(e)
                            dt.append(c1)
                            dt.append(c2)
                            df_res.append(dt)
                count1+=1
                Sd1=Sd[Sd.iloc[:,1]>rj-10]
                Sd1=Sd1[Sd1.iloc[:,1]<rj]
                Sd1=Sd1[Sd1.iloc[:,0]<5]
                S1.extend(Sd1.values.tolist())
        elif(trk=="O"):
            
            
            # In case of the O track:
            #   The segmentor begins by taking the first half of the track from
            #       the finish linerow under the finish line running our 
            #       learning algorithms on the first and last column of that row
            #       with velocities at 0.
            #   After that, it proceeds to take each point of the starting line.
            
            # For the O-track, since the further the starting point of the
            #   learning algorithms, the larger the iterations required to
            #   converge. Therefore the allocated time considers this by giving
            #   each split 2**count1/6 * altime with count1 increasing as
            #   we approach the start line.
            
            S1=[]
            ext=0
            count1=0
            Sd1=Sd[Sd.iloc[:,0]<=12]
            S1.extend(Sd1.values.tolist())
            s=S.index([12,20,0,0])
            if(srsa):
                Q,e,c1,c2,eltime=self.SARSA(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/6)+ext)
            else:
                Q,e,c1,c2,eltime=self.Qlearning(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/6)+ext)
            if(eltime<(altime*(2**count1)/6)+ext):
                ext=(altime*(2**count1)/6)+ext-eltime
            else:
                ext=0
            s=S.index([12,23,0,0])
            if(srsa):
                Q,e,c1,c2,eltime=self.SARSA(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/6)+ext)
            else:
                Q,e,c1,c2,eltime=self.Qlearning(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/6)+ext)
            if(eltime<(altime*(2**count1)/6)+ext):
                ext=(altime*(2**count1)/6)+ext-eltime
            else:
                ext=0
            count1+=1
            Sd1=Sd[Sd.iloc[:,0]>12]
            S1.extend(Sd1.values.tolist())
            for i in range(len(self.starts)):
                s=S.index([self.starts[i][0],self.starts[i][1],0,0])
                if(srsa):
                    Q,e,c1,c2,eltime=self.SARSA(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/12)+ext)
                else:
                    Q,e,c1,c2,eltime=self.Qlearning(Q,S1,s,gamma,n,maxe,restart,maxer,altime=(altime*(2**count1)/12)+ext)
                if(eltime<(altime*(2**count1)/6)+ext):
                    ext=(altime*(2**count1)/6)+ext-eltime
                else:
                    ext=0
                dt=[self.starts[i][0],self.starts[i][1],0,0]
                dt.append(e)
                dt.append(c1)
                dt.append(c2)
                df_res.append(dt)
        elif(trk=='R'):
            
            # In case of the R track:
            #   No segmentation happens and the learning algorithms are performed
            #       on each point of the starting line at 0 velocity
            
            # For the R-track, since no segmentation happens, the allowable time
            #   is at altime/5 since there are 5 different points upon which to
            #   run the algorithms on.
            ext=0
            for i in range(len(self.starts)):
                s=S.index([self.starts[i][0],self.starts[i][1],0,0])
                if(srsa):
                    Q,e,c1,c2,eltime=self.SARSA(Q,S,s,gamma,n,maxe,restart,maxer,altime=altime/5)
                else:
                    Q,e,c1,c2,eltime=self.Qlearning(Q,S,s,gamma,n,maxe,restart,maxer,altime=altime/5)
                if(eltime<(altime/5)+ext):
                    ext=(altime/5)+ext-eltime
                else:
                    ext=0
                dt=[self.starts[i][0],self.starts[i][1],0,0]
                dt.append(e)
                dt.append(c1)
                dt.append(c2)
                df_res.append(dt)
                                            
        return Q,df_res,time.time()                                 
    



    # Method Qlearning performs the Q-learning algorithm
    #
    # The method takes the following inputs:
    #   Q: The initial Q function
    #   S1: The states
    #   s: The state from where to start
    #   gamma: The discount factor
    #   n: The learning rate
    #   maxe: The maximum number of epochs
    #   restart: Whether or not to restart the car from the start incase of a crash
    #   maxer: The maximum threshold error of convergance
    #   altime: The allowable time in minutes for the algorithm to run
    #
    # The method returns the following:
    #   Q: The converged Q function
    #   err: The difference between the current Q values and the previous ones on
    #       the final run
    #   epochs: The number of epochs ran
    #   counts: A list of the epochs per epoch
    #   (end-start)/60: Elapsed time in minutes of running
    
    def Qlearning(self,Q,S1,s,gamma=0.5,n=1,maxe=20000,restart=False,maxer=10**-2,altime=0):
        
        # Setting the states and actions:
        
        S=self.S
        A=self.A
        
        # Initializing counts and err:
        
        counts=[]
        err=10**6
        
        # Defining sog as our initial state:
        
        sog=s
        
        n1=n
        epochs=0
        strans=[]
        
        # Defining the start time of the system to calculate time elpased:
        
        start=time.time()
        
        end=time.time()
        
        while(err>maxer and epochs<maxe):
            
            # Defining ss as our current state:
            
            ss=S[sog]
            
            count=0
            n=n1
            s=sog
            st=[]
            
            # Setting Qold as the previous Q:
            
            Qold=copy.deepcopy(Q)
            
            # Asessing if the time ran is greater than the allowable time:
            
            if(altime!=0):
                if((end-start)/60>altime):
                    break
            
            while(ss[:2] not in self.fins):
                st.append(ss)
                
                # Performing epsilon-greedy in choosing the action:
                
                eps=np.random.binomial(1,0.9)
                if(eps==1):
                    ai=np.argmax(Q[s])
                else:
                    ai=np.random.randint(0,len(Q[s]))
                
                # Applying non-determinism:
                
                a=np.multiply(A[ai],np.random.binomial(1,0.8)).tolist()
                
                # Checking for crashes:
                
                px,py,vx,vy,walled=self.wallchecker(ss,a,restart)
                
                # If the resulting state is in our state space S1, then update
                #   Q accoridngly:
                
                if([px,py,vx,vy] in S1):
                    sspi=S1.index([px,py,vx,vy])
                    sspimain=S.index([px,py,vx,vy])
                    Q[s,ai]=Q[s,ai]+n*(self.R([px,py,vx,vy])+gamma*np.max(Q[sspimain])-Q[s,ai])
                
                # Else, maintain the position but with 0 velocity:
                
                else:
                    sspi=S1.index([S[s][0],S[s][1],0,0])
                    sspimain=S.index([S[s][0],S[s][1],0,0])
                
                s=sspimain
                ss=S1[sspi]
                count+=1
            counts.append(count)
            epochs+=1
            err=np.max(np.abs((Q-Qold)))
            strans.append(st)
            
            # Measuring the current system time:
            
            end=time.time()
            
            # If the error is 0, therefore we can assume Q has converged for the 
            #   time being and we break:
            
            if(err==0):
                return Q,err,epochs,counts,(end-start)/60
        return Q,err,epochs,counts,(end-start)/60






    # Method SARSA performs the SARSA algorithm
    #
    # The method takes the following inputs:
    #   Q: The initial Q function
    #   S1: The states
    #   s: The state from where to start
    #   gamma: The discount factor
    #   n: The learning rate
    #   maxe: The maximum number of epochs
    #   restart: Whether or not to restart the car from the start incase of a crash
    #   maxer: The maximum threshold error of convergance
    #   altime: The allowable time in minutes for the algorithm to run
    #
    # The method returns the following:
    #   Q: The converged Q function
    #   err: The difference between the current Q values and the previous ones on
    #       the final run
    #   epochs: The number of epochs ran
    #   counts: A list of the epochs per epoch
    #   (end-start)/60: Elapsed time in minutes of running
    
    def SARSA(self,Q,S1,s,gamma=0.5,n=1,maxe=20000,restart=False,maxer=10**-2,altime=0):
        
        # Setting the states and actions:
        
        S=self.S
        A=self.A

        # Initializing counts and err:
        
        counts=[]
        err=10**6
        
        # Defining sog as our initial state:
        
        sog=s
        n1=n
        epochs=0
        
        # Defining the start time of the system to calculate time elpased:
        
        start=time.time()
        
        end=time.time()
        
        while(err>maxer and epochs<maxe):

            # Defining ss as our current state:
            
            ss=S[sog]
            
            count=0
            Qold=copy.deepcopy(Q)
            n=n1
            
            # Asessing if the time ran is greater than the allowable time:
            
            if(altime!=0):
                if((end-start)/60>altime):
                    break
            
            # Performing epsilon-greedy in choosing the action:
            
            eps=np.random.binomial(1,0.9)
            if(eps==1):
                ai=np.argmax(Q[s])
            else:
                ai=np.random.randint(0,len(Q[s]))
                
            while(ss[:2] not in self.fins):

                # Applying non-determinism:
                
                a=np.multiply(A[ai],np.random.binomial(1,0.8)).tolist()
                
                # Checking for crashes:
                
                px,py,vx,vy,walled=self.wallchecker(ss,a,restart)
                
                # If the resulting state is in our state space S1, then choose
                #   the next action with epsilon greedy and update Q accoridngly:
                
                if([px,py,vx,vy] in S1):
                    sspi=S1.index([px,py,vx,vy])
                    sspimain=S.index([px,py,vx,vy])
                    eps=np.random.binomial(1,0.9)
                    if(eps==1):
                        ain=np.argmax(Q[sspimain])
                    else:
                        ain=np.random.randint(0,len(Q[sspimain]))
                    Q[s,ai]=Q[s,ai]+n*(self.R([px,py,vx,vy])+0.5*np.max(Q[sspimain,ain])-Q[s,ai])
                    ai=ain
                
                # Else, maintain the position but with 0 velocity:
                
                else:
                    sspi=S1.index([S[s][0],S[s][1],0,0])
                    sspimain=S.index([S[s][0],S[s][1],0,0])

                s=sspimain
                ss=S1[sspi]
                count+=1
            counts.append(count)
            epochs+=1
            err=np.max(np.abs((Q-Qold)))
            
            # Measuring the current system time:
            
            end=time.time()
            
            
            # If the error is 0, therefore we can assume Q has converged for the 
            #   time being and we break:
            
            if(err==0):
                return Q,err,epochs,counts,(end-start)/60
            end=time.time()
            
                    
        return Q,err,epochs,counts,(end-start)/60
    
    
    
    
    # Method wallchecker applies an action a to a state ss and detects crashes.
    #
    # The method takes the following inputs:
    #   ss: The state
    #   a: The action
    #   restart: Whether or not to restart incase of a crash
    #
    # The method returns the following:
    #   px: The position x coordinate after applying the action
    #   py: The position y coordinate after applying the action
    #   vx: The x velocity after applying the action
    #   vy: The y velocity after applying the action
    #   walled: A boolean variable indicating whether or not a wall was found
    
    
    def wallchecker(self,ss,a,restart=False):
        
        # Calculating the velocities resulting from a:
        
        vx=ss[2]+a[0]
        vy=ss[3]+a[1]

        # Ensuring that our resulting velocities are within our limits:
        
        if(vx>5):
            vx=5
        elif(vx<-5):
            vx=-5
        if(vy>5):
            vy=5
        elif(vy<-5):
            vy=-5
            
        # p1x: Our current x coordinate
        # p1y: Our current y coordinate 
        
        p1x=ss[0]
        p1y=ss[1]
        
        # Calculating our new positions: 
        #   px: Our new x coordinate
        #   py: Our new y coordinate
        
        px=p1x+vx
        py=p1y+vy
        
        # Applying the LineGen method on p1x,p1y,px,py:
        
        ps=self.LineGen(p1x,p1y,px,py)
        
        
        walled=False
        if(ps!=[]):
            
            # Checking whether our current state location is in the start or
            #   end of the line:
            
            if(ps[0]==[p1x,p1y]):
                
                # Checking if any of the coordinates are walls:
                
                for i in range(len(ps)):
                    
                    # If a wall is found:
                    
                    if(ps[i] in self.walls):
                        
                        # Determining which crash variant to follow:
                        
                        if(restart==False):
                            
                            # Incase a restart is not required,set the car 
                            #   location to the nearest crash point on track:
                            
                            if(i>0):
                                px=ps[i-1][0]
                                py=ps[i-1][1]
                                vx=0
                                vy=0
                            else:
                                px=p1x
                                py=p1y
                                vx=0
                                vy=0
                            walled=True    
                            break
                        else:
                            
                            # Incase a restart is required when crashing,
                            #   set the car location to a random point on the
                            #   starting line:
                            
                            newps=self.starts[np.random.randint(0,len(self.starts))]
                            px=newps[0]
                            py=newps[1]
                            vx=0
                            vy=0
                
                # Ensuring that none of our points are a wall incase LineGen
                #   missed one:
                
                if(walled==False):
                    if([px,py] in self.walls):
                        if(restart==False):
                            px=ps[i-1][0]
                            py=ps[i-1][1]
                            vx=0
                            vy=0
                            walled=True
                        else:
                            newps=self.starts[np.random.randint(0,len(self.starts))]
                            px=newps[0]
                            py=newps[1]
                            vx=0
                            vy=0 
                
            else:
                for i in np.arange(len(ps)-1,-1,-1):
                    if(ps[i] in self.walls):
                        if(restart==False):
                            if(i<len(ps)-1):
                                px=ps[i+1][0]
                                py=ps[i+1][1]
                                vx=0
                                vy=0
                            else:
                                px=p1x
                                py=p1y
                                vx=0
                                vy=0
                            walled=True
                            break
                        else:
                            newps=self.starts[np.random.randint(0,len(self.starts))]
                            px=newps[0]
                            py=newps[1]
                            vx=0
                            vy=0
                if(walled==False):
                    if([px,py] in self.walls):
                        if(restart==False):
                            px=ps[i+1][0]
                            py=ps[i+1][1]
                            vx=0
                            vy=0
                            walled=True
                        else:
                            newps=self.starts[np.random.randint(0,len(self.starts))]
                            px=newps[0]
                            py=newps[1]
                            vx=0
                            vy=0
                        
        return px,py,vx,vy,walled
        
            
                
                
                
        
        
        
        
        
        
       
            