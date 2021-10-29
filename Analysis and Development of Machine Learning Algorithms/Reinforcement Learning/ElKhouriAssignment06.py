# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:48:55 2020

@author: Christopher El Khouri
        605.649.81
"""

# Importing the necessary libraries:

import RaceTrack
import sys
import numpy as np
import time
sys.stdout = open('output.txt','wt')


# The code below runs a total of 44 experiments, consisting of 11 experiments
# on each of the L,O, and R tracks. The R track has 11 experiements on each crash
# scenario.
# 3 experiments are run on the Value Iteration with the discount factor varied
# between 0.1,0.5 and 0.9.
# 3 experiments are run on SARSA and Q-learning varying the discount factors 
# similarliy to Value Iteration, however an additional run on gamma=0.5 is done. 
#z=track.LineGen(1,1,-1,-4)
count=1
ext=0
while(count<45):
    
    # Below is our track choosing code which decides on the track to experiment 
    # with depending on the experiment number (count)
    # altime is the allowable time in minutes of each track.
    
    if(count==1):
        altime=2*60
        expl=[]
        tf=open("Tracks/L-track.txt", "r")
        trk=tf.readlines()
        print("L-track Experiments:")
        print()
    elif(count==12):
        expo=[]
        tf=open("Tracks/O-track.txt", "r")
        trk=tf.readlines()
        print("O-track Experiments:")
        print()
        altime=4*60
    elif(count==23):
        expr=[]
        tf=open("Tracks/R-track.txt", "r")
        trk=tf.readlines()
        print("R-track Experiments without Restart:")
        print()
        altime=3*60
    elif(count==34):
        tf=open("Tracks/R-track.txt", "r")
        trk=tf.readlines()
        print("R-track Experiments with Restart:")
        print()
        altime=3*60
        
    # Once the track is chosen, the code below chooses the algorithm (VI,QL,S) to
    # experiment with depending on the experiment number:
    
    if(count in [1,2,3,12,13,14,23,24,25,34,35,36]):
        
        # The code below is the Value Iteration experiment varying gamma between 
        # 0.1,0.5 and 0.9:
    
        for gamma in np.arange(0.1,1,0.4):
            
            # exp is the list that holds the data generated from our algorithm
            
            exp=[]
            
            # Inserting our track into our RL class RaceTrack:
            
            track=RaceTrack.RaceTrack(trk)
            
            print('Experiment '+str(count)+':')
            print('Value Iteration')
            print('Gamma: '+str(gamma))
            print('Error Threshold: 1e-3')
            
            # The allowable time to run is 1/15 of the total allowable time

            print('Allowable Time in minutes: '+str((altime/15)+ext))
            
            # Running our Value Iteration algorithm on our track:
            
            err,Q,t,tim=track.ValueIteration(10**-3,gamma,altime=(altime/15)+ext)
            print('Number of Iterations: '+str(t))
            print('Error: '+str(err))
            print('Elapsed Training Time in minutes: '+str(tim))
            bt=track.Race(Q,restart)
            exp.append(Q)
            exp.append(t)
            exp.append(err)
            exp.append(tim)
            exp.append(bt)
            if(count<=11):
                expl.append(exp)
            elif(count<=22):
                expo.append(exp)
            else:
                expr.append(exp)
            print('Number of Steps Car Needs to Finish Race: '+str(bt))
            count+=1
    elif(count in [4,5,6,15,16,17,26,27,28,37,38,39]):
        
        # The code below is the Q-Learning experiment varying gamma between 
        # 0.1,0.5 and 0.9:
    
        for gamma in np.arange(0.1,1,0.4):
            
            # exp is the list that holds the data generated from our algorithm

            exp=[]
            
            # Inserting our track into our RL class RaceTrack:
            
            track=RaceTrack.RaceTrack(trk)
            
            # start takes the current time of the system to monitor the actual
            # running time of the algorithm:
            
            start=time.time()
            print('Experiment '+str(count)+':')
            print('Q-Learning')
            print('Gamma: '+str(gamma))
            print('n: '+str(1))
            print('Error Threshold: 1e-2')
            
            # The allowable time to run is 1/10 of the total allowable time

            print('Allowable Time in minutes: '+str((altime/10)+ext))
            
            # The code below chooses the type of RaceTrack for the segmentor to
            # run the algorithm accordingly
            
            if(count<=11):
                trkstr='L'
            elif(count<=22):
                trkstr='O'
            else:
                trkstr='R'
            
            # The code below chooses the type crash variant:
            
            if(count<=33):
                
                # restart=False denotes not to restart the race car in case of
                # a crash
                
                restart=False
            else:

                # restart=True denotes to restart the race car in case of
                # a crash
                
                restart=True
            
            # Running our Q-learning algorithm on our track:
            
            Q,df_res_q,tim=track.Qsegmentor(trkstr,srsa=False,gamma=gamma,n=1,maxe=20000,restart=restart,maxer=10**-2,altime=(altime/10))
            
            tim=(tim-start)/60
            print('Total Number of Iterations Starting from Starting Line: '+str(np.sum([df_res_q[0][5],df_res_q[1][5],df_res_q[2][5],df_res_q[3][5]])))
            print('Elapsed Training Time in minutes: '+str(np.round(tim,1)))
            bt=track.Race(Q,restart)
            exp.append(Q)
            exp.append(df_res_q)
            exp.append(tim)
            exp.append(bt)
            if(count<=11):
                expl.append(exp)
            elif(count<=22):
                expo.append(exp)
            else:
                expr.append(exp)
            print('Number of Steps Car Needs to Finish Race: '+str(bt))
            count+=1
    elif(count in [7,18,29,40]):
        exp=[]
        gamma=0.5
        track=RaceTrack.RaceTrack(trk)
        start=time.time()
        print('Experiment '+str(count)+':')
        print('Q-Learning')
        print('Gamma: '+str(gamma))
        print('n: '+str(1))
        print('Error Threshold: 1e-2')
        print('Allowable Time in minutes: '+str((altime/10)+ext))
        if(count<=11):
            trkstr='L'
        elif(count<=22):
            trkstr='O'
        else:
            trkstr='R'
        if(count<=33):
            restart=False
        else:
            restart=True
        Q,df_res_q,tim=track.Qsegmentor(trkstr,srsa=False,gamma=gamma,n=1,maxe=20000,restart=restart,maxer=10**-2,altime=(altime/10))
        tim=(tim-start)/60
        print('Total Number of Iterations Starting from Starting Line: '+str(np.sum([df_res_q[0][5],df_res_q[1][5],df_res_q[2][5],df_res_q[3][5]])))
        print('Elapsed Training Time in minutes: '+str(np.round(tim,1)))
        bt=track.Race(Q,restart)
        exp.append(Q)
        exp.append(df_res_q)
        exp.append(tim)
        exp.append(bt)
        if(count<=11):
            expl.append(exp)
        elif(count<=22):
            expo.append(exp)
        else:
            expr.append(exp)
        print('Number of Steps Car Needs to Finish Race: '+str(bt))
        count+=1
    
    elif(count in [8,9,10,19,20,21,30,31,32,41,42,43]):
        
        # The code below is the SARSA experiment varying gamma between 
        # 0.1,0.5 and 0.9:

        for gamma in np.arange(0.1,1,0.4):
            
             # exp is the list that holds the data generated from our algorithm
             
            exp=[]
            
            # Inserting our track into our RL class RaceTrack:
            
            track=RaceTrack.RaceTrack(trk)
            
            # start takes the current time of the system to monitor the actual
            # running time of the algorithm:
            
            start=time.time()
            
            print('Experiment '+str(count)+':')
            print('SARSA')
            print('Gamma: '+str(gamma))
            print('n: '+str(1))
            print('Error Threshold: 1e-2')
            
            # The allowable time to run is 1/10 of the total allowable time
            
            print('Allowable Time in minutes: '+str((altime/10)+ext))
            
            # The code below chooses the type of RaceTrack for the segmentor to
            # run the algorithm accordingly
            
            if(count<=11):
                trkstr='L'
            elif(count<=22):
                trkstr='O'
            else:
                trkstr='R'
            # The code below chooses the type crash variant:
            
            if(count<=33):
                
                # restart=False denotes not to restart the race car in case of
                # a crash
                
                restart=False
            else:

                # restart=True denotes to restart the race car in case of
                # a crash
                
                restart=True

            # Running our SARSA algorithm on our track:
                
            Q,df_res_s,tim=track.Qsegmentor(trkstr,srsa=True,gamma=gamma,n=1,maxe=20000,restart=restart,maxer=10**-2,altime=(altime/10)+ext)
            tim=(tim-start)/60
            print('Average Number of Iterations Starting from Starting Line: '+str(np.sum([df_res_s[0][5],df_res_s[1][5],df_res_s[2][5],df_res_s[3][5]])))
            print('Elapsed Training Time in minutes: '+str(tim))
            bt=track.Race(Q,restart)
            exp.append(Q)
            exp.append(df_res_s)
            exp.append(tim)
            exp.append(bt)
            if(count<=11):
                expl.append(exp)
            elif(count<=22):
                expo.append(exp)
            else:
                expr.append(exp)
            
            print('Number of Steps Car Needs to Finish Race: '+str(bt))
            count+=1
    
    elif(count in [11,22,33,44]):
        
        gamma=0.5
        exp=[]
        track=RaceTrack.RaceTrack(trk)
        start=time.time()
        print('Experiment '+str(count)+':')
        print('SARSA')
        print('Gamma: '+str(gamma))
        print('n: '+str(1))
        print('Error Threshold: 1e-2')
        print('Allowable Time in minutes: '+str((altime/10)+ext))
        if(count<=11):
            trkstr='L'
        elif(count<=22):
            trkstr='O'
        else:
            trkstr='R'
        if(count<=33):
            restart=False
        else:
            restart=True
        Q,df_res_s,tim=track.Qsegmentor(trkstr,srsa=True,gamma=gamma,n=1,maxe=20000,restart=restart,maxer=10**-2,altime=(altime/10)+ext)
        tim=(tim-start)/60
        print('Average Number of Iterations Starting from Starting Line: '+str(np.sum([df_res_s[0][5],df_res_s[1][5],df_res_s[2][5],df_res_s[3][5]])))
        print('Elapsed Training Time in minutes: '+str(tim))
        bt=track.Race(Q,restart)
        exp.append(Q)
        exp.append(df_res_s)
        exp.append(tim)
        exp.append(bt)
        if(count<=11):
            expl.append(exp)
        elif(count<=22):
            expo.append(exp)
        else:
            expr.append(exp)
        
        print('Number of Steps Car Needs to Finish Race: '+str(bt))
        count+=1



        

