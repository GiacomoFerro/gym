#-*- coding: iso-8859-15 -*-
import sys
import os

import numpy as np 
import collections
from collections import Iterable
from gym import spaces, utils
from gym.envs.toy_text import discrete

#lib for file log
import time
import datetime

import gym

class ValueIterationAgent(object):

    def __init__(self, observation_space, action_space,**userconfig):#only with discrete env
    	
		if not isinstance(observation_space, spaces.Discrete):
			print('Observation space incompatible. (Only supports Discrete observation spaces.)')
			sys.exit(1)
		if not isinstance(action_space, spaces.Discrete):
			print('Action space incompatible. (Only supports Discrete action spaces.)')
			sys.exit(1)
    
		self.observation_space = observation_space
		self.action_space = action_space
		self.action_n = action_space.n
		self.config = {
			"init_mean" : 0.0,      # Initialize Q values with this mean
			"init_std" : 0.0,       # Initialize Q values with this standard deviation
			"learning_rate" : 0.1,
			"eps": 0.05,            # Epsilon in epsilon greedy policies
			"discount": 0.99,#from 0.95 to 0.99 for taxi game
			"n_iter": 10}        # Number of iterations
		self.config.update(userconfig)
		self.q = collections.defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])
			
    def evaluate_rewards_and_transitions(self, problem, mutate=False):
        # Enumerate state and action space sizes
        num_states = problem.observation_space.n
        num_actions = problem.action_space.n

        # Intiailize T and R matrices
        R = np.zeros((num_states, num_actions, num_states))
        T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
        for state in range(num_states):
            for action in range(num_actions):
                for transition in problem.env.P[state][action]:
                    probability, next_state, reward, done = transition
                    R[state, action, next_state] = reward
                    T[state, action, next_state] = probability

            # Normalize T across state + action axes
                T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
        if mutate:
            problem.env.R = R
            problem.env.T = T  
        return R, T


    def value_iteration(self, problem, R=None, T=None, gamma=0.9, max_iterations=10**6, delta=10**-3):
        """ Runs Value Iteration on a gym problem """
        
        value_fn = np.zeros(problem.observation_space.n)
        if R is None or T is None:
            R, T = self.evaluate_rewards_and_transitions(problem)

        for i in range(max_iterations):
            previous_value_fn = value_fn.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.max(Q, axis=1)

            if np.max(np.abs(value_fn - previous_value_fn)) < delta:
                break

        # Get and return optimal policy
        policy = np.argmax(Q, axis=1)
        #policy= collections.defaultdict(lambda:policy)
        
        return policy, i + 1
    
    def act(self, observation, eps=None):
		if eps is None:
			eps = self.config["eps"]
        # epsilon greedy.
		if np.random.random() > 1-eps:#soft greedy       	
			if isinstance(self.q[observation],Iterable):
				action=np.argmax(self.q[observation])
		else:
			action=self.action_space.sample()
		
		return action

    def learn(self, env, it):
        config = self.config        
        
        problem = gym.make('Taxi-v2')
        sumRewards=0
        for t in range(config["n_iter"]):
            #valuto politica    
            value_policy, iters = self.value_iteration(problem,max_iterations=it)
            obs = problem.reset()   
            
            
            meanRewards=0
            done=False
            l=0
            
            while not done and l < 100:
                obs, reward, done, _ = env.step(value_policy[obs])
                sumRewards+=reward
                l+=1
            
			
        #scrivo su file le rewards..
        dirPath="Taxi_Analysis"#you can change the name path
        try:
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)#make folder in the current directory
        except OSError:
            print ('Error: Creating directory. ' +  dirPath)
        	
        f=open("Taxi_Analysis/rewards_taxi_value_iteration.txt","a")#a to append rewards mean!!
        
        print(sumRewards)
        meanRewards=sumRewards/config["n_iter"] 
        f.write(str(meanRewards)+"\n")
        f.close()
        
    def LogUpdate(self):
       i = datetime.datetime.now()
       f=open("Taxi_Analysis/log.txt","a")
       f.write("INFO TRIAL:%s/%s/%s" %(i.day, i.month, i.year)+"---")
       f.write(time.strftime("%H:%M:%S")+"\n\n")
            
       f.write("\nuser config=%s\n" %self.config)
       
       f.write("observation space = %s\n"%self.observation_space)
       f.write("action space = %s\n"%self.action_space)
       f.write("LIST OF OBSERVATIONS:"+"\n")
       f.write("%s" %self.q.keys())
            
       f.write("\n VALUE ITERATIONS TABLE (OBS AND ACTIONS REWARDS):"+"\n")
       f.write("SIZE OF THE TABLE:%s\n" %len(self.q))
       f.write("%s" %self.q.items())
       
       rewardsFile=open("Taxi_Analysis/rewards_taxi_value_iteration.txt","r")
       f.write("\nLIST OF REWARDS MEAN FOR EACH EPISODE OF THIS TRIAL:\n%s\n" %rewardsFile.read())
       
       f.write("\n\n")
       f.close()
       
       
