import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import Policy_Iteration_Agent

env=taxi.TaxiEnv()#make TaxiEnv

agent=Policy_Iteration_Agent.PolicyIterationAgent(env.observation_space,env.action_space)

print("BEGIN THE POLICY ITERATION")
for i in range(50):
	agent.learn(env,i+1) #learn best choices and act on the env i=1 on first iteration
	print("trial number: %d" %i)

agent.LogUpdate()

