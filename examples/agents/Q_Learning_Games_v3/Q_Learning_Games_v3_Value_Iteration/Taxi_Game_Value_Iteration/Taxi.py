import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import Value_Iteration_Agent

env=taxi.TaxiEnv()#make TaxiEnv

agent=Value_Iteration_Agent.ValueIterationAgent(env.observation_space,env.action_space)

print("BEGIN THE VALUE ITERATION")
for i in range(50):
	agent.learn(env,i+1) #learn best choices and act on the env
	print("trial number: %d" %i)

agent.LogUpdate()

