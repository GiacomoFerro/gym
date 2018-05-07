#comparison between Sarsa q-learning and tabular-q-learning with matplotlib
import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import tabular_q_agent_taxi
import Policy_Iteration_Agent
import Value_Iteration_Agent

env=taxi.TaxiEnv()#make TaxiEnv

agentpolicy=Policy_Iteration_Agent.PolicyIterationAgent(env.observation_space,env.action_space)

print("BEGIN POLICY ITERATION")
for i in range(50):
	agentpolicy.learn(env,i+1) #learn best choices and act on the env
	print("trial number: %d" %i)
	
agent=tabular_q_agent_taxi.TabularQAgentTaxi(env.observation_space,env.action_space)

print("BEGIN TABULAR Q-LEARNING")#tabular q learning Algo
for i in range(50):
	agent.learn(env)
	print("trial number: %d" %i)

agentvalue=Value_Iteration_Agent.ValueIterationAgent(env.observation_space,env.action_space)

print("BEGIN VALUE ITERATION")
for i in range(50):
	agentvalue.learn(env,i+1)
	print("trial number: %d" %i)

