### Policy Iteration and Value Iteration Agents implementation

An implementation of a Policy iteration and Value iteration agent using OpenAI Gym toolkit as support.

### Prerequisites

1. 	You have to know Python programming language.
2. 	You have to install OpenAI Gym toolkit. 
	If you have not installed OpenAI Gym yet follow the instructions here: https://github.com/openai/gym#openai-gym. You can do a minimal or a full installation. Otherwise you can install envs you need by hand.

### After Installation: recommendations about how to write the code

After the installation of the OpenAI Gym you won't need to install anything else. 
We will use the file "tabular_q_agent.py" contained in examples/agents as starting point. I suggest you to copy this file because it will be used later.
We will create two subclasses of tabular_q_agent class to make the games work. 

Now you have to choose a game that uses only discrete observation spaces and discrete action spaces. These spaces must be represented by an Integer or by an array of Integers. For example, spaces like Boxes, Matrices and MultiArrays are NOT allowed. 

You can watch the class env file in order to verify this fact: you have to go in the right subfolder of the game contained in the envs folder and check if the obs_space and the action_space can be connected with DiscreteEnv class in discrete.py, which is contained in the same folder of the environment class file. It's important to specify that DiscreteEnv class refers to Discrete class: this file, which can be found in a gym folder named "spaces", describes discrete space features.

I've chosen for you one games that follow these specs: "Taxi", contained in envs/toy_text.

### TAXI GAME:

-----------------------------------------------------------------------------------------------------------------------------------

There are three different folders:

1. Policy iteration folder 
2. Value iteration folder
3. Q-learning vs Policy and Value iteration folder

-----------------------------------------------------------------------------------------------------------------------------------

***

To find a theoretical explanation on the policy iteration and value iteration refer to the following links:

* VALUE ITERATION:  https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node19.html

* POLICY ITERATION: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node20.html

We recommend reading the relative parts of the book: "Artificial Intelligence: A Modern Approach by Stuart Russell and Peter Norvig"

***

* Explanation on the Policy iteration (first point):

The folder contains Taxi.py script and Policy_iteration_Agent.py. In addition there is a subdirectory Taxi_Analysis that serves to print the graphs of the rewards.
The only news from the previous version is the construction of the PolicyIterationAgent class that creates an agent able to learn a policy to follow on the game.
The learn() function in fact launches the function policy_iteration() to learn a good policy and then performs action on the env following this policy.
The act() function of choosing actions is not used for this reason.

* Explanation on the Value iteration (second point):

The folder contains Taxi.py script and Policy_iteration_Agent.py. The only news from the previous version is the construction of the ValueIterationAgent class that creates an agent able to learn a policy to follow on the game. The learn() function in fact launches the function value_iteration() to learn a good policy and then performs action on the env following this policy. The act() function of choosing actions is not used for this reason.

* Explanation on the comparison Plots (third point):

There are 3 scripts that allow you to create the 3 distinct agents. These 3 agents are: ValueIterationAgent, PolicyIterationAgent and TabularQAgentTaxi (which is a class derived from TabularQAgent). The learn() function on the 3 agents is launched and the rewards average are saved to file. In the Taxi_Analysis folder you can then print the graphs thanks to the matplotlib library by reading the rewards from the files.


### IMPORTANT CHANGES TO THE ORIGINAL tabular_q_agent.py BEFORE RUNNING THE CODE:

In taxi and roulette cases I've modified tabularQAgent class into tabular_q_agent_taxi (This class is a subclass of tabular_q_agent class):

1. 	I've added libraries to make the code work.
2. 	I've replaced self.q[obs.items()] with self.q[obs] because both envs have an Integer as obs descriptor instead of an Integers type array.
3.	I've calculated the arithmetic mean of the rewards after each trial and I've saved it on an external file. 
	We will use this file to do the rewards' plot.
5.	Remember that every game has it own rules. You might have to modify the code like me to make the game work. 
6.	I've added a function LogUpdate() that creates a file log.txt to save some stats.

### How to run the code:

- To shell: python2.7 Taxi.py for policy iteration or value iteration
- In the end you can save the graph on your pc running to shell: python2.7 print_taxi_rewards_plot.py 

For comparison plots:
- To shell: python2.7 compare_q_learning_policy_and_value_it_taxi.py
- After iterations you can run to shell: python2.7 print_taxi_comparison_plots.py to see the plot

### Authors:

* Giacomo Ferro - https://github.com/JackIron

### License:

* MIT license

### Acknowledgments: 

* tabular_q_agent.py Former Developers 
* softmax eps Former Developers
