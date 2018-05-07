#libreria per generare grafici
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#lib to remove files
import os

print("Make the comparison Plots")

f=open("rewards_taxi_qlearning.txt","r")
stringa=f.readline()
n=0

plt.figure(figsize=(15, 10))

while stringa!="":#count the number of rewards
	n+=1 
	stringa=f.readline()

newRewards=[]
rewards=[0 for i in range(n)]

#read q-learning rewards
f=open("rewards_taxi_qlearning.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	rewards[n]=float(stringa)
	n+=1
	stringa=f.readline()           

f.close()

eps=range(1,51)
newRewards.append({"x": eps, "y": rewards, "ls": "-", "label": "Q-Learning"})

for s in newRewards:#add q learning rewards
        plt.plot(s["x"], s["y"], label=s["label"])

#read policy rewards
f=open("rewards_taxi_pol_iteration.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	rewards[n]=int(stringa)
	n+=1
	stringa=f.readline()           

f.close()

eps=range(1,51)
newRewards[:]=[]
newRewards.append({"x": eps, "y": rewards, "ls": "-", "label": "Policy Iteration"})

for s in newRewards:#add pol it rewards
       plt.plot(s["x"], s["y"], label=s["label"])

#read value rewards
f=open("rewards_taxi_value_iteration.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	rewards[n]=int(stringa)
	n+=1
	stringa=f.readline()           

f.close()

eps=range(1,51)
newRewards[:]=[]
newRewards.append({"x": eps, "y": rewards, "ls": "-", "label": "Value Iteration"})

for s in newRewards:
        plt.plot(s["x"], s["y"], label=s["label"])
        
plt.title("Taxi Game comparison plot")
       
plt.xlabel("Trials")
plt.ylabel("Rewards")
plt.grid()#put the grid
plt.legend(loc="lower right")

plt.show()#print in output the plot and give the possibility to save it on your computer 
plt.savefig("rewards_taxi_comparison.png")

os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v3/Q_learning_vs_Policy_Value_Iteration/Taxi_Game/Taxi_Analysis/rewards_taxi_value_iteration.txt")
os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v3/Q_learning_vs_Policy_Value_Iteration/Taxi_Game/Taxi_Analysis/rewards_taxi_pol_iteration.txt")
os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v3/Q_learning_vs_Policy_Value_Iteration/Taxi_Game/Taxi_Analysis/rewards_taxi_qlearning.txt")



