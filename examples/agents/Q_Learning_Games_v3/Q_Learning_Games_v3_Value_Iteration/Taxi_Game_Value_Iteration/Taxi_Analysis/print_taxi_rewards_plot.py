#libreria per generare grafici
import matplotlib.pyplot as plt
#lib to remove files
import os

print("Make the Rewards Plot")

f=open("rewards_taxi_value_iteration.txt","r")

n=0
stringa=f.readline() #conto le ricompense
while stringa!="":
    n+=1
    stringa=f.readline()

newRewards=[]
rewards=[0 for i in range(n)]

f=open("rewards_taxi_value_iteration.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
    rewards[n]=int(stringa)
    n+=1
    stringa=f.readline()           

f.close()

eps=range(1,51)
newRewards.append({"x": eps, "y": rewards, "ls": "-", "label": "Value Iteration"})


plt.figure(figsize=(15, 10))
for s in newRewards:
        plt.plot(s["x"], s["y"], label=s["label"])
        
plt.title("Rewards collected over the time for Taxi game with Value Iteration")
       
plt.xlabel("Trials")
plt.ylabel("Rewards")
plt.grid()#put the grid
 
plt.show()#print in output the plot and give the possibility to save it on your computer 
plt.savefig("rewards_taxi_value_iteration.png")

os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v3/Q_Learning_Games_v3_Value_Iteration/Taxi_Game_Value_Iteration/Taxi_Analysis/rewards_taxi_value_iteration.txt")


