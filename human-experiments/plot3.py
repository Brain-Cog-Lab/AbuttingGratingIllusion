
"""
Plot the results of human vs deep learning on AG-MNIST
"""
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 11))
hor4 = np.array([0.77,0.92,0.65])*100
ver4 = np.array([0.62,0.55,0.59])*100
ul4 = np.array([0.57,0.58,0.53])*100
ur4 = np.array([0.35,0.42,0.58])*100
hor6 = np.array([0.58,0.52,0.58])*100
ver6 = np.array([0.26,0.31,0.31])*100
ul6 = np.array([0.33,0.58,0.22])*100
ur6 = np.array([0.12,0.34,0.32])*100
plt.ylim(0,100)
# 人类结果
plt.scatter([1,1,1],hor4, s=100,color='orange',alpha=0.5, label='I=4, human performance')
plt.scatter([2,2,2],ver4, s=100,color='orange', alpha=0.5)
plt.scatter([3,3,3],ul4, s=100,color='orange', alpha=0.5)
plt.scatter([4,4,4],ur4, s=100,color='orange', alpha=0.5)

plt.scatter([1,1,1],hor6, s=50,color='green', alpha=0.5, label='I=6, human performance')
plt.scatter([2,2,2],ver6, s=50,color='green', alpha=0.5)
plt.scatter([3,3,3],ul6, s=50,color='green', alpha=0.5)
plt.scatter([4,4,4],ur6, s=50,color='green', alpha=0.5)

#模型结果
plt.scatter(1,19.04, s=100,color='orange',alpha=0.5, marker='^', label='I=4, deep learning performance') #hor4
plt.scatter(1,18.94, s=50,color='green',alpha=0.5, marker='^', label='I=6, deep learning performance') #hor6
plt.scatter(2,12.45, s=100,color='orange',alpha=0.5, marker='^') #ver4
plt.scatter(2,20.76, s=50,color='green',alpha=0.5, marker='^') #ver6
plt.scatter(3,12.1, s=100,color='orange',alpha=0.5, marker='^') #ul4
plt.scatter(3,9.37, s=50,color='green',alpha=0.5, marker='^') #ul6
plt.scatter(4,12.14, s=100,color='orange',alpha=0.5, marker='^') #ur4
plt.scatter(4,18.92, s=50,color='green',alpha=0.5, marker='^') #ur6

#plt.axhline(100,color='red',linestyle='--')
plt.axhline(10,color='blue',linestyle='--', label = 'Random level')

plt.xticks([0,1,2,3,4,5], labels=['','hor','ver','ul','ur',''],fontsize=15)
plt.yticks(np.arange(0, 105, step=5),fontsize=15)
plt.xlabel('Direction of gratings',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.legend(loc='center',bbox_to_anchor=(0.5,1.07),frameon=False)
plt.show()
