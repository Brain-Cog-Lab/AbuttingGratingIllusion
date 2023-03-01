"""
Plot the results of human vs deep learning on ag-silhouette
"""
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 11))
hor4 = np.array([0.75625,0.80625,0.78125])*100
ver4 = np.array([0.76875,0.6375,0.6625])*100
ul4 = np.array([0.725,0.60625,0.75])*100
ur4 = np.array([0.7,0.7625,0.69375])*100
hor8 = np.array([0.65625,0.71875,0.6875])*100
ver8 = np.array([0.61875,0.59375,0.5375])*100
ul8 = np.array([0.6375,0.6625,0.54375])*100
ur8 = np.array([0.49375,0.825,0.6375])*100
plt.ylim(0,100)
# 人类结果
plt.scatter([1,1,1],hor4, s=100,color='orange',alpha=0.5, label='I=4, human performance')
plt.scatter([2,2,2],ver4, s=100,color='orange', alpha=0.5)
plt.scatter([3,3,3],ul4, s=100,color='orange', alpha=0.5)
plt.scatter([4,4,4],ur4, s=100,color='orange', alpha=0.5)

plt.scatter([1,1,1],hor8, s=50,color='green', alpha=0.5, label='I=8, human performance')
plt.scatter([2,2,2],ver8, s=50,color='green', alpha=0.5)
plt.scatter([3,3,3],ul8, s=50,color='green', alpha=0.5)
plt.scatter([4,4,4],ur8, s=50,color='green', alpha=0.5)

#模型结果
plt.scatter(1,0.4*100, s=100,color='orange',alpha=0.5, marker='^', label='I=4, deep learning performance') #hor4
plt.scatter(1,0.14375*100, s=50,color='green',alpha=0.5, marker='^', label='I=8, deep learning performance') #hor8
plt.scatter(2,0.4*100, s=100,color='orange',alpha=0.5, marker='^') #ver4
plt.scatter(2,0.15*100, s=50,color='green',alpha=0.5, marker='^') #ver8
plt.scatter(3,0.3625*100, s=100,color='orange',alpha=0.5, marker='^') #ul4
plt.scatter(3,0.09375*100, s=50,color='green',alpha=0.5, marker='^') #ul8
plt.scatter(4,0.41875*100, s=100,color='orange',alpha=0.5, marker='^') #ur4
plt.scatter(4,0.1125*100, s=50,color='green',alpha=0.5, marker='^') #ur8

plt.axhline(75,color='red',linestyle='--', label='Human average performance on silhouettes')
plt.axhline(6.25,color='blue',linestyle='--', label = 'Random level')
plt.xticks([0,1,2,3,4,5], labels=['','hor','ver','ul','ur',''],fontsize=15)
plt.yticks(np.arange(0, 105, step=5),fontsize=15)
plt.xlabel('Direction of gratings',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.legend(loc='center',bbox_to_anchor=(0.5,1.07),frameon=False)
plt.show()
