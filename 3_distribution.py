'''
Robustness on AG-silhouette
This file plots the histograms of pretrained models on the AG-silhouettes
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

files = os.listdir('results/silhouettes')
fig,axes = plt.subplots(1,6,figsize=(18,3))
for i in range(6):
    acc_lst=[]
    for filename in files:
        results = pd.read_csv('results/silhouettes/'+filename)
        #for _ in results['silhouettes']:
        #    acc_lst.append(_*100)

        for _ in results[f'ag/ag_i{i*2+4}_ul']:
            if _ >0.2:
                print(filename)
            acc_lst.append(_*100)
    #axes[i].set_xlabel('Accuracy(%)')
    
    axes[i].set_xlim(0,50)
    axes[i].set_ylim(0,100)
    axes[i].set_xticks(range(0,55,5))
    axes[i].set_yticks(range(0,105,10))
    axes[i].hist(acc_lst, bins=range(0,52,2))
    axes[i].axvline([6.25],linestyle='dotted',color='red')
axes[0].set_ylabel('UL')
axes[0].set_title('I=4')
axes[1].set_title('I=6')
axes[2].set_title('I=8')
axes[3].set_title('I=10')
axes[4].set_title('I=12')
axes[5].set_title('I=14')
plt.show()