'''
Robustness on AG-silhouette
This file plots the histogram of best-performing pretrained models' performance on the AG-silhouette.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
fig,axes = plt.subplots(1,5,figsize=(24,3))
files = os.listdir('results/silhouettes')


results = pd.read_csv('results/silhouettes/data_augmentation_results5.csv')
lst = []
i=1
lst.append(results['silhouettes'][i]*100)
lst.append(results['ag/ag_i4_hor'][i]*100)
lst.append(results['ag/ag_i4_ver'][i]*100)
lst.append(results['ag/ag_i4_ul'][i]*100)
lst.append(results['ag/ag_i4_ur'][i]*100)
lst.append(results['ag/ag_i6_hor'][i]*100)
lst.append(results['ag/ag_i6_ver'][i]*100)
lst.append(results['ag/ag_i6_ul'][i]*100)
lst.append(results['ag/ag_i6_ur'][i]*100)
axes[0].set_ylabel('Accuracy(%)')
axes[0].set_xlabel('Gratings')
axes[0].set_title('resnet50_deepaugment')
axes[0].set_ylim(0,100)

axes[0].bar(['orig','hor4','ver4','ul4','ur4','hor6','ver6','ul6','ur6'],lst, color=['black','r','g','b','orange','r','g','b','orange'])
for _ in range(9):
    axes[0].text(_-0.25,lst[_],int(lst[_]), fontsize=10)



lst = []
i=2
lst.append(results['silhouettes'][i]*100)
lst.append(results['ag/ag_i4_hor'][i]*100)
lst.append(results['ag/ag_i4_ver'][i]*100)
lst.append(results['ag/ag_i4_ul'][i]*100)
lst.append(results['ag/ag_i4_ur'][i]*100)
lst.append(results['ag/ag_i6_hor'][i]*100)
lst.append(results['ag/ag_i6_ver'][i]*100)
lst.append(results['ag/ag_i6_ul'][i]*100)
lst.append(results['ag/ag_i6_ur'][i]*100)
axes[1].set_title('resnet50_deepaugment_augmix')
axes[1].set_ylim(0,100)
axes[1].set_xlabel('Gratings')
axes[1].bar(['orig','hor4','ver4','ul4','ur4','hor6','ver6','ul6','ur6'],lst, color=['black','r','g','b','orange','r','g','b','orange'])
for _ in range(9):
    axes[1].text(_-0.25,lst[_],int(lst[_]), fontsize=10)



lst = []
i=3
lst.append(results['silhouettes'][i]*100)
lst.append(results['ag/ag_i4_hor'][i]*100)
lst.append(results['ag/ag_i4_ver'][i]*100)
lst.append(results['ag/ag_i4_ul'][i]*100)
lst.append(results['ag/ag_i4_ur'][i]*100)
lst.append(results['ag/ag_i6_hor'][i]*100)
lst.append(results['ag/ag_i6_ver'][i]*100)
lst.append(results['ag/ag_i6_ul'][i]*100)
lst.append(results['ag/ag_i6_ur'][i]*100)
axes[2].set_title('resnext101_32x8d_deepaugment_augmix')
axes[2].set_ylim(0,100)
axes[2].set_xlabel('Gratings')
axes[2].bar(['orig','hor4','ver4','ul4','ur4','hor6','ver6','ul6','ur6'],lst, color=['black','r','g','b','orange','r','g','b','orange'])
for _ in range(9):
    axes[2].text(_-0.25,lst[_],int(lst[_]), fontsize=10)


results = pd.read_csv('results/silhouettes/timm_models_results3.csv')
lst = []
i=1
lst.append(results['silhouettes'][i]*100)
lst.append(results['ag/ag_i4_hor'][i]*100)
lst.append(results['ag/ag_i4_ver'][i]*100)
lst.append(results['ag/ag_i4_ul'][i]*100)
lst.append(results['ag/ag_i4_ur'][i]*100)
lst.append(results['ag/ag_i6_hor'][i]*100)
lst.append(results['ag/ag_i6_ver'][i]*100)
lst.append(results['ag/ag_i6_ul'][i]*100)
lst.append(results['ag/ag_i6_ur'][i]*100)

axes[3].set_title('convnext_xlarge_384_in22ft1k')
axes[3].set_ylim(0,100)
axes[3].set_xlabel('Gratings')
axes[3].bar(['orig','hor4','ver4','ul4','ur4','hor6','ver6','ul6','ur6'],lst, color=['black','r','g','b','orange','r','g','b','orange'])
for _ in range(9):
    axes[3].text(_-0.25,lst[_],int(lst[_]), fontsize=10)


results = pd.read_csv('results/silhouettes/torchvision_models_results4.csv')
lst = []
i=7
lst.append(results['silhouettes'][i]*100)
lst.append(results['ag/ag_i4_hor'][i]*100)
lst.append(results['ag/ag_i4_ver'][i]*100)
lst.append(results['ag/ag_i4_ul'][i]*100)
lst.append(results['ag/ag_i4_ur'][i]*100)
lst.append(results['ag/ag_i6_hor'][i]*100)
lst.append(results['ag/ag_i6_ver'][i]*100)
lst.append(results['ag/ag_i6_ul'][i]*100)
lst.append(results['ag/ag_i6_ur'][i]*100)
axes[4].set_title('efficientnet_b4')
axes[4].set_ylim(0,100)
axes[4].set_xlabel('Gratings')
axes[4].bar(['orig','hor4','ver4','ul4','ur4','hor6','ver6','ul6','ur6'],lst, color=['black','r','g','b','orange','r','g','b','orange'])
for _ in range(9):
    axes[4].text(_-0.25,lst[_],int(lst[_]), fontsize=10)
plt.show()
