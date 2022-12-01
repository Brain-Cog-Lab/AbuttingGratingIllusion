'''
Robustness of AG-MNIST
This file plot the accuracy against the epoch.
'''

import json
import matplotlib.pyplot as plt

def load_json(path = 'test.json'):
    with open(path, "r") as f:
        results = f.read()
    return json.loads(results)

def plot(ax, data, range, label):
    ax.plot(range,data,'-',label=label)
def show(ax,title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0,101,10))
    ax.set_yticks(range(0,101,10))
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    #plt.legend(loc='center',bbox_to_anchor=(1.10,0.5),frameon=False)
    ax.grid()


fig,axes = plt.subplots(1,2)
ax1 = axes[0] 
ax2 = axes[1] 
fc_results = load_json('results/MNIST-28/acc_fc28.json')
conv_results = load_json('results/MNIST-28/acc_conv28.json')
length = 100
plot(ax1, fc_results['original'][0:length],range(1,length+1),'original')
plot(ax1, fc_results['hor2'][0:length],range(1,length+1),'hor2')
plot(ax1, fc_results['hor4'][0:length],range(1,length+1),'hor4')
plot(ax1, fc_results['hor6'][0:length],range(1,length+1),'hor6')
plot(ax1, fc_results['hor8'][0:length],range(1,length+1),'hor8')
plot(ax2, conv_results['original'][0:length],range(1,length+1),'original')
plot(ax2, conv_results['hor2'][0:length],range(1,length+1),'hor2')
plot(ax2, conv_results['hor4'][0:length],range(1,length+1),'hor4')
plot(ax2, conv_results['hor6'][0:length],range(1,length+1),'hor6')
plot(ax2, conv_results['hor8'][0:length],range(1,length+1),'hor8')
'''
plot(results['ver2'][0:length],range(1,length+1),'ver2')
plot(results['ver4'][0:length],range(1,length+1),'ver4')
plot(results['ver6'][0:length],range(1,length+1),'ver6')
plot(results['ver8'][0:length],range(1,length+1),'ver8')
plot(results['UL2'][0:length],range(1,length+1),'UL2')
plot(results['UL4'][0:length],range(1,length+1),'UL4')
plot(results['UL6'][0:length],range(1,length+1),'UL6')
plot(results['UL8'][0:length],range(1,length+1),'UL8')
plot(results['UR2'][0:length],range(1,length+1),'UR2')
plot(results['UR4'][0:length],range(1,length+1),'UR4')
plot(results['UR6'][0:length],range(1,length+1),'UR6')
plot(results['UR8'][0:length],range(1,length+1),'UR8')
'''
show(ax1,"Accuracy of FC against AG-MNIST", "epochs", "acc(%)")
show(ax2,"Accuracy of CNN against AG-MNIST", "epochs", "acc(%)")
plt.legend(loc='center',bbox_to_anchor=(1.17,0.5),frameon=False)
ax2.set_ylabel("")
plt.show()