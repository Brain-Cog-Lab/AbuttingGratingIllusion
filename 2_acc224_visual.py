'''
Robustness of high-resolution AG-MNIST
This file plot the accuracy against the epoch.
'''
import json
import matplotlib.pyplot as plt

def load_json(path = 'test.json'):
    with open(path, "r") as f:
        results = f.read()
    return json.loads(results)

def plot(ax, data, range, sign, label):
    ax.plot(range,data,sign,label=label)

def show(ax, xlabel, ylabel):
    #ax.set_title(title)
    ax.set_xlabel("epochs")
    #ax.set_ylabel(ylabel)
    ax.set_xticks(range(0,101,10))
    ax.set_yticks(range(0,101,10))
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    

    ax.grid() 


fig,axes = plt.subplots(1,4,figsize=(12,2))

ax1 = axes[0] 
ax2 = axes[1] 
ax3 = axes[2] 
ax4 = axes[3] 

#ax1.set_title('AlexNet')
#ax2.set_title('Vgg11(BN)')
#ax3.set_title('ResNet18')
#ax4.set_title('DenseNet121')
ax1.set_ylabel("acc(%)")


results1 = load_json(f'results/MNIST-224/acc_alexnet224_pretrainFalse_finetuneFalse.json')
results2 = load_json(f'results/MNIST-224/acc_vgg224_pretrainFalse_finetuneFalse.json')
results3 = load_json(f'results/MNIST-224/acc_resnet224_pretrainFalse_finetuneFalse.json')
results4 = load_json(f'results/MNIST-224/acc_densenet224_pretrainFalse_finetuneFalse.json')
length = 100

AGs= ['hor4','hor8','hor16','hor32',
      'ver4','ver8','ver16','ver32',
      'UL4','UL8','UL16','UL32',
      'UR4','UR8','UR16','UR32',]
for ax,results in [(ax1,results1), (ax2,results2), (ax3,results3), (ax4,results4)]:
    plot(ax,results['original'][0:length],range(1,length+1),'-','original')
    #plot(ax,results['hor4'][0:length],range(1,length+1),'-','hor4')
    #plot(ax,results['hor8'][0:length],range(1,length+1),'-','hor8')
    #plot(ax,results['hor16'][0:length],range(1,length+1),'-','hor16')
    #plot(ax,results['hor32'][0:length],range(1,length+1),'-','hor32')

    #plot(ax,results['ver4'][0:length],range(1,length+1),'-','ver4')
    #plot(ax,results['ver8'][0:length],range(1,length+1),'-','ver8')
    #plot(ax,results['ver16'][0:length],range(1,length+1),'-','ver16')
    #plot(ax,results['ver32'][0:length],range(1,length+1),'-','ver32')
    
    #plot(ax,results['UL4'][0:length],range(1,length+1),'-','UL4')
    #plot(ax,results['UL8'][0:length],range(1,length+1),'-','UL8')
    #plot(ax,results['UL16'][0:length],range(1,length+1),'-','UL16')
    #plot(ax,results['UL32'][0:length],range(1,length+1),'-','UL32')
   
    plot(ax,results['UR4'][0:length],range(1,length+1),'-','UR4')
    plot(ax,results['UR8'][0:length],range(1,length+1),'-','UR8')
    plot(ax,results['UR16'][0:length],range(1,length+1),'-','UR16')
    plot(ax,results['UR32'][0:length],range(1,length+1),'-','UR32')
 
    show(ax, "epochs", "acc(%)")
ax4.legend(loc='center',bbox_to_anchor=(1.25,0.5),frameon=False)
plt.show()


