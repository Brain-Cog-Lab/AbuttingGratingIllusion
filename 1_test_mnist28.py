'''
Robustness of AG-MNIST
This file trains models from scratch with the original MNIST trainset,
and test them with the original MNIST test set and the AG-MNIST test set for each epoch during the training.

'''
from utils.dataloader import load_MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import random
from utils.abutting_grating_illusion import ag_distort_28
import json


class FCNet(nn.Module):
    def __init__(self, layers=[784,100,100,10]):
        nn.Module.__init__(self)
        assert layers[0] == 784 and layers[-1] == 10, "The input layer has to be 784 while the output has to be 10."
        self.fc = []
        for i in range(len(layers)-1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        self.sequential = nn.Sequential(*self.fc)


    def forward(self, x):
        x = x.view(-1, 28*28)
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))
        x = self.fc[-1](x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_set, epochs, device, learning_rate=0.001):
    start_time = time.time()
    # The path and the name of the model you would like to save
    torch.autograd.set_detect_anomaly(True)
    model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_set, 0):
            batch_start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            #print(f"Batch time:{time.time() - batch_start_time}")
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('Accuracy of the network on the 60000 train images: %d %%' % (
            100 * correct / total))
        accs['original'].append(test_model(model, test_set, device))
        accs['hor2'].append(test_AG(model, test_set, threshold=0.5, interval=2, phase=1, direction=(1,0), device=device))
        accs['hor4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(1,0), device=device))
        accs['hor6'].append(test_AG(model, test_set, threshold=0.5, interval=6, phase=3, direction=(1,0), device=device))
        accs['hor8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(1,0), device=device))
        accs['ver2'].append(test_AG(model, test_set, threshold=0.5, interval=2, phase=1, direction=(0,1), device=device))
        accs['ver4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(0,1), device=device))
        accs['ver6'].append(test_AG(model, test_set, threshold=0.5, interval=6, phase=3, direction=(0,1), device=device))
        accs['ver8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(0,1), device=device))
        accs['UR2'].append(test_AG(model, test_set, threshold=0.5, interval=2, phase=1, direction=(1,1), device=device))
        accs['UR4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(1,1), device=device))
        accs['UR6'].append(test_AG(model, test_set, threshold=0.5, interval=6, phase=3, direction=(1,1), device=device))
        accs['UR8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(1,1), device=device))
        accs['UL2'].append(test_AG(model, test_set, threshold=0.5, interval=2, phase=1, direction=(-1,1), device=device))
        accs['UL4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(-1,1), device=device))
        accs['UL6'].append(test_AG(model, test_set, threshold=0.5, interval=6, phase=3, direction=(-1,1), device=device))
        accs['UL8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(-1,1), device=device))
    print('Finished Training')
    print(time.time() - start_time)
    return model

def test_model(model, dataset, device):
    model.to(device)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                     
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return (100 * correct / total)

def test_AG(model, dataset, threshold, interval, phase, direction, device):
    model.to(device)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset, 0):
            images, labels = data[0].to(device), data[1].to(device)
            images = ag_distort_28(images, threshold, interval, phase, direction)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                       
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return (100 * correct / total)



if __name__ == "__main__":
    
    # fcnet
    accs = {'original':[],'hor2':[],'hor4':[],'hor6':[],'hor8':[],
                          'ver2':[],'ver4':[],'ver6':[],'ver8':[],
                          'UR2':[],'UR4':[],'UR6':[],'UR8':[],
                          'UL2':[],'UL4':[],'UL6':[],'UL8':[],}
    train_set = load_MNIST(train=True)
    test_set = load_MNIST(train=False)
    model = FCNet()
    train_model(model, train_set, epochs=100, device='cuda:0', learning_rate=0.001)
    with open('acc_fc28.json', "w") as f:
        f.write(json.dumps(accs,indent=4)) 
    '''
    #convnet
    accs = {'original':[],'hor2':[],'hor4':[],'hor6':[],'hor8':[],
                          'ver2':[],'ver4':[],'ver6':[],'ver8':[],
                          'UR2':[],'UR4':[],'UR6':[],'UR8':[],
                          'UL2':[],'UL4':[],'UL6':[],'UL8':[],}
    train_set = load_MNIST(train=True)
    test_set = load_MNIST(train=False)
    model = ConvNet()
    train_model(model, train_set, epochs=100, device='cuda:0', learning_rate=0.001)
    with open('acc_conv28.json', "w") as f:
        f.write(json.dumps(accs,indent=4)) 

    '''