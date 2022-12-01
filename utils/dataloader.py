import torch
from torchvision import datasets, transforms
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from PIL import Image
import torchvision.transforms as transforms

import numpy as np


def load_MNIST(train = False, batch_size = 100):
    path = './datasets' # might need to change based on where to call this function
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  
    transform = transforms.Compose([transforms.ToTensor()])
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=False, transform=transform),
                batch_size=batch_size, shuffle=False)
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=False, transform=transform),
                batch_size=batch_size, shuffle=False)
        return test_loader





def load_SIL(type='silhouettes'):
    '''
    Date:2021.9.9
    Function: load MNIST dataset
    Param:

    '''
    path = './datasets/SIL/'+type # might need to change based on where to call this function
    labels = os.listdir(path)
    datasets = []
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = transforms.Compose([transforms.ToTensor()])
    for label in labels:
        for img_name in os.listdir(f"{path}/{label}"):
            img_path = f"{path}/{label}/{img_name}"
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)

            datasets.append((img, label))
    return datasets




