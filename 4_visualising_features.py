'''
Visualize the convolutional filters of the ResNet50's first layer
'''
from queue import Full
from matplotlib import image
from torchvision.models import resnet50, vit_b_16, alexnet, vgg11
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import model_zoo

from torchvision import utils

def save_image_matrix(image_list, filename, size):
    '''
    列表形式输入
    Date:2021.9.10
    Function: Save a list of figure in the form of a matrix m*n
    Param:
        image_list: a list of images, the images has to be two dims.
        filename: The filename
        size: the width and height of unified image
    '''

    matrix = []
    m, n = size
    assert len(image_list) >= m*n, "The number of images is less than m*n."
    for i in range(n):
        matrix.append(torch.cat(image_list[i*m:i*m+m],2))
    utils.save_image(torch.cat(matrix, 1), filename)



random_model = resnet50()
#random_model.target_layers = [getattr(random_model, target_layer)] #######
random_model.eval()

imagenet_model = resnet50(pretrained=True)
#imagenet_model.target_layers = [getattr(imagenet_model, target_layer)] #########
imagenet_model.eval()
  
checkpoint = torch.load('./pretrained_models/data_augmentation_models/resnet50_augmix.tar')   
arch = checkpoint['arch']
augmix_model = torchvision.models.__dict__[arch]()
augmix_model = torch.nn.DataParallel(augmix_model).cuda()   
augmix_model.load_state_dict(checkpoint['state_dict']) 
#augmix_model.target_layers = [getattr(augmix_model.module, target_layer)] #######
augmix_model.eval()


SIN_model = torchvision.models.resnet50(pretrained=False)
SIN_model = torch.nn.DataParallel(SIN_model).cuda()
checkpoint = model_zoo.load_url('resnet50_train_60_epochs-c8e5653e.pth.tar')
SIN_model.load_state_dict(checkpoint["state_dict"])


checkpoint = torch.load('./pretrained_models/data_augmentation_models/deepaugment/deepaugment.pth.tar')   
arch = checkpoint['arch']
deepaugment_model = torchvision.models.__dict__[arch]()
deepaugment_model = torch.nn.DataParallel(deepaugment_model).cuda()   
deepaugment_model.load_state_dict(checkpoint['state_dict']) 
#deepaugment_model.target_layers = [getattr(deepaugment_model.module, target_layer)] ############
deepaugment_model.eval()

checkpoint = torch.load('./pretrained_models/data_augmentation_models/deepaugment/deepaugment_and_augmix.pth.tar')   
arch = checkpoint['arch']
augmix_deepaugment_model = torchvision.models.__dict__[arch]()
augmix_deepaugment_model = torch.nn.DataParallel(augmix_deepaugment_model).cuda()   
augmix_deepaugment_model.load_state_dict(checkpoint['state_dict']) 
#augmix_deepaugment_model.target_layers = [getattr(augmix_deepaugment_model.module, target_layer)] #########
augmix_deepaugment_model.eval()


kernels = []
weights = augmix_model.module.conv1.weight
weights = weights - torch.min(weights)
weights = weights/torch.max(weights)
for _ in range(64):
    _kernel = torch.zeros(3,8,8)
    for x in range(3):
        for y in range(7):
            for z in range(7):
                _kernel[x][y][z] = weights[_][x][y][z]
    kernels.append(_kernel)
save_image_matrix(kernels, 'augmix_features.png', (8,8))