'''
Robustness on AG-silhouette
This file tests the performance of pretrained models on the AG-silhouette.
'''
import os
import sys
from collections import OrderedDict
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo
from utils.dataloader import load_SIL

from utils.probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
import numpy as np
from torchvision import transforms
import json
import pandas as pd
import timm
import time


torchvision_models = ['alexnet',
                      'vgg11','vgg13','vgg16','vgg19',
                      'vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn',
                      'resnet18','resnet34','resnet50','resnet101','resnet152',
                      'squeezenet1_0','squeezenet1_1',
                      'densenet121','densenet161','densenet169','densenet201',
                      'inception_v3',
                      'googlenet',
                      'shufflenet_v2_x0_5','shufflenet_v2_x1_0',
                      'mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small',
                      'resnext50_32x4d','resnext101_32x8d',
                      'wide_resnet50_2','wide_resnet101_2',
                      'mnasnet0_5','mnasnet1_0',
                      'efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7',
                      'regnet_y_400mf','regnet_y_800mf','regnet_y_1_6gf','regnet_y_3_2gf','regnet_y_8gf','regnet_y_16gf','regnet_y_32gf','regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf','regnet_x_8gf','regnet_x_16gf','regnet_x_32gf',
                      'vit_b_16','vit_b_32','vit_l_16','vit_l_32',
                      'convnext_tiny','convnext_small','convnext_base','convnext_large',
                      ]
timm_models = ['convnext_tiny', 'convnext_small', 
               'convnext_base', 'convnext_base_in22ft1k', 'convnext_base_384_in22ft1k',
               'convnext_large', 'convnext_large_in22ft1k', 'convnext_large_384_in22ft1k',
               'convnext_xlarge_in22ft1k', 'convnext_xlarge_384_in22ft1k',

               'mixer_b16_224', 'mixer_b16_224_miil', 'mixer_l16_224',

               'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
               'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224',


               'inception_v3','inception_v4',
               'adv_inception_v3','ens_adv_inception_resnet_v2','gluon_inception_v3','inception_resnet_v2',
]           
            
            
            
data_aumgentation_models = ['resnet50_trained_on_SIN', 'resnet50_trained_on_SIN_and_IN', 
                            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN', 'alexnet_trained_on_SIN', 'vgg16_trained_on_SIN',
                            'Speckle','Gauss_mult','Gauss_sigma_0.5',
                            'ANT','ANT_SIN','ANT3x3','ANT3x3_SIN',
                            'resnet50_cutmix','resnet50_feature_cutmix',
                            'resnet101_cutmix','resnet152_cutmix','resnext_cutmix',
                            'resnet50_cutout',
                            'resnet50_mixup','resnet50_manifold_mixup',
                            'resnet50_augmix', 'resnet50_deepaugment',
                            'resnet50_deepaugment_augmix','resnext101_32x8d_deepaugment_augmix',

                            ]
















def load_model(model_name):
    model_urls = {
            'resnet50_trained_on_SIN': 'resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'alexnet_trained_on_SIN': 'alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
            'vgg16_trained_on_SIN': 'vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar'
    }
    SIN_models = ["resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_IN", "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "vgg16_trained_on_SIN", "alexnet_trained_on_SIN"]
    if model_name in SIN_models:
        if "resnet50" in model_name:
            print("Using the ResNet50 architecture.")
            model = torchvision.models.resnet50(pretrained=False)
            model = torch.nn.DataParallel(model).cuda()
            checkpoint = model_zoo.load_url(model_urls[model_name])
        elif "vgg16" in model_name:
            print("Using the VGG-16 architecture.")
            # download model from URL manually and save to desired location
            model = torchvision.models.vgg16(pretrained=False)
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            checkpoint = model_zoo.load_url(model_urls[model_name])
        elif "alexnet" in model_name:
            print("Using the AlexNet architecture.")
            model = torchvision.models.alexnet(pretrained=False)
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            checkpoint = model_zoo.load_url(model_urls[model_name])
        model.load_state_dict(checkpoint["state_dict"])
    elif model_name in ['resnet50_cutmix','resnet50_feature_cutmix', 'resnet101_cutmix']:
        checkpoint = torch.load(f'./pretrained_models/data_augmentation_models/cutmix/{model_name}.tar')   
        if 'resnet50' in model_name:
            model = torchvision.models.resnet50()
        elif 'resnet101' in model_name:
            model = torchvision.models.resnet101()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict']) 
    elif model_name == 'resnet152_cutmix':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/cutmix/resnet152_cutmix.pth')   
        model = torchvision.models.resnet152()
        model.load_state_dict(checkpoint) 
        model = torch.nn.DataParallel(model).cuda()   
    elif model_name == 'resnext_cutmix':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/cutmix/resnext_cutmix.pth.tar')   
        model = timm.create_model('resnext101_32x4d')
        model.load_state_dict({k.replace('module.',''):checkpoint[k] for k in checkpoint}) 
        model = torch.nn.DataParallel(model).cuda()   
    elif model_name in ['resnet50_mixup','resnet50_manifold_mixup']:
        checkpoint = torch.load(f'./pretrained_models/data_augmentation_models/mixup/{model_name}.tar')   
        model = torchvision.models.resnet50()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])                
    elif model_name == 'resnet50_cutout':
        checkpoint = torch.load(f'./pretrained_models/data_augmentation_models/resnet50_cutout.tar')   
        model = torchvision.models.resnet50()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])  

    elif model_name == 'resnet50_augmix':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/resnet50_augmix.tar')   
        arch = checkpoint['arch']
        model = torchvision.models.__dict__[arch]()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])          
    elif model_name == 'resnet50_deepaugment':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/deepaugment/deepaugment.pth.tar')   
        arch = checkpoint['arch']
        model = torchvision.models.__dict__[arch]()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])     
    elif model_name == 'resnet50_deepaugment_augmix':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/deepaugment/deepaugment_and_augmix.pth.tar')   
        arch = checkpoint['arch']
        model = torchvision.models.__dict__[arch]()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict']) 
    elif model_name == 'resnext101_32x8d_deepaugment_augmix':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/deepaugment/resnext101_augmix_and_deepaugment.pth.tar')   
        arch = checkpoint['arch']
        model = torchvision.models.__dict__[arch]()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])  
    elif model_name == 'adv':
        checkpoint = torch.load('./pretrained_models/data_augmentation_models/adversarial_imagenet_model_weights_2px.pth.tar')   
        arch = checkpoint['arch']
        model = torchvision.models.__dict__[arch]()
        model = torch.nn.DataParallel(model).cuda()   
        model.load_state_dict(checkpoint['state_dict'])  
    elif model_name == '21k':
        model = timm.create_model('vit_base_patch16_224_miil', pretrained=True)
        model = torch.nn.DataParallel(model).cuda()   
    elif model_name in ['ANT','ANT_SIN','ANT3x3','ANT3x3_SIN','Gauss_mult','Gauss_sigma_0.5','Speckle']:
        model = torchvision.models.resnet50()
        checkpoint = torch.load(f'./pretrained_models/data_augmentation_models/game_of_noise/{model_name}_Model.pth') 
        model.load_state_dict(checkpoint['model_state_dict'])
        model = torch.nn.DataParallel(model).cuda() 
   
    elif model_name in timm_models:
        model = timm.create_model(model_name, pretrained=True)
        model = torch.nn.DataParallel(model).cuda()

    '''    
    elif model_name in torchvision_models:
        model = getattr(torchvision.models,model_name)(pretrained=True)
        model = torch.nn.DataParallel(model).cuda()

    '''
    model.eval()    
    return model


if __name__ == "__main__":
    start_time = time.time()
    results = {}
    normalize = torchvision.transforms.Compose([torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datasets = ['silhouettes'] + [f'ag/ag_i{i}_hor' for i in range(2,34,2)]\
                               + [f'ag/ag_i{i}_ver' for i in range(2,34,2)]\
                               + [f'ag/ag_i{i}_ul' for i in range(2,34,2)]\
                               + [f'ag/ag_i{i}_ur' for i in range(2,34,2)]    


    models = data_aumgentation_models




    for model_name in models:
        print('#######################')
        print('MODEL:',model_name)
        if model_name not in results:
            results[model_name] = {}

        model = load_model(model_name) # change to different model as desired
        model.eval()

        mapping = ImageNetProbabilitiesTo16ClassesMapping()
        for dataset_name in datasets:
            print(f'Using {dataset_name}')
            dataset = load_SIL(dataset_name)
            total = 0
            correct = 0
            for imgs, labels in dataset:
                imgs = 1- imgs
                imgs = normalize(imgs)
                outputs = model(imgs)
                softmax_output_numpy = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy() # replace with conversion
                decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_numpy[0])
                total += 1 # labels.size(0)  
                correct += (decision_from_16_classes == labels)
            acc = correct/float(total)
            print(acc)
            results[model_name][dataset_name] = acc
    print(time.time()-start_time,'s')    

    df = pd.DataFrame(index = models, columns = datasets)
    for model_name in models:
        df.loc[model_name] = results[model_name]
    df.to_csv('test.csv')


