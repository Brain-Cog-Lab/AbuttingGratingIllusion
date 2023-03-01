'''
Robustness of high-resolution AG-MNIST
This file trains models from scratch with the high-resolution MNIST trainset,
and test them with the high-resolution MNIST test set and the high-resolution AG-MNIST test set for each epoch during the training.
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
from utils.abutting_grating_illusion import ag_distort_224, transform_224
import json
from torchvision import models
import argparse
import re





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
            inputs = transform_224(inputs)
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
        original_acc = test_model(model, test_set, device)
        accs['original'].append(original_acc)        
        accs['hor4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(1,0), device=device))
        accs['hor8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(1,0), device=device))
        accs['hor16'].append(test_AG(model, test_set, threshold=0.5, interval=16, phase=8, direction=(1,0), device=device))
        accs['hor32'].append(test_AG(model, test_set, threshold=0.5, interval=32, phase=16, direction=(1,0), device=device))
        accs['ver4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(0,1), device=device))
        accs['ver8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(0,1), device=device))
        accs['ver16'].append(test_AG(model, test_set, threshold=0.5, interval=16, phase=8, direction=(0,1), device=device))
        accs['ver32'].append(test_AG(model, test_set, threshold=0.5, interval=32, phase=16, direction=(0,1), device=device))
        accs['UR4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(1,1), device=device))
        accs['UR8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(1,1), device=device))
        accs['UR16'].append(test_AG(model, test_set, threshold=0.5, interval=16, phase=8, direction=(1,1), device=device))
        accs['UR32'].append(test_AG(model, test_set, threshold=0.5, interval=32, phase=16, direction=(1,1), device=device))
        accs['UL4'].append(test_AG(model, test_set, threshold=0.5, interval=4, phase=2, direction=(-1,1), device=device))
        accs['UL8'].append(test_AG(model, test_set, threshold=0.5, interval=8, phase=4, direction=(-1,1), device=device))
        accs['UL16'].append(test_AG(model, test_set, threshold=0.5, interval=16, phase=8, direction=(-1,1), device=device))
        accs['UL32'].append(test_AG(model, test_set, threshold=0.5, interval=32, phase=16, direction=(-1,1), device=device))
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
            images = transform_224(images)
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
            images = ag_distort_224(images, threshold, interval, phase, direction)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                       
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return (100 * correct / total)

def set_parameter_requires_grad(model, feature_extracting):
    if use_pretrained:
        if feature_extracting:
            print("Using feature extraction")
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("Using finetuning")


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet()
        if use_pretrained:
            print("Load pretrained model")
            model_ft.load_state_dict(torch.load('./pretrained_models/imagenet_models/alexnet.pth'))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn()
        if use_pretrained:
            model_ft.load_state_dict(torch.load('./pretrained_models/imagenet_models/vgg11_bn.pth'))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121()
        if use_pretrained:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            state_dict = torch.load('./pretrained_models/imagenet_models/densenet121.pth')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model_ft.load_state_dict(state_dict)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neural networks')
    parser.add_argument('-model', default='resnet', type=str, help='Choose the network structure')
    parser.add_argument('-device', default='cuda:0', type=str, help='Choose the device')    
    parser.add_argument('--pretrain', action='store_true', default = False, help='Choose whether to use the pretrained model')
    parser.add_argument('--finetune', action='store_true', default = False, help='Choose whether to use the finetune or the feature extraction mode')
    args = parser.parse_args()
    print(f"Using {args.model}, pretrain:{args.pretrain}, finetune:{args.finetune}")
    model_name = args.model
    use_pretrained = args.pretrain
    if use_pretrained:
        print("Using pretrained model")
    feature_extract = not args.finetune

    num_classes = 10
    # fcnet
    accs = {'original':[],'hor4':[],'hor8':[],'hor16':[],'hor32':[],
                        'ver4':[],'ver8':[],'ver16':[],'ver32':[],
                        'UR4':[],'UR8':[],'UR16':[],'UR32':[],
                        'UL4':[],'UL8':[],'UL16':[],'UL32':[],}

    train_set = load_MNIST(train=True)
    test_set = load_MNIST(train=False)

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=feature_extract, use_pretrained=use_pretrained)

    train_model(model_ft, train_set, epochs=100, device=args.device, learning_rate=0.001)

    with open(f'acc_{model_name}224_pretrain{args.pretrain}_finetune{args.finetune}.json', "w") as f:
        f.write(json.dumps(accs,indent=4)) 


