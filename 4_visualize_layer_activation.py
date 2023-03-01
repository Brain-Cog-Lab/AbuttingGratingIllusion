'''
This file visualizes the average activation of ResNet50's early layers.
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
import cv2
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

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


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




random_model = resnet50().cuda()
imagenet_model = resnet50(pretrained=True).cuda()
#SIN_model = load_model('resnet50_trained_on_SIN')
#cutout_model = load_model('resnet50_cutout')
#cutmix_model = load_model('resnet50_cutmix')
#mixup_model = load_model('resnet50_mixup')
augmix_model = load_model('resnet50_augmix')
#ant_model = load_model('ANT_SIN')
deepaugment_model = load_model('resnet50_deepaugment')
augmix_deepaugment_model = load_model('resnet50_deepaugment_augmix')





name = 'bottle/bottle2'
imgs = [f'./datasets/SIL/silhouettes/{name}.png'] + [f'./datasets/SIL/ag/ag_i{i}_hor/{name}.png' for i in range(4,16,2)]


params = [('random',random_model),
          ('imagenet',imagenet_model),
          #('SIN',SIN_model),
          #('cutout',cutout_model),
          #('mixup',mixup_model),
          ('augmix',augmix_model),
          ('deepaugment',deepaugment_model),
          ('augmix_deepaugment',augmix_deepaugment_model),]

target_layers = ['conv1','bn1','relu','maxpool']


with torch.no_grad():
    result_imgs = []
    for model_name, model in params:  
        print(model_name)
        for img_path in imgs:
            img = Image.open(img_path)
            #img = Image.open('./datasets/SIL/ag/ag_i4_hor/bird/bird9.png')
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            #transform = transforms.Compose([transforms.ToTensor()])
            rgb_img = transforms.ToTensor()(img).detach().cpu().numpy()
            rgb_img = np.transpose(rgb_img, (1,2,0))


            input_tensor = transform(img).unsqueeze(0)
            #kernel_i = 47
            if model_name in ['random', 'imagenet']:
                model.target_layers = [getattr(model, target_layer) for target_layer in target_layers]

            elif model_name in ['augmix', 'deepaugment', 'augmix_deepaugment']:
                model.target_layers = [getattr(model.module, target_layer) for target_layer in target_layers]
            #model.target_layers = [model.module.stem[0],model.module.stem[1]]
            _x = input_tensor.cuda()
            for target_layer in model.target_layers:
                print(target_layer)
                _x = target_layer(_x)              

            x = torch.mean(_x, (0,1)).cpu().detach().numpy()
            #x = _x[0][_].cpu().detach().numpy()
            #x = x[0][47].cpu().detach().numpy()
            x = cv2.resize(x, (224, 224))
                
            x = x-np.min(x)

            x = x/(np.max(x)+1e-9)

            grayscale_cam = x

            # In this example grayscale_cam has only one image in the batch:

            #grayscale_cam = grayscale_cam[0, :]

            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            visualization = torch.Tensor(visualization).permute(2,0,1)/255.

            result_imgs.append(visualization)
        #save_image_matrix(result_imgs, f'layer1.png', (8,8))
    save_image_matrix(result_imgs, f'maxpool.png', (len(result_imgs)//len(params),len(params)))
        


