'''
Visualization of high-resolution AG-MNIST samples
'''
from utils.dataloader import load_MNIST
from utils.visualize import save_image, save_image_matrix
from utils.abutting_grating_illusion import ag_distort_224, transform_224
import torch

datasets = load_MNIST()
params = [(0.5,4,2),(0.5,8,4),(0.5,16,8),(0.5,32,16)]
for imgs, labels in datasets:
    for threshold, interval, phase in params:    
        ag_images = transform_224(imgs) 
        ag_images = ag_distort_224(imgs, threshold=threshold, interval=interval, phase=phase, direction=(0,1))
        #save_image(img[0], f'original.png')  
        save_image(ag_images[0], f'ag_th{threshold}_int{interval}_ph{phase}_ver.png')
    break

