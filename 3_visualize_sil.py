'''
Visualization of AG-silhouette samples
'''
from utils.dataloader import load_MNIST, load_SIL
from utils.visualize import save_image, save_image_matrix
from utils.abutting_grating_illusion import ag_distort_silhouette
import torch

datasets = load_SIL()
params = [(0.5,4,2),(0.5,6,3),(0.5,8,4),(0.5,10,5),(0.5,12,6),(0.5,14,7)]
for imgs, labels in datasets:
    for threshold, interval, phase in params:   
        #ag_images = imgs  
        ag_images = ag_distort_silhouette(imgs, threshold=threshold, interval=interval, phase=phase, direction=(0,1))   
        save_image(ag_images[0], f'ag_th{threshold}_int{interval}_ph{phase}_ver.png')
        #save_image(img[0], f'original.png')
    break

