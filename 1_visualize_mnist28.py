'''
Visualization of AG-MNIST samples
'''

from utils.dataloader import load_MNIST
from utils.visualize import save_image, save_image_matrix
from utils.abutting_grating_illusion import ag_distort_28
import torch

datasets = load_MNIST()
params = [(0.5,2,1),(0.5,4,2),(0.5,6,3),(0.5,8,4)]
for imgs, labels in datasets:
    for threshold, interval, phase in params:
        img_lst = torch.ones(10,1,29,29)
        #ag_images = imgs
        ag_images = ag_distort_28(imgs, threshold=threshold, interval=interval, phase=phase, direction=(1,0))
        for ag_image, label in zip(ag_images,labels):
            if img_lst[label][0][0][0] == 1:
                img_lst[label,0,0:28,0:28] = ag_image
        save_image_matrix(img_lst, f'ag_th{threshold}_int{interval}_ph{phase}.png', (1,10))
        #save_image_matrix(img_lst, f'original.png', (1,10))
    break

