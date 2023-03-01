from matplotlib import pyplot as plt
import torch
from torchvision import utils
import os

def save_image(image, filename):  
    assert len(image.shape) == 3, "The image must have only three dimensions of C,W,H."
    utils.save_image(image, filename)

def save_image_matrix(images, filename, size):
    '''
    Tensor形式输入
    Date:2021.10.20
    Function: Save figures in the form of a matrix m*n
    Param:
        images: torch tensor, [B,C,W,H]
        filename: The filename
        size: tuple, the width and height of unified image
    '''
    for img in images:
        assert len(img.shape) == 3, "The image must have only three dimensions."
    if isinstance(images, list):
        C,W,H = images[0].shape
        _images = torch.zeros(len(images),C,W,H)
        for i in range(len(images)):
            _images[i] = images[i]
        images = _images
    matrix = []
    m, n = size
    assert len(images) >= m*n, "The number of images is less than m*n."
    for i in range(n):
        line = []
        for j in range(m):
            line.append(images[i*m+j])
        matrix.append(torch.cat(line,1))
    utils.save_image(torch.cat(matrix, 2), filename)

