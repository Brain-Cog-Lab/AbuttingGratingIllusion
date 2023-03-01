<<<<<<< HEAD
# Challenging Deep Learning Models with Image Distortion based on the Abutting Grating Illusion
This repository contains code from our paper [Challenging Deep Learning Models with Image Distortion based on the Abutting Grating Illusion] published in Cell Patterns. https://www.cell.com/patterns/fulltext/S2666-3899(23)00026-0
=======

# This repository has been moved to the following link as part of the Brain-Cog-Lab:
https://github.com/Brain-Cog-Lab/AbuttingGratingDistortion
>>>>>>> 3fc53a687f97ee5c17082f217e3c9f302c2a92dc

If you use our code or refer to this project, please cite this paper: Jinyu Fan, and Yi Zeng. Challenging Deep Learning Models with Image Distortion based on the Abutting Grating Illusion. Patterns, DOI：https://doi.org/10.1016/j.patter.2023.100695

## Paper Introduction
Even state-of-the-art deep learning models lack fundamental abilities compared with humans. While many image distortions have been proposed to compare deep learning with humans, they depend on mathematical transformations instead of human cognitive functions. Illusory contour has been a popular topic for psychology research, providing insights into visual perception of both humans and various animal species. However, illusory contour perception is seldom studied in deep learning research. The crucial reason is that illusory contour samples are unnatural visual stimuli designed manually, which cannot be directly integrated into the datasets. This paper aims to create illusory contour datasets by distorting existing datasets in deep learning research. To be more specific, we propose an image distortion based on the abutting grating illusion, which is a classic illusory contour phenomenon.
![gr1_lrg](https://user-images.githubusercontent.com/48897111/222035741-8944ffec-935f-4594-8128-14777f5b8c06.jpg)

The distortion generates illusory contour perception using line gratings abutting each other. We applied the method to MNIST, high-resolution MNIST, and “16-class-ImageNet” silhouettes. Many models, including models trained from scratch and 109 models pretrained with ImageNet or various data augmentation techniques, were tested. Our results show that abutting grating distortion is challenging even for state-of-the-art deep learning models. We discovered that DeepAugment models outperformed other pretrained models. Visualization of early layers indicates that better-performing models exhibit the endstopping property, which is consistent with neuroscience discoveries. Twenty-four human subjects classified distorted samples to validate the distortion.
![image](https://user-images.githubusercontent.com/48897111/205000986-b5b2f85c-8720-4731-b637-f427faaebe9b.png)
![image](https://user-images.githubusercontent.com/48897111/205001093-6151814b-d4b1-4eec-9dc7-b989340ad076.png)
![image](https://user-images.githubusercontent.com/48897111/205001212-892814cd-a668-4bc0-b9ec-982f022e8d6f.png)

## Run
### Code of Abutting Grating Distortion
The code to generate Abutting Grating Distortion can be found in "utils/abutting_grating_illusions.py"

### Visualize samples of AG-MNIST, high-resolution AG-MNIST, AG-silhouette
The following files generate samples of AG distortion
AG-MNIST : 1_visualize_mnist28.py
high-resolution AG-MNIST : 2_visualize_mnist224.py
AG-silhouette ：3_visualize_sil.py




### Train and test model from scratch
"1_test_mnist28.py" trains models with original MNIST and test them on AG-MNIST.
"2_test_mnist224.py" trains models with high-resolution MNIST and test them on high-resolution AG-MNIST.

"1_acc28_visual.py" and "2_acc224_visual.py" plot the accuracy against the epochs during the training.

### Testing pretrained models
"3_imagenet_pretrained_test.py" tests pretrained models on the AG-silhouette.
Note that the pretrained models we tested are from torchvision and timm packages.
We also tested data augmentation models from GitHub, but we do not include them in this repository.
You can find the data augmentation models below:
https://github.com/hendrycks/imagenet-r
https://github.com/clovaai/CutMix-PyTorch
https://github.com/bethgelab/game-of-noise
https://github.com/google-research/augmix
https://github.com/rgeirhos/texture-vs-shape

"3_distribution.py" plots the distribution of models' accuracy.
"3_best_performing_models.py" plots the accuracy of best-performing models, which we defined as the outliners in the distributions.

### Visualization of activation
"4_visualize_layer_activation.py" visualizes the average activation of ResNet50 when confronting with the abutting grating samples.

"4_visualize_filter_activation.py" visualizes the globally normalized activation of each filter.

"4_visualising_features.py" visualizes the features of first convolution layer of ResNet50.


### Human experiments
In the "human-experiments" folder we provide the code for the human experiments.

