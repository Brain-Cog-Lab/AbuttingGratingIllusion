# AbuttingGratingIllusion
The code for "Abutting Grating Illusion: Cognitive Challenge to Deep Learning Models"

## Code of Abutting Grating Distortion
The code to generate Abutting Grating Distortion can be found in "utils/abutting_grating_illusions.py"

## Visualize samples of AG-MNIST, high-resolution AG-MNIST, AG-silhouette
The following files generate samples of AG distortion
AG-MNIST : 1_visualize_mnist28.py
high-resolution AG-MNIST : 2_visualize_mnist224.py
AG-silhouette ：3_visualize_sil.py

## Train and test model from scratch
"1_test_mnist28.py" trains models with original MNIST and test them on AG-MNIST.
"2_test_mnist224.py" trains models with high-resolution MNIST and test them on high-resolution AG-MNIST.

"1_acc28_visual.py" and "2_acc224_visual.py" plot the accuracy against the epochs during the training.

## Testing pretrained models
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

## Visualization of activation
"4_visualize_layer_activation.py" visualizes the average activation of ResNet50 when confronting with the abutting grating samples.

"4_visualize_filter_activation.py" visualizes the globally normalized activation of each filter.

"4_visualising_features.py" visualizes the features of first convolution layer of ResNet50.


## Human experiments
In the "human-experiments" folder we provide the code for the human experiments.
