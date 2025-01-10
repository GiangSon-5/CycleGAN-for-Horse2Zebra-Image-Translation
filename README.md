# Project Introduction: CycleGAN for Horse2Zebra Image Translation

This project implements CycleGAN, a deep learning technique, to translate images between two unpaired datasets: horses and zebras. The goal is to train a model that can generate realistic zebra images from input horse images and vice versa.

# I am using Kaggle's GPU for my project, and here is the link to my project (If you can't access it, it's because I set it to private mode):
[Kaggle Notebook: CycleGAN](https://www.kaggle.com/code/nguyenquyetgiangson/cyclegan)

# Demo:
![demo1](https://github.com/GiangSon-5/CycleGAN-for-Horse2Zebra-Image-Translation/blob/main/images/demo1.jpg)

![demo2](https://github.com/GiangSon-5/CycleGAN-for-Horse2Zebra-Image-Translation/blob/main/images/demo2.jpg)

## Steps Taken in the Project

### Data Preparation:
- Downloaded and preprocessed the Horse2Zebra dataset.
- Created custom datasets (Horses and Zebras) inheriting from `torch.utils.data.Dataset` to load and transform images.
- Defined image transformations including resizing, converting to tensors, and normalization.
- Split the datasets into training and testing sets.

### Model Architecture:
Implemented two main components:
- **Generators (G_A2B, G_B2A)**: These models learn to translate images from one domain (horses) to another (zebras) and vice versa. They utilize a ResNet block for feature extraction and residual connections to preserve image details.
- **Discriminators (D_A, D_B)**: These models distinguish between real images from each domain and the generated images by the corresponding generator. They employ convolutional layers and LeakyReLU activations.

### Loss Functions and Optimization:
Defined several loss functions:
- **GAN Loss**: Measures the ability of the discriminator to distinguish real from generated images.
- **Cycle Loss**: Encourages the generated images to be cycled back to the original domain (e.g., horse -> zebra -> horse) and maintain image consistency.
- **Identity Loss**: Penalizes generators for deviating from the input image during the cycle, promoting preservation of image structure.

Used Adam optimizer to update the weights of generators and discriminators during training.

### Training Loop:
- Iterated through epochs and mini-batches.
- Updated generator and discriminator weights alternately:
  - **Generators**: Aim to minimize the combined loss (GAN, cycle, and identity).
  - **Discriminators**: Aim to maximize the ability to distinguish real from generated images.
- Evaluated the model performance on test images periodically.

### Evaluation and Results (See Section: Performance and Results)
- Visually inspected the generated zebra images from horse inputs and vice versa.
- Calculated quantitative metrics (optional) to assess image quality and translation accuracy.

## Tools and Libraries Used
- Python
- PyTorch (deep learning framework)
- Torchvision (dataset utilities)
- NumPy (numerical computations)
- Matplotlib (visualization)
- tqdm (progress bar)

# Explanation of Project Functions

## Detailed explanations of the key project functions:

### ResNetBlock(n_features): 
This class implements a residual block, a fundamental building block in modern convolutional neural networks. It takes the number of input features (`n_features`) and performs two convolutional operations with instance normalization and ReLU activation. The block's output is then added to its input, creating a residual connection that facilitates better gradient flow during training and enables the network to learn deeper representations.

### CycleGAN_Generator(): 
This class defines the architecture of the generator network. It consists of an encoder that downsamples the input image to extract high-level features, a transformer section containing several ResNet blocks to perform the domain transformation, and a decoder that upsamples the transformed features back to the original image resolution, producing the translated image.

### CycleGAN_Discriminator(): 
This class defines the architecture of the discriminator network. It takes an image as input and progressively downsamples it using convolutional layers with LeakyReLU activations. The final output is a single value representing the probability that the input image is real (from the target domain).

## Loss Function and Parameters

### Loss Functions:
- **criterion_gan (MSELoss)**: Measures the mean squared error between the discriminator's output and the target values (1 for real images, 0 for generated/fake images).
- **criterion_cycle (L1Loss)**: Computes the L1 norm (mean absolute difference) between the cycled image (e.g., horse -> zebra -> horse) and the original image.
- **criterion_id (L1Loss)**: Computes the L1 norm between the image translated within its own domain and the original image.

### Optimizers:
- **d_opt_A, d_opt_B, g_opt_A2B, g_opt_B2A (Adam)**: The Adam optimizer is used to update the network weights.

### Hyperparameters:
- **lambda_cycle, lambda_identity**: These are weighting factors that control the importance of the cycle consistency and identity losses, respectively.
- **target_real, target_fake**: These tensors represent the target values for the discriminator during training (1 for real, 0 for fake).

![Loss](https://github.com/GiangSon-5/CycleGAN-for-Horse2Zebra-Image-Translation/blob/main/images/loss.jpg)
