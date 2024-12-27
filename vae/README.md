# Variational Auto-Encoder (VAE)

Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Overview
This repository contains the implementation of a basic Linear Variational Auto-Encoder (VAE) designed for generating MNIST digits.

## Features
- **Training**: Train the VAE model on the MNIST dataset.
- **Digit Generation**: Use the trained model to generate digit images.
- **Latent Space Visualization**: Visualize the latent space to observe where each digit resides.

## File Descriptions
- **`training.py`**: Script to train the VAE model. 
- **`inference.py`**: Script to generate digit images using the trained model. Generated images are saved in the `figures/` folder.
- **`latent_space.py`**: Script to visualize the latent space, providing insights into the distribution of digits in the latent space.