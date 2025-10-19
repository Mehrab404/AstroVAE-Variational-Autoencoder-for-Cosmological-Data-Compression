# AstroVAE: Cosmological Data Compression with VAE

AstroVAE uses a Variational Autoencoder (VAE) to compress 3-channel cosmological maps (Mgas, Vgas, B) from the [CAMELS](https://www.camel-simulations.org/) (Cosmology and Astrophysics with MachinE Learning Simulations) project into a 256D latent space, outperforming PCA in reconstruction quality (MSE, PSNR, SSIM). Includes AstroVAE.ipynb for preprocessing, training, and evaluation, and a report (AstroVAE Report.pdf).

## Overview
This project explores the application of a Variational Autoencoder (VAE) for the dimensionality reduction of high-dimensional, multi-channel cosmological simulation data from the CAMELS (Cosmology and Astrophysics with MachinE Learning Simulations) project. The primary goal is to create a compact and meaningful latent-space representation of complex astrophysical data, preserving significant spatial and physical features more effectively than traditional linear methods like Principal Component Analysis (PCA). 

## Problem Statement
Cosmological simulations generate vast datasets that are computationally expensive to store and analyze. Efficiently compressing this data without losing critical information about the underlying physics is a major challenge. This project demonstrates that a VAE can capture non-linear relationships in the data, leading to a more robust and accurate low-dimensional representation.

## Features

-   Compresses 3x256x256 maps (~768x reduction).
    
-   VAE with U-Net architecture (conv layers, skip connections).
    
-   Data augmentation, channel-specific transforms (log, arcsinh), Z-score normalization.
    
-   Metrics: MSE=0.001456, PSNR=28.37 dB, SSIM=0.987 (VAE) vs. MSE=0.492135, PSNR=3.079 dB, SSIM=0.127 (PCA).
    
-   GPU support (NVIDIA RTX 5060).
    

## Dataset

-   [CMD CAMELS Multifield Dataset](https://camels-multifield-dataset.readthedocs.io/en/latest/) (IllustrisTNG, z=0.00).
    
-   Channels: Gas Mass (Mgas), Velocity (Vgas), Magnetic Field (B).
    
-   CV subset for training/validation; LH for testing (1000 samples).
    

## Requirements

-   Python 3.11
    
-   torch, numpy, scikit-learn, matplotlib, pytorch-msssim
    

## Installation

```bash
git clone https://github.com/yourusername/AstroVAE.git
cd AstroVAE
pip install -r requirements.txt
```

Download CAMELS data to G:\\Camel\_Dataset\\data.

## Usage

Run AstroVAE.ipynb:

-   Preprocess data (augmentation, transforms, normalization).
    
-   Train VAE (200 epochs, β=0.1, Adam 3e-4).
    
-   Evaluate on CV/LH test sets.
    
-   Visualize loss curves, reconstructions.
    

Set seed: set\_seed(100).

## Results

-   VAE excels in capturing nonlinear features vs. PCA (~64.8% variance).
    
-   Visuals in report: loss curves, reconstructions.
    
-   See Section 3.5 in report.
    

## Limitations

-   Noise amplification in low-signal areas.
    
-   VAE sensitive to hyperparameters.
    
-   Limited LH evaluation scope.
    

## Contributing

Fork, branch, and submit PRs. Open issues for major changes.

## License

MIT License
