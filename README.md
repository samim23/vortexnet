# VortexNet: Neural Computing through Fluid Dynamics

This repository contains **toy implementations** of the concepts introduced in the research paper [VortexNet: Neural Computing through Fluid Dynamics](https://samim.io/p/2025-01-18-vortextnet/). These examples demonstrate how PDE-based vortex layers and fluid-inspired mechanisms can be integrated into neural architectures, such as autoencoders for different datasets.

> **Note**: These are **toy prototypes** for educational purposes and are _not_ intended as fully optimized or physically precise fluid solvers.

## Contents

- [`vortexnet_mnist.py`](#vortexnet_mnistpy):  
  A demonstration script for building and training a VortexNet Autoencoder on the MNIST dataset.
- [`vortexnext_image.py`](#vortexnext_imagepy):  
  An advanced script for building and training a VortexNet Autoencoder on custom image datasets with enhanced features like data augmentation and latent space interpolation.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/samim23/vortexnet.git
cd vortexnet
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Install the required Python packages using `pip`:

```bash
pip install torch torchvision matplotlib pyyaml scikit-learn seaborn tensorboard
```

### 3. Prepare the Data

- **MNIST Dataset**:  
  The MNIST dataset will be automatically downloaded by `vortexnet_mnist.py` if not already present.

- **Custom Image Dataset**:  
  For `vortexnext_image.py`, place your images (JPEG, PNG, or JPEG formats) inside the `my_data/` directory.

### 4. Run the Scripts

#### a. VortexNet MNIST Autoencoder (`vortexnet_mnist.py`)

This script builds and trains a VortexNet Autoencoder on the MNIST dataset.

**Usage:**

```bash
python3.11 vortexnet_mnist.py
```

#### b. VortexNet Image Autoencoder (`vortexnext_image.py`)

This advanced script builds and trains a VortexNet Autoencoder on custom image datasets with enhanced features.

**Usage:**

```bash
python3.11 vortexnext_image.py --config config_image.yaml
```

## Notes

- **Configuration Files**:  
  Ensure the configuration file (`config_image.yaml`) is properly set up before running the scripts.

- **Output Directory**:  
  All outputs, including logs, reconstructed images, and model checkpoints, are saved in the `output_dir` specified in the respective configuration files.

- **TensorBoard**:  
  For monitoring training progress, you can launch TensorBoard pointing to the `output_dir`
