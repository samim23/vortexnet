# VortexNet: Neural Computing through Fluid Dynamics

This repository contains **toy implementations** of the concepts introduced in the research paper [VortexNet: Neural Computing through Fluid Dynamics](https://samim.io/p/2025-01-18-vortextnet/). While not a production-ready package, these examples illustrate how PDE-based vortex layers and fluid-inspired mechanisms can be embedded into a neural architecture (e.g., an autoencoder on MNIST).

## Contents

- `vortexnet.py`:  
  A demonstration script showing how to:
  1. Define vortex-style PDE layers using PyTorch (with Laplacian and advection terms).
  2. Integrate an optional adaptive damping mechanism.
  3. Build a simple autoencoder leveraging these PDE blocks for image reconstruction on MNIST.

> **Note**: This is a **toy prototype** for educational purposes. It is _not_ intended as a fully optimized or physically precise fluid solver.

## Getting Started

1. **Install Dependencies**

   - Python 3.8+
   - PyTorch (1.11+ recommended)
   - torchvision (for MNIST and `save_image`)
   - matplotlib (optional, for visualization)

   ```bash
   pip install torch torchvision matplotlib
   ```

2. **Run the Autoencoder Example**

`python vortexnet.py --epochs 5 --batch_size 64 --lr 1e-3 \
    --hidden_dim 32 --pde_steps 3 --adaptive_damping`

3. **Inspect Results**

   - After training, the script saves and optionally displays reconstructed images (as PNG files or via matplotlib).
   - You should see console logs for training loss and reconstruction loss.
