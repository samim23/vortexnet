import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


########################################
# Utility Functions
########################################

def laplacian_conv2d(x, padding_mode='circular'):
    """
    Applies a Laplacian kernel via 2D convolution, with manual padding for boundary conditions.
    padding_mode can be 'circular', 'reflect', or 'constant' (for zero-padding).
    """
    # Laplacian stencil
    lap_kernel = torch.tensor([[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]], dtype=torch.float32)
    lap_kernel = lap_kernel.unsqueeze(0).unsqueeze(0).to(x.device)  # shape (1,1,3,3)

    batch, ch, H, W = x.shape
    weight = lap_kernel.expand(ch, 1, 3, 3)

    # We'll split x by channel so we can apply the same kernel to each channel independently
    x_splitted = torch.split(x, 1, dim=1)  # ch chunks, each of shape (batch,1,H,W)

    conv_results = []
    for i, x_c in enumerate(x_splitted):
        # Manually pad x_c depending on padding_mode
        x_c_padded = F.pad(x_c, (1,1,1,1), mode=padding_mode)  
        # Now convolve with padding=0 (since we already padded)
        y = F.conv2d(x_c_padded, weight[i:i+1], padding=0)
        conv_results.append(y)

    return torch.cat(conv_results, dim=1)


def grad2d(x, padding_mode='circular'):
    """
    Compute gradients in x and y directions for each channel, using small 3x3 stencils
    with manual padding.
    """
    # Horizontal gradient kernel (diff wrt x):
    gx_kernel = torch.tensor([[-0.5, 0., 0.5]], dtype=torch.float32)  # shape (1,3)
    gx_kernel = gx_kernel.unsqueeze(0).unsqueeze(0)                   # shape (1,1,1,3)

    # Vertical gradient kernel (diff wrt y):
    gy_kernel = torch.tensor([[-0.5], [0.], [0.5]], dtype=torch.float32)  # shape (3,1)
    gy_kernel = gy_kernel.unsqueeze(0).unsqueeze(0)                       # shape (1,1,3,1)

    gx_kernel = gx_kernel.to(x.device)
    gy_kernel = gy_kernel.to(x.device)

    batch, ch, H, W = x.shape
    grad_x_list, grad_y_list = [], []

    for c in range(ch):
        x_c = x[:, c:c+1]  # shape (batch,1,H,W)

        # Manually pad for horizontal gradient
        x_c_padded_x = F.pad(x_c, (1,1,0,0), mode=padding_mode)  # left=1, right=1, top=0, bottom=0
        gx = F.conv2d(x_c_padded_x, gx_kernel, padding=0)

        # Manually pad for vertical gradient
        x_c_padded_y = F.pad(x_c, (0,0,1,1), mode=padding_mode)  # top=1, bottom=1
        gy = F.conv2d(x_c_padded_y, gy_kernel, padding=0)

        grad_x_list.append(gx)
        grad_y_list.append(gy)

    grad_x = torch.cat(grad_x_list, dim=1)
    grad_y = torch.cat(grad_y_list, dim=1)
    return grad_x, grad_y


############################################
# PDE Layer
############################################

class VortexPDEStep(nn.Module):
    """
    Single step of the PDE update:
       dS/dt = nu * Laplacian(S) - (V . ∇)S + F
    """
    def __init__(self, channels=1, nu=0.1, dt=0.1):
        super().__init__()
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float32))
        self.dt = dt
        self.forcing_amp = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.channels = channels

    def forward(self, S, forcing=None, padding_mode='circular'):
        if forcing is None:
            forcing = torch.zeros_like(S)

        # Laplacian term
        lap = laplacian_conv2d(S, padding_mode=padding_mode)
        lap_term = self.nu * lap

        # Advection-like term: -(V . ∇)S
        grad_x, grad_y = grad2d(S, padding_mode=padding_mode)
        if self.channels == 2:
            # interpret S as (u, v)
            u = S[:, 0:1]
            v = S[:, 1:2]
            adv = u * grad_x + v * grad_y
        else:
            # naive approach for 1 channel: S * ∂S/∂x + S * ∂S/∂y
            adv = S * grad_x + S * grad_y

        adv_term = -adv

        # PDE update
        dSdt = lap_term + adv_term + self.forcing_amp * forcing
        S_new = S + self.dt * dSdt
        return S_new


class VortexPDELayer(nn.Module):
    """
    Unroll PDE steps T times in forward pass.
    """
    def __init__(self, channels=1, nu=0.1, dt=0.1, steps=3):
        super().__init__()
        self.steps = steps
        self.pde_step = VortexPDEStep(channels=channels, nu=nu, dt=dt)

    def forward(self, S, forcing=None, padding_mode='circular'):
        for _ in range(self.steps):
            S = self.pde_step(S, forcing=forcing, padding_mode=padding_mode)
        return S


##########################
# Adaptive Damping
##########################

class AdaptiveDamping(nn.Module):
    """
    gamma(t) = alpha * tanh( beta * local_metric ) + gamma0
    Then we scale S by (1 - gamma(t)).
    """
    def __init__(self, alpha=0.05, beta=0.05, gamma0=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.gamma0 = nn.Parameter(torch.tensor(gamma0, dtype=torch.float32))

    def forward(self, S, local_metric):
        gamma_t = self.alpha * torch.tanh(self.beta * local_metric) + self.gamma0
        damping = 1.0 - gamma_t
        return S * damping


##################################################
# A PDE Block + Damping
##################################################

class VortexBlock(nn.Module):
    """
    A block that does multiple PDE unrollings + optional adaptive damping.
    """
    def __init__(self, channels=1, nu=0.1, dt=0.1, steps=3, use_adaptive_damping=False):
        super().__init__()
        self.pde_layer = VortexPDELayer(channels=channels, nu=nu, dt=dt, steps=steps)
        self.use_adaptive_damping = use_adaptive_damping
        if use_adaptive_damping:
            self.adaptive_damping = AdaptiveDamping(alpha=0.05, beta=0.05, gamma0=0.0)

    def forward(self, S, forcing=None, padding_mode='circular'):
        S_out = self.pde_layer(S, forcing=forcing, padding_mode=padding_mode)
        if self.use_adaptive_damping:
            energy = (S_out**2).mean()
            S_out = self.adaptive_damping(S_out, energy)
        return S_out


##################################################
# VortexAutoencoder for MNIST Reconstruction
##################################################

class VortexAutoencoder(nn.Module):
    """
    Simple AE:
    - Encoder: downsample => PDE-based VortexBlock => flatten => latent
    - Decoder: expand latent => PDE-based VortexBlock => upsample => reconstruct
    """
    def __init__(self, hidden_dim=32, pde_channels=1, pde_steps=3, use_adaptive_damping=False):
        super().__init__()

        # MNIST is (batch, 1, 28, 28)
        # 1) Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU()
        )
        # PDE block in the encoder
        self.encoder_pde = VortexBlock(channels=16, nu=0.05, dt=0.1, steps=pde_steps,
                                       use_adaptive_damping=use_adaptive_damping)
        
        # Flatten and map to latent
        self.encoder_fc = nn.Linear(16*7*7, hidden_dim)

        # 2) Decoder
        self.decoder_fc = nn.Linear(hidden_dim, 16*7*7)
        self.decoder_pde = VortexBlock(channels=16, nu=0.05, dt=0.1, steps=pde_steps,
                                       use_adaptive_damping=use_adaptive_damping)

        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_pde(x, padding_mode='circular')
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x = self.encoder_fc(x)
        return x

    def decode(self, z):
        x = self.decoder_fc(z)
        b = x.shape[0]
        x = x.view(b, 16, 7, 7)
        x = self.decoder_pde(x, padding_mode='circular')
        x = self.decoder_deconv(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon


##################################################
# Training Script
##################################################

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--pde_steps", type=int, default=3)
    parser.add_argument("--adaptive_damping", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNIST data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = VortexAutoencoder(hidden_dim=args.hidden_dim,
                              pde_channels=1,
                              pde_steps=args.pde_steps,
                              use_adaptive_damping=args.adaptive_damping).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f}")

        # Quick test on a smaller batch
        model.eval()
        with torch.no_grad():
            imgs, _ = next(iter(test_loader))
            imgs = imgs.to(device)
            recon = model(imgs)
            test_loss = criterion(recon, imgs).item()
        print(f"Test Reconstruction Loss: {test_loss:.4f}")

    # Sample reconstructions
    with torch.no_grad():
        sample_imgs, _ = next(iter(test_loader))
        sample_imgs = sample_imgs.to(device)
        recons = model(sample_imgs)
    
    print("Sample input shape:", sample_imgs.shape)
    print("Reconstructed shape:", recons.shape)

    # -------------------------------------------------------------------------
    # 1) Save a grid of original and reconstructed images to disk
    # -------------------------------------------------------------------------
    vutils.save_image(sample_imgs, "original_samples.png", nrow=8, normalize=True)
    vutils.save_image(recons, "reconstructed_samples.png", nrow=8, normalize=True)
    print("Saved 'original_samples.png' and 'reconstructed_samples.png'")

    # -------------------------------------------------------------------------
    # 2) Show some images in a matplotlib figure (top: original, bottom: recon)
    # -------------------------------------------------------------------------
    sample_imgs_cpu = sample_imgs.cpu()
    recons_cpu = recons.cpu().detach()

    n = 8  # how many images to display
    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))
    for i in range(n):
        # Original
        axes[0, i].imshow(sample_imgs_cpu[i, 0], cmap='gray', interpolation='nearest')
        axes[0, i].axis('off')
        # Reconstructed
        axes[1, i].imshow(recons_cpu[i, 0], cmap='gray', interpolation='nearest')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()