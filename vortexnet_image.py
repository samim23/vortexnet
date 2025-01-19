import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import yaml
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import logging
import math

########################################
# 1) Custom Single-Folder Dataset
########################################

class MySingleFolderDataset(Dataset):
    """
    Custom dataset for loading images from a single folder.
    Returns images and dummy labels.
    """
    def __init__(self, root: str, transform: callable = None):
        """
        Args:
            root (str): Root directory path.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')  # Keep color
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label

########################################
# 2) Utility Functions
########################################

def laplacian_conv2d(x: torch.Tensor, padding_mode: str = 'circular') -> torch.Tensor:
    """
    Applies a Laplacian convolution to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, H, W).
        padding_mode (str): Padding mode to use.

    Returns:
        torch.Tensor: Tensor after applying Laplacian convolution.
    """
    lap_kernel = torch.tensor([[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]], dtype=torch.float32)
    lap_kernel = lap_kernel.unsqueeze(0).unsqueeze(0).to(x.device)  # Shape: (1, 1, 3, 3)

    batch, ch, H, W = x.shape
    weight = lap_kernel.expand(ch, 1, 3, 3)  # Shape: (channels, 1, 3, 3)

    x_splitted = torch.split(x, 1, dim=1)  # Split into individual channels
    conv_results = []
    for i, x_c in enumerate(x_splitted):
        x_c_padded = F.pad(x_c, (1, 1, 1, 1), mode=padding_mode)
        y = F.conv2d(x_c_padded, weight[i:i+1], padding=0)
        conv_results.append(y)
    return torch.cat(conv_results, dim=1)

def grad2d(x: torch.Tensor, padding_mode: str = 'circular') -> tuple:
    """
    Computes the 2D gradients of the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, H, W).
        padding_mode (str): Padding mode to use.

    Returns:
        tuple: Gradients along x and y axes.
    """
    gx_kernel = torch.tensor([[-0.5, 0., 0.5]], dtype=torch.float32)  # Horizontal gradient
    gx_kernel = gx_kernel.unsqueeze(0).unsqueeze(0).to(x.device)   # Shape: (1, 1, 1, 3)
    
    gy_kernel = torch.tensor([[-0.5], [0.], [0.5]], dtype=torch.float32)  # Vertical gradient
    gy_kernel = gy_kernel.unsqueeze(0).unsqueeze(0).to(x.device)       # Shape: (1, 1, 3, 1)

    batch, ch, H, W = x.shape
    grad_x_list, grad_y_list = [], []

    for c in range(ch):
        x_c = x[:, c:c+1]  # Shape: (batch, 1, H, W)
        
        # Horizontal gradient
        x_c_padded_x = F.pad(x_c, (1, 1, 0, 0), mode=padding_mode)
        gx = F.conv2d(x_c_padded_x, gx_kernel, padding=0)
        
        # Vertical gradient
        x_c_padded_y = F.pad(x_c, (0, 0, 1, 1), mode=padding_mode)
        gy = F.conv2d(x_c_padded_y, gy_kernel, padding=0)
        
        grad_x_list.append(gx)
        grad_y_list.append(gy)

    grad_x = torch.cat(grad_x_list, dim=1)
    grad_y = torch.cat(grad_y_list, dim=1)
    return grad_x, grad_y

def curl2d(x: torch.Tensor, y: torch.Tensor, padding_mode: str = 'circular') -> torch.Tensor:
    """
    Computes the 2D curl of a vector field (x, y).

    Args:
        x (torch.Tensor): x-component of the vector field.
        y (torch.Tensor): y-component of the vector field.
        padding_mode (str): Padding mode to use.

    Returns:
        torch.Tensor: Curl of the vector field.
    """
    grad_x, _ = grad2d(x, padding_mode=padding_mode)
    _, grad_y = grad2d(y, padding_mode=padding_mode)
    return grad_y - grad_x  # Curl in 2D

########################################
# 3) PDE Layer Components
########################################

class ResonantCoupledVortexPDEStep(nn.Module):
    """
    PDE step incorporating resonant coupling based on the Strouhal-Neural number (Sn).
    """
    def __init__(self, channels: int = 3, nu_init: float = 0.1, dt: float = 0.1, coupling_strength_init: float = 0.01):
        super().__init__()
        self.nu = nn.Parameter(torch.full((channels,), nu_init, dtype=torch.float32))
        self.dt = dt
        self.forcing_amp = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.coupling_strength = nn.Parameter(torch.full((channels,), coupling_strength_init, dtype=torch.float32))
        self.channels = channels
        
        # Register kernels as buffers to avoid TracerWarnings
        self.register_buffer('lap_kernel', torch.tensor([[0., 1., 0.],
                                                       [1., -4., 1.],
                                                       [0., 1., 0.]], dtype=torch.float32))
        self.register_buffer('gx_kernel', torch.tensor([[-0.5, 0., 0.5]], dtype=torch.float32))
        self.register_buffer('gy_kernel', torch.tensor([[-0.5], [0.], [0.5]], dtype=torch.float32))

    def forward(self, S: torch.Tensor, forcing: torch.Tensor = None, padding_mode: str = 'circular') -> torch.Tensor:
        """
        Forward pass for the PDE step.
        """
        if forcing is None:
            forcing = torch.zeros_like(S)
        
        # Laplacian term
        lap = laplacian_conv2d(S, padding_mode=padding_mode)
        lap_term = (self.nu.view(1, -1, 1, 1) * lap)
        
        # Advection term
        grad_x, grad_y = grad2d(S, padding_mode=padding_mode)
        adv = S * grad_x + S * grad_y
        adv_term = -adv
        
        # Resonant Coupling based on Strouhal-Neural number (Sn)
        A = torch.norm(S, dim=(2,3), keepdim=True)  # Shape: (batch, channels, 1, 1)
        D = 1.0
        
        # Expand coupling_strength to match input dimensions
        f = self.coupling_strength.view(1, -1, 1, 1)  # Shape: (1, channels, 1, 1)
        
        # Calculate Sn with proper broadcasting
        Sn = (f * D) / (A + 1e-8)  # Now shapes will match
        
        # Adjust coupling strength based on Sn
        alpha = 0.05
        beta = 0.05
        gamma0 = 0.0
        phi = (alpha * torch.tanh(beta * Sn) + gamma0).expand_as(S)  # Expand to match input size
        
        coupling = phi * S  # Now dimensions will match
        
        # PDE update
        dSdt = lap_term + adv_term + coupling + self.forcing_amp * forcing
        S_new = S + self.dt * dSdt
        return S_new

class AdaptiveDampingExtended(nn.Module):
    """
    Adaptive damping mechanism based on the magnitude of loss gradients to stabilize training.
    """
    def __init__(self, alpha: float = 0.05, beta: float = 0.05, gamma0: float = 0.0):
        """
        Args:
            alpha (float): Scaling factor for damping.
            beta (float): Controls the nonlinearity of the damping function.
            gamma0 (float): Baseline damping offset.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.gamma0 = nn.Parameter(torch.tensor(gamma0, dtype=torch.float32))

    def forward(self, S: torch.Tensor, loss_grad: torch.Tensor) -> torch.Tensor:
        """
        Applies adaptive damping based on the gradient of the loss function.

        Args:
            S (torch.Tensor): Current state tensor.
            loss_grad (torch.Tensor): Gradient of the loss with respect to S.

        Returns:
            torch.Tensor: Damped state tensor.
        """
        # Compute the magnitude of the gradient
        grad_magnitude = torch.norm(loss_grad, dim=(2,3), keepdim=True)
        
        # Compute gamma(t) based on the local Lyapunov exponent
        gamma_t = self.alpha * torch.tanh(self.beta * grad_magnitude) + self.gamma0
        
        # Apply damping
        damping = 1.0 - gamma_t
        S_damped = S * damping
        return S_damped

class ResonantCoupledVortexPDEStepWithDamping(nn.Module):
    """
    PDE step with resonant coupling and optional adaptive damping.
    """
    def __init__(self, channels: int = 32, nu_init: float = 0.1, dt: float = 0.1, 
                 coupling_strength_init: float = 0.01, use_adaptive_damping: bool = False):
        """
        Args:
            channels (int): Number of channels in the input tensor.
            nu_init (float): Initial viscosity parameter.
            dt (float): Time step size.
            coupling_strength_init (float): Initial coupling strength.
            use_adaptive_damping (bool): Whether to use adaptive damping.
        """
        super().__init__()
        self.pde_step = ResonantCoupledVortexPDEStep(
            channels=channels, 
            nu_init=nu_init, 
            dt=dt, 
            coupling_strength_init=coupling_strength_init
        )
        self.use_adaptive_damping = use_adaptive_damping
        self.S_out = None  # To store the state
        if use_adaptive_damping:
            self.adaptive_damping = AdaptiveDampingExtended(alpha=0.05, beta=0.05, gamma0=0.0)
    
    def forward(self, S: torch.Tensor, forcing: torch.Tensor = None, padding_mode: str = 'circular', loss_grad: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the PDE step with optional adaptive damping.

        Args:
            S (torch.Tensor): Input state tensor of shape (batch, channels, H, W).
            forcing (torch.Tensor, optional): Forcing term tensor of the same shape as S.
            padding_mode (str): Padding mode for convolutions.
            loss_grad (torch.Tensor, optional): Gradient of the loss with respect to S.

        Returns:
            torch.Tensor: Updated state tensor.
        """
        S_out = self.pde_step(S, forcing=forcing, padding_mode=padding_mode)
        self.S_out = S_out.clone().detach()  # Store for stability loss
        
        if self.use_adaptive_damping and loss_grad is not None:
            S_out = self.adaptive_damping(S_out, loss_grad)
        
        return S_out

########################################
# 4) Stability Loss
########################################

class StabilityLoss(nn.Module):
    """
    Stability regularization loss to penalize high activation magnitudes.
    """
    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha (float): Weighting factor for the stability loss.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Computes the stability loss based on the magnitude of the state.

        Args:
            S (torch.Tensor): Current state tensor.

        Returns:
            torch.Tensor: Stability loss scalar.
        """
        # Penalize high activation magnitudes
        stability_loss = torch.mean(S ** 2)
        return self.alpha * stability_loss

########################################
# 5) VortexAutoencoder with Resonant Coupling and Adaptive Damping
########################################

class VortexAutoencoderFull(nn.Module):
    """
    VortexNet Autoencoder integrating PDE-based layers with resonant coupling and adaptive damping.
    """
    def __init__(self, hidden_dim: int = 64, pde_channels: int = 32, pde_steps: int = 1, 
                 use_adaptive_damping: bool = False, image_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pde_channels = pde_channels
        self.pde_steps = pde_steps
        self.use_adaptive_damping = use_adaptive_damping
        
        # Dynamically compute bottleneck size
        self.bottleneck_size = image_size // (2**4)  # Four downsampling layers

        # Encoder: 3x128x128 -> pde_channels x8x8
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.pde_channels, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=self.pde_channels, out_channels=self.pde_channels, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU()
        )
        
        # Enhanced PDE block in the encoder
        self.encoder_pde = ResonantCoupledVortexPDEStepWithDamping(
            channels=self.pde_channels, 
            nu_init=0.05, 
            dt=0.1, 
            coupling_strength_init=0.01,
            use_adaptive_damping=self.use_adaptive_damping
        )
        
        # Fully connected layers
        self.encoder_fc = nn.Linear(self.pde_channels * self.bottleneck_size * self.bottleneck_size, self.hidden_dim)
        self.decoder_fc = nn.Linear(self.hidden_dim, self.pde_channels * self.bottleneck_size * self.bottleneck_size)
        
        # Enhanced PDE block in the decoder
        self.decoder_pde = ResonantCoupledVortexPDEStepWithDamping(
            channels=self.pde_channels, 
            nu_init=0.05, 
            dt=0.1, 
            coupling_strength_init=0.01,
            use_adaptive_damping=self.use_adaptive_damping
        )
        
        # Decoder: pde_channels x8x8 -> 3x128x128
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.pde_channels, out_channels=self.pde_channels, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.pde_channels, out_channels=32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.Sigmoid()  # Ensure outputs are in [0,1]
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into the latent space.
        """
        x = self.encoder_conv(x)  # Shape: (batch, pde_channels, bottleneck_size, bottleneck_size)
        for step in range(self.pde_steps):
            x = self.encoder_pde(x, padding_mode='circular')
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent tensor back to the image space.
        """
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.pde_channels, self.bottleneck_size, self.bottleneck_size)
        for step in range(self.pde_steps):
            x = self.decoder_pde(x, padding_mode='circular')
        x = self.decoder_deconv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon


########################################
# 6) Latent Space Interpolation Functions
########################################

def interpolate_latent(model: nn.Module, imgA: torch.Tensor, imgB: torch.Tensor, steps: int = 10, device: str = "cpu") -> list:
    """
    Performs linear interpolation between two latent vectors and decodes the interpolations.

    Args:
        model (nn.Module): Trained autoencoder model.
        imgA (torch.Tensor): First image tensor of shape (3, 128, 128).
        imgB (torch.Tensor): Second image tensor of shape (3, 128, 128).
        steps (int): Number of interpolation steps.
        device (str): Device to perform computations on.

    Returns:
        list: List of interpolated image tensors.
    """
    model.eval()
    
    # Add batch dimension and move to device
    xA = imgA.unsqueeze(0).to(device)  # Shape: (1, 3, 128, 128)
    xB = imgB.unsqueeze(0).to(device)  # Shape: (1, 3, 128, 128)
    
    with torch.no_grad():
        # Encode images to latent vectors
        zA = model.encode(xA)  # Shape: (1, hidden_dim)
        zB = model.encode(xB)  # Shape: (1, hidden_dim)
        
        # Interpolate between zA and zB
        interpolations = []
        for i in range(steps + 1):
            alpha = i / steps
            z = (1 - alpha) * zA + alpha * zB  # Linear interpolation
            x_dec = model.decode(z)             # Decode to image
            interpolations.append(x_dec.squeeze(0))  # Remove batch dimension
    
    return interpolations

def save_interpolated_images(interpolations: list, output_path: str, nrow: int = 5):
    """
    Saves interpolated images as a grid.

    Args:
        interpolations (list): List of image tensors.
        output_path (str): Path to save the grid image.
        nrow (int): Number of images per row in the grid.
    """
    grid = vutils.make_grid(interpolations, nrow=nrow, normalize=True, padding=2)
    vutils.save_image(grid, output_path)
    print(f"Saved interpolated images to '{output_path}'")

def plot_interpolated_images(interpolations: list, nrow: int = 5):
    """
    Plots interpolated images as a grid.

    Args:
        interpolations (list): List of image tensors.
        nrow (int): Number of images per row in the grid.
    """
    grid = vutils.make_grid(interpolations, nrow=nrow, normalize=True, padding=2)
    np_grid = grid.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC for plotting
    
    plt.figure(figsize=(nrow * 2, (len(interpolations) // nrow) * 2))
    plt.imshow(np_grid)
    plt.axis('off')
    plt.title('Latent Space Interpolation')
    plt.tight_layout()
    plt.show()

########################################
# 7) Training and Evaluation Script
########################################

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Slower, more gradual warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Slower decay with multiple cycles
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def plot_training_progress(losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.close()

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            recons = model(imgs)
            loss = criterion(recons, imgs)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_reconstruction_samples(model, dataloader, device, epoch, output_dir):
    """
    Saves reconstruction samples at the current epoch.
    """
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        sample_batch, _ = next(iter(dataloader))
        sample_batch = sample_batch.to(device)
        recons = model(sample_batch)
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        sample_batch_denorm = torch.clamp(sample_batch * std + mean, 0, 1)
        recons_denorm = torch.clamp(recons * std + mean, 0, 1)
        
        # Create comparison grid: original | reconstruction
        comparison = torch.cat([sample_batch_denorm, recons_denorm], dim=0)
        
        # Save images
        filename = os.path.join(output_dir, f'reconstruction_epoch_{epoch:03d}.png')
        vutils.save_image(comparison, filename, 
                         nrow=sample_batch.size(0),  # Number of images per row
                         normalize=False,  # We already normalized
                         padding=2)
        
        print(f"Saved reconstruction samples for epoch {epoch} to '{filename}'")
        
        # Save individual original and reconstruction pairs
        for i in range(sample_batch.size(0)):
            orig = sample_batch_denorm[i:i+1]
            recon = recons_denorm[i:i+1]
            pair = torch.cat([orig, recon], dim=0)
            pair_filename = os.path.join(output_dir, f'reconstruction_epoch_{epoch:03d}_sample_{i:02d}.png')
            vutils.save_image(pair, pair_filename, 
                            nrow=2,  # Two images per row (original and reconstruction)
                            normalize=False,
                            padding=2)

def main():
    print("Starting the VortexNet Autoencoder training process...")
    
    # Parse configuration
    parser = argparse.ArgumentParser(description="VortexNet Autoencoder with Enhanced Features")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configurations
    training_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    interpolation_cfg = config['interpolation']
    image_size = int(data_cfg['image_size'])

    # Enforce correct types
    try:
        training_cfg['lr'] = float(training_cfg['lr'])
        training_cfg['weight_decay'] = float(training_cfg['weight_decay'])
        training_cfg['epochs'] = int(training_cfg['epochs'])
        training_cfg['batch_size'] = int(training_cfg['batch_size'])
        training_cfg['log_interval'] = int(training_cfg['log_interval'])
        training_cfg['save_interval'] = int(training_cfg['save_interval'])
        training_cfg['use_mixed_precision'] = bool(training_cfg['use_mixed_precision'])

        model_cfg['hidden_dim'] = int(model_cfg['hidden_dim'])
        model_cfg['pde_channels'] = int(model_cfg['pde_channels'])
        model_cfg['pde_steps'] = int(model_cfg['pde_steps'])
        model_cfg['use_adaptive_damping'] = bool(model_cfg['use_adaptive_damping'])


        interpolation_cfg['steps'] = int(interpolation_cfg['steps'])
    except ValueError as ve:
        print(f"Configuration type casting error: {ve}")
        logging.error(f"Configuration type casting error: {ve}")
        return

    # Debugging: Print types of critical parameters after casting
    print("Training Configuration:")
    for key, value in training_cfg.items():
        print(f"  {key}: {value} (type: {type(value)})")

    # Assertions to ensure correct types
    assert isinstance(training_cfg['lr'], float), "lr must be a float."
    assert isinstance(training_cfg['weight_decay'], float), "weight_decay must be a float."
    # Add similar assertions for other critical parameters as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up logging
    os.makedirs(data_cfg['output_dir'], exist_ok=True)
    logging.basicConfig(filename=os.path.join(data_cfg['output_dir'], 'training.log'), 
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info(f"Training started on device: {device}")
    print("Logging initialized.")

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=data_cfg['output_dir'])
    print("TensorBoard initialized.")

    # Define image transformations: Resize to 128x128, keep RGB, convert to tensor
    # transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),  # Shape: (3, image_size, image_size)
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
    #                          std=[0.229, 0.224, 0.225]),
    # ])

    # Resize to 128x128, keep RGB, convert to tensor, add Data augmentation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.RandomRotation(10),           # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    # Initialize dataset and dataloader
    print(f"Loading dataset from '{data_cfg['data_dir']}'...")
    dataset = MySingleFolderDataset(root=data_cfg['data_dir'], transform=transform)
    if len(dataset) == 0:
        print(f"No images found in '{data_cfg['data_dir']}'. Please add images and try again.")
        logging.error(f"No images found in '{data_cfg['data_dir']}'. Exiting.")
        return
    dataloader = DataLoader(dataset, batch_size=training_cfg['batch_size'], shuffle=True, 
                            num_workers=8, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} images.")

    # Initialize the VortexAutoencoder model
    print("Initializing the VortexAutoencoder model...")
    model = VortexAutoencoderFull(
        hidden_dim=model_cfg['hidden_dim'],
        pde_channels=model_cfg['pde_channels'],
        pde_steps=model_cfg['pde_steps'],
        use_adaptive_damping=model_cfg['use_adaptive_damping'],
        image_size=data_cfg['image_size']
    ).to(device)
    print("Model initialized.")
    logging.info("Model initialized.")

    # Define optimizer and loss functions
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg['lr'],
        weight_decay=training_cfg['weight_decay'],
        betas=(0.9, 0.999)  # Default Adam betas
    )

    # Add gradient clipping
    max_grad_norm = training_cfg.get('gradient_clip', 1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Initialize loss functions
    recon_criterion = nn.MSELoss()
    stability_loss_fn = StabilityLoss(alpha=0.01).to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    recon_criterion = nn.MSELoss()
    stability_loss_fn = StabilityLoss(alpha=0.01).to(device)

    # Optional: Initialize mixed precision scaler
    scaler = GradScaler() if training_cfg['use_mixed_precision'] else None
    if scaler:
        print("Mixed precision training enabled.")
    else:
        print("Mixed precision training disabled.")

    num_training_steps = len(dataloader) * training_cfg['epochs']
    num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, training_cfg['epochs'] + 1):
        model.train()
        epoch_losses = []
        
        for batch_idx, (imgs, _) in enumerate(dataloader, 1):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            recons = model(imgs)
            recon_loss = recon_criterion(recons, imgs)
            
            S_out = model.encoder_pde.S_out
            if S_out is not None:
                stability = stability_loss_fn(S_out)
            else:
                stability = 0.0
                
            loss = recon_loss + stability
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % training_cfg['log_interval'] == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch}][Batch {batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
                
        # Compute average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        # Validate model
        val_loss = validate_model(model, dataloader, recon_criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch}/{training_cfg['epochs']}] "
              f"Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Plot progress
        plot_training_progress(train_losses, data_cfg['output_dir'])

        # Save reconsruction
        if epoch % training_cfg['reconstruction_interval'] == 0:
            save_reconstruction_samples(model, dataloader, device, epoch, data_cfg['output_dir'])
        
        # Save checkpoint
        if epoch % training_cfg['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, 
                      os.path.join(data_cfg['output_dir'], f'checkpoint_epoch_{epoch}.pth'))

    # Save the final trained model
    final_model_path = os.path.join(data_cfg['output_dir'], "vortex_autoencoder_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to '{final_model_path}'")
    logging.info(f"Saved final model to '{final_model_path}'")
    writer.add_text('Final Model Path', final_model_path, training_cfg['epochs'])

    ########################################
    # 8) Latent Space Interpolation
    ########################################

    # Select two random images from the dataset for interpolation
    print("Starting latent space interpolation...")
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        try:
            sample_batch, _ = next(iter(dataloader))
            if sample_batch.size(0) < 2:
                print("Not enough images in the batch for interpolation. Exiting interpolation step.")
                logging.warning("Not enough images in the batch for interpolation. Exiting interpolation step.")
            else:
                imgA = sample_batch[0]  # First image in the batch
                imgB = sample_batch[1]  # Second image in the batch

                # Perform interpolation
                interpolations = interpolate_latent(
                    model=model,
                    imgA=imgA,
                    imgB=imgB,
                    steps=interpolation_cfg['steps'],
                    device=device
                )

                # Save interpolated images as a grid
                interp_output_path = os.path.join(data_cfg['output_dir'], "interpolated_samples.png")
                save_interpolated_images(interpolations, interp_output_path, nrow=interpolation_cfg['steps'] + 1)
                logging.info(f"Saved interpolated images to '{interp_output_path}'")

                # Plot interpolated images
                plot_interpolated_images(interpolations, nrow=interpolation_cfg['steps'] + 1)
        except StopIteration:
            print("Dataset iterator is empty. Cannot perform interpolation.")
            logging.error("Dataset iterator is empty. Cannot perform interpolation.")

    ########################################
    # 9) Save Original and Reconstructed Images
    ########################################

    # Save original and reconstructed images from the last batch
    print("Saving original and reconstructed images...")
    with torch.no_grad():
        recons = model(sample_batch.to(device))
    
    # Denormalize images for saving
    def denormalize(tensor, mean, std):
        """
        Denormalizes a tensor using the provided mean and std.

        Args:
            tensor (torch.Tensor): Normalized tensor.
            mean (torch.Tensor): Mean for each channel.
            std (torch.Tensor): Std for each channel.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    sample_batch_denorm = denormalize(sample_batch.to(device).clone(), mean, std)
    recons_denorm = denormalize(recons.clone(), mean, std)

    # Save original images
    orig_output_path = os.path.join(data_cfg['output_dir'], "original_samples_128.png")
    vutils.save_image(sample_batch_denorm, orig_output_path, nrow=4, normalize=False)
    
    # Save reconstructed images
    recon_output_path = os.path.join(data_cfg['output_dir'], "reconstructed_samples_128.png")
    vutils.save_image(recons_denorm, recon_output_path, nrow=4, normalize=False)
    
    print(f"Saved original images to '{orig_output_path}'")
    print(f"Saved reconstructed images to '{recon_output_path}'")
    logging.info(f"Saved original images to '{orig_output_path}'")
    logging.info(f"Saved reconstructed images to '{recon_output_path}'")
    writer.add_image('Original Images', vutils.make_grid(sample_batch_denorm[:8], nrow=4), training_cfg['epochs'])
    writer.add_image('Reconstructed Images', vutils.make_grid(recons_denorm[:8], nrow=4), training_cfg['epochs'])

    ########################################
    # 10) Closing and Cleanup
    ########################################

    writer.close()
    logging.info("Training and evaluation completed.")

    ########################################
    # 11) Optional: Visualize Latent Space with t-SNE
    ########################################

    def visualize_latent_space(model: nn.Module, dataloader: DataLoader, device: str, num_samples: int = 1000):
        """
        Visualizes the latent space using t-SNE with adjusted parameters for small datasets.
        """
        print("Visualizing latent space with t-SNE...")
        model.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.to(device)
                z = model.encode(imgs)
                latents.append(z.cpu())
                labels.extend(lbls.numpy())
                if len(latents) * imgs.size(0) >= num_samples:
                    break
        
        latents = torch.cat(latents, dim=0).numpy()[:num_samples]
        labels = labels[:num_samples]
        
        # Adjust t-SNE parameters based on dataset size
        n_samples = latents.shape[0]
        perplexity = min(n_samples - 1, 30)  # Ensure perplexity is less than n_samples
        
        # Initialize t-SNE with adjusted parameters
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=42,
            learning_rate='auto',  # Use auto learning rate
            init='pca'  # Use PCA initialization for better stability
        )
        
        try:
            latents_2d = tsne.fit_transform(latents)
            
            plt.figure(figsize=(10, 10))
            sns.scatterplot(
                x=latents_2d[:,0], 
                y=latents_2d[:,1], 
                hue=labels, 
                palette='viridis',
                legend='full'
            )
            plt.title(f't-SNE Visualization (perplexity={perplexity})')
            
            # Save the plot
            tsne_output_path = os.path.join(data_cfg['output_dir'], 'latent_space_tsne.png')
            plt.savefig(tsne_output_path)
            print(f"Saved t-SNE visualization to {tsne_output_path}")
            
            # Close the figure to free memory
            plt.close()
            
        except Exception as e:
            print(f"Warning: t-SNE visualization failed with error: {str(e)}")
            print("This is not critical - continuing with the rest of the program.")

    visualize_latent_space(model, dataloader, device, num_samples=1000)

    ########################################
    # 12) Optional: Export Model for Deployment
    ########################################

    # Example: Export to ONNX
    print("Exporting model to ONNX format...")
    onnx_model_path = os.path.join(data_cfg['output_dir'], "vortex_autoencoder.onnx")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=11, 
                      input_names=['input'], output_names=['output'])
    print(f"Exported model to ONNX format at '{onnx_model_path}'")
    logging.info(f"Exported model to ONNX format at '{onnx_model_path}'")

########################################
# 15) Main Guard to Execute main()
########################################

if __name__ == '__main__':
    main()


########################################
# 14) Running the Script
########################################

# To run the script, execute the following command in your terminal:
# python vortexnet_autoencoder.py --config config.yaml

# Ensure that the 'config.yaml' file is properly configured and the data directory exists with appropriate images.
