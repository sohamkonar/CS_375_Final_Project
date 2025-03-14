import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import logging

# Configure logging
logger = logging.getLogger("bold5000_model")

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [1024, 512, 256], dropout_rate: float = 0.3):
        super(Encoder, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Add dropout for regularization
            if i < len(dims) - 2:  # No dropout on the last layer
                layers.append(nn.Dropout(dropout_rate))
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # For variational encoder
        self.fc_mu = nn.Linear(dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        features = self.feature_extractor(x)
        
        # Variational parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            # During inference, just use the mean
            return mu

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, img_channels: int = 3, img_size: int = 128, use_residual: bool = True):
        super(Decoder, self).__init__()
        
        self.img_size = img_size
        self.img_channels = img_channels
        self.use_residual = use_residual
        
        # Calculate initial feature map size
        initial_size = img_size // 16  # Upsampled 4 times (2^4 = 16)
        initial_channels = 512
        
        self.initial_linear = nn.Sequential(
            nn.Linear(latent_dim, initial_channels * initial_size * initial_size),
            nn.LeakyReLU(0.2)
        )
        
        self.initial_size = initial_size
        self.initial_channels = initial_channels
        
        # Upsample to get the final image
        if use_residual:
            # Decoder with residual blocks
            self.decoder = nn.Sequential(
                # First upsampling: initial_size -> initial_size*2
                nn.ConvTranspose2d(initial_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                ResidualBlock(256, 256),
                
                # Second upsampling: initial_size*2 -> initial_size*4
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                ResidualBlock(128, 128),
                
                # Third upsampling: initial_size*4 -> initial_size*8
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                ResidualBlock(64, 64),
                
                # Final upsampling: initial_size*8 -> img_size
                nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # Scale output to [0, 1]
            )
        else:
            # Original decoder
            self.decoder = nn.Sequential(
                # First upsampling: initial_size -> initial_size*2
                nn.ConvTranspose2d(initial_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                # Second upsampling: initial_size*2 -> initial_size*4
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                # Third upsampling: initial_size*4 -> initial_size*8
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                
                # Final upsampling: initial_size*8 -> img_size
                nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # Scale output to [0, 1]
            )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch_size, latent_dim]
        x = self.initial_linear(z)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        x = self.decoder(x)
        return x

class FMRItoImageModel(nn.Module):
    def __init__(self, 
                 fmri_dim: int, 
                 latent_dim: int, 
                 img_channels: int = 3, 
                 img_size: int = 128, 
                 hidden_dims: List[int] = [1024, 512, 256],
                 dropout_rate: float = 0.3,
                 use_residual: bool = True,
                 variational: bool = True):
        super(FMRItoImageModel, self).__init__()
        
        self.variational = variational
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(fmri_dim, latent_dim, hidden_dims, dropout_rate)
        self.decoder = Decoder(latent_dim, img_channels, img_size, use_residual)
        
        logger.info(f"Initialized model with fMRI dim: {fmri_dim}, latent dim: {latent_dim}, "
                   f"variational: {variational}, residual blocks: {use_residual}")
        
    def forward(self, fmri: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.variational and self.training:
            z, mu, logvar = self.encoder(fmri)
            reconstructed_image = self.decoder(z)
            return reconstructed_image, mu, logvar
        else:
            z = self.encoder(fmri)
            reconstructed_image = self.decoder(z)
            return reconstructed_image
    
    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence loss for variational model"""
        # KL divergence between the learned distribution and a standard normal distribution
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def get_latent_representation(self, fmri: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from fMRI data"""
        with torch.no_grad():
            if self.variational:
                z = self.encoder(fmri)
                if isinstance(z, tuple):
                    z = z[0]  # Get the sampled latent vector
                return z
            else:
                return self.encoder(fmri)
                
    def generate_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Generate image from latent representation"""
        with torch.no_grad():
            return self.decoder(z)

# GAN-based model for improved image quality
class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, img_size: int = 128):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels, 64, batch_norm=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=0)
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

class FMRItoImageGAN(nn.Module):
    def __init__(self, 
                 fmri_dim: int, 
                 latent_dim: int, 
                 img_channels: int = 3, 
                 img_size: int = 128,
                 hidden_dims: List[int] = [1024, 512, 256],
                 dropout_rate: float = 0.3):
        super(FMRItoImageGAN, self).__init__()
        
        self.generator = FMRItoImageModel(
            fmri_dim=fmri_dim,
            latent_dim=latent_dim,
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=True,
            variational=True
        )
        
        self.discriminator = Discriminator(img_channels, img_size)
        
    def forward(self, fmri: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.generator(fmri)