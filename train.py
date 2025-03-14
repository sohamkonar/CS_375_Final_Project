import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import lpips
from pytorch_msssim import SSIM
import torchmetrics
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Configure logging
logger = logging.getLogger("bold5000_train")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 5, min_delta: float = 0, checkpoint_path: str = 'best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.checkpoint_path = checkpoint_path
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save model checkpoint
            logger.info(f"Validation loss decreased to {val_loss:.4f}. Saving model checkpoint...")
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                self.early_stop = True
        return self.early_stop

class PerceptualLoss(nn.Module):
    """Perceptual loss using LPIPS"""
    def __init__(self, net: str = 'alex'):
        super(PerceptualLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y)

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                device: torch.device,
                config: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Train the model with the given configuration
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda or cpu)
        config: Dictionary containing training configuration
        
    Returns:
        Dictionary containing training and validation metrics
    """
    # Extract configuration parameters
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    use_gan = config.get('use_gan', False)
    use_perceptual_loss = config.get('use_perceptual_loss', False)
    use_ssim_loss = config.get('use_ssim_loss', False)
    kl_weight = config.get('kl_weight', 0.1)  # Weight for KL divergence loss in VAE
    grad_clip_value = config.get('grad_clip_value', 1.0)  # Value for gradient clipping
    use_lr_scheduler = config.get('use_lr_scheduler', True)  # Whether to use learning rate scheduler
    scheduler_type = config.get('scheduler_type', 'plateau')  # Type of scheduler: 'plateau' or 'cosine'
    tensorboard_log_dir = config.get('tensorboard_log_dir', 'runs')  # Directory for TensorBoard logs
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    # Initialize TensorBoard writer
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    
    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize learning rate scheduler
    if use_lr_scheduler:
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}. Not using a scheduler.")
            use_lr_scheduler = False
    
    # Initialize GAN optimizer if using GAN
    if use_gan:
        if not hasattr(model, 'discriminator'):
            logger.warning("GAN training requested but model does not have a discriminator. Falling back to standard training.")
            use_gan = False
        else:
            optimizer_G = optim.Adam(model.generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer_D = optim.Adam(model.discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Initialize schedulers for GAN
            if use_lr_scheduler:
                if scheduler_type == 'plateau':
                    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
                    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)
                elif scheduler_type == 'cosine':
                    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=learning_rate * 0.01)
                    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=learning_rate * 0.01)
    
    # Initialize loss functions
    mse_loss = nn.MSELoss()
    
    # Initialize perceptual loss if requested
    if use_perceptual_loss:
        perceptual_loss = PerceptualLoss().to(device)
    
    # Initialize SSIM loss if requested
    if use_ssim_loss:
        ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    
    # Initialize adversarial loss if using GAN
    if use_gan:
        adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, checkpoint_path=checkpoint_path)
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'learning_rate': []
    }
    
    if use_perceptual_loss:
        metrics['train_perceptual'] = []
        metrics['val_perceptual'] = []
    
    if use_ssim_loss:
        metrics['train_ssim'] = []
        metrics['val_ssim'] = []
    
    if use_gan:
        metrics['train_gen_loss'] = []
        metrics['train_disc_loss'] = []
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_mse_sum = 0.0
        train_perceptual_sum = 0.0
        train_ssim_sum = 0.0
        train_loss_sum = 0.0
        train_gen_loss_sum = 0.0
        train_disc_loss_sum = 0.0
        train_samples = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for fmri, target_img in train_progress:
            fmri, target_img = fmri.to(device), target_img.to(device)
            batch_size = fmri.size(0)
            train_samples += batch_size
            
            # Standard training (non-GAN)
            if not use_gan:
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(model, 'variational') and model.variational and model.training:
                    output, mu, logvar = model(fmri)
                    
                    # Calculate losses
                    mse = mse_loss(output, target_img)
                    
                    # Add perceptual loss if requested
                    if use_perceptual_loss:
                        p_loss = perceptual_loss(output, target_img)
                        train_perceptual_sum += p_loss.item() * batch_size
                    else:
                        p_loss = torch.tensor(0.0, device=device)
                    
                    # Add SSIM loss if requested
                    if use_ssim_loss:
                        s_loss = 1 - ssim_loss(output, target_img)  # 1 - SSIM to convert similarity to loss
                        train_ssim_sum += s_loss.item() * batch_size
                    else:
                        s_loss = torch.tensor(0.0, device=device)
                    
                    # Calculate KL divergence loss
                    kl_loss = model.kl_divergence_loss(mu, logvar)
                    
                    # Combine losses
                    loss = mse + kl_weight * kl_loss
                    
                    if use_perceptual_loss:
                        loss += 0.1 * p_loss
                    
                    if use_ssim_loss:
                        loss += 0.1 * s_loss
                else:
                    output = model(fmri)
                    
                    # Calculate losses
                    mse = mse_loss(output, target_img)
                    
                    # Add perceptual loss if requested
                    if use_perceptual_loss:
                        p_loss = perceptual_loss(output, target_img)
                        train_perceptual_sum += p_loss.item() * batch_size
                    else:
                        p_loss = torch.tensor(0.0, device=device)
                    
                    # Add SSIM loss if requested
                    if use_ssim_loss:
                        s_loss = 1 - ssim_loss(output, target_img)  # 1 - SSIM to convert similarity to loss
                        train_ssim_sum += s_loss.item() * batch_size
                    else:
                        s_loss = torch.tensor(0.0, device=device)
                    
                    # Combine losses
                    loss = mse
                    
                    if use_perceptual_loss:
                        loss += 0.1 * p_loss
                    
                    if use_ssim_loss:
                        loss += 0.1 * s_loss
                
                # Backward pass and optimization
                loss.backward()
                
                # Apply gradient clipping
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                optimizer.step()
                
                # Update metrics
                train_mse_sum += mse.item() * batch_size
                train_loss_sum += loss.item() * batch_size
                
                # Update progress bar
                train_progress.set_postfix(loss=loss.item(), mse=mse.item())
            
            # GAN training
            else:
                # Train Generator
                optimizer_G.zero_grad()
                
                # Generate images
                if hasattr(model.generator, 'variational') and model.generator.variational and model.generator.training:
                    fake_imgs, mu, logvar = model.generator(fmri)
                    kl_loss = model.generator.kl_divergence_loss(mu, logvar)
                else:
                    fake_imgs = model.generator(fmri)
                    kl_loss = torch.tensor(0.0, device=device)
                
                # Calculate generator losses
                mse = mse_loss(fake_imgs, target_img)
                
                # Add perceptual loss if requested
                if use_perceptual_loss:
                    p_loss = perceptual_loss(fake_imgs, target_img)
                    train_perceptual_sum += p_loss.item() * batch_size
                else:
                    p_loss = torch.tensor(0.0, device=device)
                
                # Add SSIM loss if requested
                if use_ssim_loss:
                    s_loss = 1 - ssim_loss(fake_imgs, target_img)
                    train_ssim_sum += s_loss.item() * batch_size
                else:
                    s_loss = torch.tensor(0.0, device=device)
                
                # Adversarial loss
                validity = model.discriminator(fake_imgs)
                real_labels = torch.ones_like(validity, device=device)
                g_loss = adversarial_loss(validity, real_labels)
                
                # Combine losses
                gen_loss = mse + 0.1 * g_loss
                
                if hasattr(model.generator, 'variational') and model.generator.variational:
                    gen_loss += kl_weight * kl_loss
                
                if use_perceptual_loss:
                    gen_loss += 0.1 * p_loss
                
                if use_ssim_loss:
                    gen_loss += 0.1 * s_loss
                
                # Backward pass and optimization
                gen_loss.backward()
                
                # Apply gradient clipping
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip_value)
                
                optimizer_G.step()
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real images
                real_validity = model.discriminator(target_img)
                real_labels = torch.ones_like(real_validity, device=device)
                real_loss = adversarial_loss(real_validity, real_labels)
                
                # Fake images
                fake_validity = model.discriminator(fake_imgs.detach())
                fake_labels = torch.zeros_like(fake_validity, device=device)
                fake_loss = adversarial_loss(fake_validity, fake_labels)
                
                # Combined discriminator loss
                d_loss = (real_loss + fake_loss) / 2
                
                # Backward pass and optimization
                d_loss.backward()
                
                # Apply gradient clipping
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip_value)
                
                optimizer_D.step()
                
                # Update metrics
                train_mse_sum += mse.item() * batch_size
                train_gen_loss_sum += gen_loss.item() * batch_size
                train_disc_loss_sum += d_loss.item() * batch_size
                train_loss_sum += (gen_loss.item() + d_loss.item()) * batch_size
                
                # Update progress bar
                train_progress.set_postfix(
                    g_loss=gen_loss.item(), 
                    d_loss=d_loss.item(), 
                    mse=mse.item()
                )
        
        # Calculate average training metrics
        train_mse = train_mse_sum / train_samples
        train_loss = train_loss_sum / train_samples
        
        metrics['train_mse'].append(train_mse)
        metrics['train_loss'].append(train_loss)
        
        if use_perceptual_loss:
            train_perceptual = train_perceptual_sum / train_samples
            metrics['train_perceptual'].append(train_perceptual)
        
        if use_ssim_loss:
            train_ssim = train_ssim_sum / train_samples
            metrics['train_ssim'].append(train_ssim)
        
        if use_gan:
            train_gen_loss = train_gen_loss_sum / train_samples
            train_disc_loss = train_disc_loss_sum / train_samples
            metrics['train_gen_loss'].append(train_gen_loss)
            metrics['train_disc_loss'].append(train_disc_loss)
        
        # Validation phase
        model.eval()
        val_mse_sum = 0.0
        val_perceptual_sum = 0.0
        val_ssim_sum = 0.0
        val_loss_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for fmri, target_img in val_progress:
                fmri, target_img = fmri.to(device), target_img.to(device)
                batch_size = fmri.size(0)
                val_samples += batch_size
                
                # Forward pass
                if use_gan:
                    output = model.generator(fmri)
                    if isinstance(output, tuple):
                        output = output[0]  # Get the image output, ignore mu and logvar
                else:
                    output = model(fmri)
                    if isinstance(output, tuple):
                        output = output[0]  # Get the image output, ignore mu and logvar
                
                # Calculate losses
                mse = mse_loss(output, target_img)
                
                # Add perceptual loss if requested
                if use_perceptual_loss:
                    p_loss = perceptual_loss(output, target_img)
                    val_perceptual_sum += p_loss.item() * batch_size
                else:
                    p_loss = torch.tensor(0.0, device=device)
                
                # Add SSIM loss if requested
                if use_ssim_loss:
                    s_loss = 1 - ssim_loss(output, target_img)
                    val_ssim_sum += s_loss.item() * batch_size
                else:
                    s_loss = torch.tensor(0.0, device=device)
                
                # Combine losses
                loss = mse
                
                if use_perceptual_loss:
                    loss += 0.1 * p_loss
                
                if use_ssim_loss:
                    loss += 0.1 * s_loss
                
                # Update metrics
                val_mse_sum += mse.item() * batch_size
                val_loss_sum += loss.item() * batch_size
                
                # Update progress bar
                val_progress.set_postfix(loss=loss.item(), mse=mse.item())
        
        # Calculate average validation metrics
        val_mse = val_mse_sum / val_samples
        val_loss = val_loss_sum / val_samples
        
        metrics['val_mse'].append(val_mse)
        metrics['val_loss'].append(val_loss)
        
        if use_perceptual_loss:
            val_perceptual = val_perceptual_sum / val_samples
            metrics['val_perceptual'].append(val_perceptual)
        
        if use_ssim_loss:
            val_ssim = val_ssim_sum / val_samples
            metrics['val_ssim'].append(val_ssim)
        
        # Update learning rate scheduler
        if use_lr_scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            metrics['learning_rate'].append(current_lr)
            
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
                if use_gan:
                    scheduler_G.step(val_loss)
                    scheduler_D.step(val_loss)
            elif scheduler_type == 'cosine':
                scheduler.step()
                if use_gan:
                    scheduler_G.step()
                    scheduler_D.step()
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        
        if use_perceptual_loss:
            writer.add_scalar('Perceptual/train', train_perceptual, epoch)
            writer.add_scalar('Perceptual/val', val_perceptual, epoch)
        
        if use_ssim_loss:
            writer.add_scalar('SSIM_Loss/train', train_ssim, epoch)
            writer.add_scalar('SSIM_Loss/val', val_ssim, epoch)
        
        if use_gan:
            writer.add_scalar('Generator_Loss/train', train_gen_loss, epoch)
            writer.add_scalar('Discriminator_Loss/train', train_disc_loss, epoch)
        
        if use_lr_scheduler:
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log sample images to TensorBoard (first batch only)
        if epoch % 5 == 0:  # Log every 5 epochs to avoid too many images
            with torch.no_grad():
                # Get a batch from validation set
                fmri_sample, target_img_sample = next(iter(val_loader))
                fmri_sample, target_img_sample = fmri_sample.to(device), target_img_sample.to(device)
                
                # Generate output
                if use_gan:
                    output_sample = model.generator(fmri_sample)
                    if isinstance(output_sample, tuple):
                        output_sample = output_sample[0]
                else:
                    output_sample = model(fmri_sample)
                    if isinstance(output_sample, tuple):
                        output_sample = output_sample[0]
                
                # Log only the first few images
                num_images = min(8, fmri_sample.size(0))
                img_grid_real = torchvision.utils.make_grid(target_img_sample[:num_images], normalize=True, scale_each=True)
                img_grid_fake = torchvision.utils.make_grid(output_sample[:num_images], normalize=True, scale_each=True)
                
                writer.add_image('Images/Real', img_grid_real, epoch)
                writer.add_image('Images/Generated', img_grid_fake, epoch)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}")
        
        if use_perceptual_loss:
            logger.info(f"Train Perceptual: {train_perceptual:.4f}, Val Perceptual: {val_perceptual:.4f}")
        
        if use_ssim_loss:
            logger.info(f"Train SSIM Loss: {train_ssim:.4f}, Val SSIM Loss: {val_ssim:.4f}")
        
        if use_gan:
            logger.info(f"Train Generator Loss: {train_gen_loss:.4f}, Discriminator Loss: {train_disc_loss:.4f}")
        
        if use_lr_scheduler:
            logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model
    logger.info(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    
    return metrics