import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import lpips
from pytorch_msssim import SSIM
import torchmetrics
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

# Configure logging
logger = logging.getLogger("bold5000_evaluate")

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  output_dir: str,
                  config: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Evaluate the model on test data and save visualizations
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save results
        config: Dictionary containing evaluation configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if config is None:
        config = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    mse_loss = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    
    # Initialize result storage
    metrics = {
        'mse': 0.0,
        'lpips': 0.0,
        'ssim': 0.0,
        'psnr': 0.0
    }
    
    # For visualization
    num_samples_to_visualize = min(16, len(test_loader.dataset))
    sample_indices = np.random.choice(len(test_loader.dataset), num_samples_to_visualize, replace=False)
    
    # For latent space visualization
    all_latents = []
    all_fmri_ids = []  # To track which subject/image the latent represents
    
    # Evaluation loop
    model.eval()
    test_samples = 0
    
    with torch.no_grad():
        sample_idx = 0
        fmri_samples = []
        real_images = []
        reconstructed_images = []
        
        for batch_idx, (fmri, target_img) in enumerate(tqdm(test_loader, desc="Evaluating")):
            fmri, target_img = fmri.to(device), target_img.to(device)
            batch_size = fmri.size(0)
            test_samples += batch_size
            
            # Get model output
            if hasattr(model, 'generator'):  # GAN model
                output = model.generator(fmri)
                # Extract latent representation for visualization
                if hasattr(model.generator, 'get_latent_representation'):
                    latents = model.generator.get_latent_representation(fmri).cpu().numpy()
                    all_latents.append(latents)
                    # Store indices for each sample
                    all_fmri_ids.extend(list(range(batch_idx * batch_size, batch_idx * batch_size + batch_size)))
            else:
                output = model(fmri)
                # Extract latent representation for visualization
                if hasattr(model, 'get_latent_representation'):
                    latents = model.get_latent_representation(fmri).cpu().numpy()
                    all_latents.append(latents)
                    # Store indices for each sample
                    all_fmri_ids.extend(list(range(batch_idx * batch_size, batch_idx * batch_size + batch_size)))
            
            # Handle tuple output from variational models
            if isinstance(output, tuple):
                output = output[0]
            
            # Calculate metrics
            mse = mse_loss(output, target_img).item()
            lpips_val = lpips_loss(output, target_img).mean().item()
            ssim_val = ssim_metric(output, target_img).item()
            psnr_val = psnr_metric(output, target_img).item()
            
            # Update metrics
            metrics['mse'] += mse * batch_size
            metrics['lpips'] += lpips_val * batch_size
            metrics['ssim'] += ssim_val * batch_size
            metrics['psnr'] += psnr_val * batch_size
            
            # Save samples for visualization
            for i in range(batch_size):
                if sample_idx in sample_indices:
                    fmri_samples.append(fmri[i].cpu().numpy())
                    real_images.append(target_img[i].cpu().numpy())
                    reconstructed_images.append(output[i].cpu().numpy())
                sample_idx += 1
    
    # Calculate average metrics
    for key in metrics:
        metrics[key] /= test_samples
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"MSE: {metrics['mse']:.4f}")
    logger.info(f"LPIPS: {metrics['lpips']:.4f}")
    logger.info(f"SSIM: {metrics['ssim']:.4f} (higher is better)")
    logger.info(f"PSNR: {metrics['psnr']:.4f} dB (higher is better)")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Visualize results
    visualize_reconstructions(real_images, reconstructed_images, output_dir)
    
    # Visualize latent space if we have latent representations
    if all_latents:
        all_latents = np.vstack(all_latents)
        visualize_latent_space(all_latents, all_fmri_ids, output_dir)
    
    return metrics

def visualize_reconstructions(real_images: List[np.ndarray], 
                              reconstructed_images: List[np.ndarray], 
                              output_dir: str) -> None:
    """
    Visualize original and reconstructed images side by side
    
    Args:
        real_images: List of original images as numpy arrays
        reconstructed_images: List of reconstructed images as numpy arrays
        output_dir: Directory to save visualizations
    """
    num_samples = len(real_images)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        real_img = np.transpose(real_images[i], (1, 2, 0))  # CHW -> HWC
        axes[i, 0].imshow(real_img)
        axes[i, 0].set_title(f"Original Image {i+1}")
        axes[i, 0].axis('off')
        
        # Reconstructed image
        recon_img = np.transpose(reconstructed_images[i], (1, 2, 0))  # CHW -> HWC
        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title(f"Reconstructed Image {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstructions.png'))
    plt.close()
    
    # Create a grid visualization for many samples
    if num_samples > 4:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(2, grid_size, figsize=(grid_size * 3, 6))
        
        for i in range(grid_size):
            for j in range(2):
                if j == 0:  # Original images
                    row_title = "Original"
                    img_list = real_images
                else:  # Reconstructed images
                    row_title = "Reconstructed"
                    img_list = reconstructed_images
                
                if i < len(img_list):
                    img = np.transpose(img_list[i], (1, 2, 0))  # CHW -> HWC
                    axes[j, i].imshow(img)
                    axes[j, i].set_title(f"{row_title} {i+1}")
                
                axes[j, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstructions_grid.png'))
        plt.close()

def visualize_latent_space(latents: np.ndarray, 
                          fmri_ids: List[int], 
                          output_dir: str,
                          perplexity: int = 30) -> None:
    """
    Visualize the latent space using t-SNE
    
    Args:
        latents: Latent representations as numpy array
        fmri_ids: List of fMRI IDs or indices
        output_dir: Directory to save visualizations
        perplexity: t-SNE perplexity parameter
    """
    # Apply t-SNE for dimensionality reduction
    logger.info(f"Applying t-SNE to latent space of shape {latents.shape}...")
    
    # Limit the number of samples for t-SNE if there are too many
    max_tsne_samples = 1000
    if latents.shape[0] > max_tsne_samples:
        logger.info(f"Too many samples for t-SNE, randomly selecting {max_tsne_samples}...")
        indices = np.random.choice(latents.shape[0], max_tsne_samples, replace=False)
        latents_subset = latents[indices]
        fmri_ids_subset = [fmri_ids[i] for i in indices]
    else:
        latents_subset = latents
        fmri_ids_subset = fmri_ids
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = tsne.fit_transform(latents_subset)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': latents_2d[:, 0],
        'y': latents_2d[:, 1],
        'fmri_id': fmri_ids_subset
    })
    
    # Plot the latent space
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='fmri_id', palette='viridis', alpha=0.7)
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Limit the number of legend items if there are too many
    if len(set(fmri_ids_subset)) > 20:
        plt.legend([], [], frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_space_tsne.png'))
    plt.close()
    
    # Save the latent representations for further analysis
    np.save(os.path.join(output_dir, 'latent_representations.npy'), latents)
    np.save(os.path.join(output_dir, 'fmri_ids.npy'), np.array(fmri_ids))

def generate_from_fmri(model: nn.Module, 
                      fmri_data: torch.Tensor, 
                      device: torch.device) -> torch.Tensor:
    """
    Generate images from fMRI data
    
    Args:
        model: The trained model
        fmri_data: fMRI data tensor
        device: Device to run on (cuda or cpu)
        
    Returns:
        Generated images as tensor
    """
    model.eval()
    with torch.no_grad():
        fmri_data = fmri_data.to(device)
        
        # Handle different model types
        if hasattr(model, 'generator'):  # GAN model
            output = model.generator(fmri_data)
        else:
            output = model(fmri_data)
        
        # Handle tuple output from variational models
        if isinstance(output, tuple):
            output = output[0]
    
    return output

def interpolate_latent_space(model: nn.Module, 
                            fmri_data1: torch.Tensor, 
                            fmri_data2: torch.Tensor, 
                            device: torch.device,
                            steps: int = 10) -> List[torch.Tensor]:
    """
    Interpolate between two fMRI samples in the latent space
    
    Args:
        model: The trained model
        fmri_data1: First fMRI data tensor
        fmri_data2: Second fMRI data tensor
        device: Device to run on (cuda or cpu)
        steps: Number of interpolation steps
        
    Returns:
        List of interpolated images
    """
    model.eval()
    with torch.no_grad():
        fmri_data1 = fmri_data1.to(device)
        fmri_data2 = fmri_data2.to(device)
        
        # Get latent representations
        if hasattr(model, 'generator'):  # GAN model
            if hasattr(model.generator, 'get_latent_representation'):
                z1 = model.generator.get_latent_representation(fmri_data1)
                z2 = model.generator.get_latent_representation(fmri_data2)
                
                # Interpolate in latent space
                interpolated_images = []
                for alpha in np.linspace(0, 1, steps):
                    z_interp = z1 * (1 - alpha) + z2 * alpha
                    
                    # Generate image from interpolated latent
                    img = model.generator.generate_from_latent(z_interp)
                    interpolated_images.append(img)
                
                return interpolated_images
            else:
                logger.warning("Model does not support latent space interpolation")
                return []
        else:
            if hasattr(model, 'get_latent_representation'):
                z1 = model.get_latent_representation(fmri_data1)
                z2 = model.get_latent_representation(fmri_data2)
                
                # Interpolate in latent space
                interpolated_images = []
                for alpha in np.linspace(0, 1, steps):
                    z_interp = z1 * (1 - alpha) + z2 * alpha
                    
                    # Generate image from interpolated latent
                    img = model.generate_from_latent(z_interp)
                    interpolated_images.append(img)
                
                return interpolated_images
            else:
                logger.warning("Model does not support latent space interpolation")
                return []

def visualize_interpolation(interpolated_images: List[torch.Tensor], 
                           output_dir: str,
                           filename: str = 'interpolation.png') -> None:
    """
    Visualize interpolation between two images
    
    Args:
        interpolated_images: List of interpolated images
        output_dir: Directory to save visualization
        filename: Filename for the visualization
    """
    if not interpolated_images:
        return
    
    num_images = len(interpolated_images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    
    if num_images == 1:
        axes = [axes]
    
    for i, img in enumerate(interpolated_images):
        img_np = img.cpu().squeeze(0).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        axes[i].imshow(img_np)
        axes[i].set_title(f"Step {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def evaluate_subject_differences(model: nn.Module, 
                               test_loader: DataLoader, 
                               device: torch.device,
                               output_dir: str) -> None:
    """
    Evaluate differences in reconstruction quality between subjects
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save results
    """
    # This function assumes the dataset provides subject information
    # If not available, this function would need to be modified
    
    if not hasattr(test_loader.dataset, 'get_subject_ids'):
        logger.warning("Dataset does not provide subject IDs, skipping subject difference analysis")
        return
    
    # Initialize metrics per subject
    subject_metrics = {}
    
    # Evaluation loop
    model.eval()
    mse_loss = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for fmri, target_img, subject_id in tqdm(test_loader, desc="Evaluating subjects"):
            fmri, target_img = fmri.to(device), target_img.to(device)
            
            # Get model output
            if hasattr(model, 'generator'):  # GAN model
                output = model.generator(fmri)
            else:
                output = model(fmri)
            
            # Handle tuple output from variational models
            if isinstance(output, tuple):
                output = output[0]
            
            # Calculate per-pixel MSE
            mse = mse_loss(output, target_img)
            
            # Average over all dimensions except batch
            mse = mse.mean(dim=[1, 2, 3])
            
            # Update metrics per subject
            for i, sid in enumerate(subject_id):
                if sid not in subject_metrics:
                    subject_metrics[sid] = []
                subject_metrics[sid].append(mse[i].item())
    
    # Calculate average metrics per subject
    subject_avg_mse = {sid: np.mean(mses) for sid, mses in subject_metrics.items()}
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    subjects = list(subject_avg_mse.keys())
    mses = list(subject_avg_mse.values())
    
    plt.bar(subjects, mses)
    plt.xlabel('Subject ID')
    plt.ylabel('Average MSE')
    plt.title('Reconstruction MSE by Subject')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_differences.png'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'subject_metrics.txt'), 'w') as f:
        for sid, mse in subject_avg_mse.items():
            f.write(f"Subject {sid}: MSE = {mse:.4f}\n")
    
    logger.info(f"Subject difference analysis completed and saved to {output_dir}")