import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Union, Any

from dataset import Bold5000Dataset, AdaptiveBatchSampler
from model import FMRItoImageModel, FMRItoImageGAN
from train import train_model
from evaluate import evaluate_model, visualize_interpolation, interpolate_latent_space
from download_dataset import setup_bold5000

# Configure logging
def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fmri2image_{timestamp}.log")
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('fmri2image')

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='fMRI to Image Reconstruction')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='data/BOLD5000',
                        help='Directory containing the BOLD5000 dataset')
    parser.add_argument('--download', action='store_true',
                        help='Download the BOLD5000 dataset if not available')
    parser.add_argument('--subjects', type=str, nargs='+', default=['CSI1', 'CSI2', 'CSI3', 'CSI4'],
                        help='List of subjects to use from the dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size of the images (height and width)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'variational', 'gan'],
                        help='Type of model to use')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of the latent space')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512, 256],
                        help='Dimensions of hidden layers in the encoder')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    parser.add_argument('--use_residual', action='store_true',
                        help='Use residual blocks in the decoder')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--adaptive_batch', action='store_true',
                        help='Use adaptive batch sizing based on GPU memory')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--use_perceptual_loss', action='store_true',
                        help='Use perceptual loss (LPIPS) for training')
    parser.add_argument('--use_ssim_loss', action='store_true',
                        help='Use structural similarity index (SSIM) loss for training')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='Weight for KL divergence loss in variational models')
    
    # Data augmentation arguments
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation during training')
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='Apply random horizontal flips during augmentation')
    parser.add_argument('--rotation', action='store_true',
                        help='Apply random rotations during augmentation')
    parser.add_argument('--color_jitter', action='store_true',
                        help='Apply random color jitter during augmentation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    # GPU arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'interpolate'],
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a trained model for evaluation or interpolation')
    
    # Interpolation arguments
    parser.add_argument('--interpolate_samples', type=int, nargs=2, default=[0, 1],
                        help='Indices of two samples to interpolate between')
    parser.add_argument('--interpolate_steps', type=int, default=10,
                        help='Number of steps for interpolation')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_dir, args.log_level)
    logger.info(f"Starting fMRI to Image Reconstruction with arguments: {args}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Download dataset if requested
    if args.download:
        logger.info("Downloading BOLD5000 dataset...")
        setup_bold5000(selected_subjects=args.subjects, data_dir=args.data_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    
    # Configure data augmentation
    augmentation_config = None
    if args.augment:
        augmentation_config = {
            'horizontal_flip': args.horizontal_flip,
            'rotation': args.rotation,
            'color_jitter': args.color_jitter
        }
        logger.info(f"Using data augmentation with config: {augmentation_config}")
    
    # Create dataset
    dataset = Bold5000Dataset(
        data_dir=args.data_dir,
        subjects=args.subjects,
        img_size=args.img_size,
        augmentation=augmentation_config
    )
    
    # Split dataset
    logger.info(f"Dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
    
    # Create data loaders
    if args.adaptive_batch:
        logger.info("Using adaptive batch sizing")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=AdaptiveBatchSampler(train_dataset, base_batch_size=args.batch_size),
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=AdaptiveBatchSampler(val_dataset, base_batch_size=args.batch_size),
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=AdaptiveBatchSampler(test_dataset, base_batch_size=args.batch_size),
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Get fMRI dimension from dataset
    fmri_dim = dataset.get_fmri_dim()
    logger.info(f"fMRI dimension: {fmri_dim}")
    
    # Create or load model
    if args.model_path is None or args.mode == 'train':
        logger.info(f"Creating new model of type: {args.model_type}")
        
        # Create model based on type
        if args.model_type == 'gan':
            model = FMRItoImageGAN(
                fmri_dim=fmri_dim,
                latent_dim=args.latent_dim,
                img_channels=3,
                img_size=args.img_size,
                hidden_dims=args.hidden_dims,
                dropout_rate=args.dropout_rate
            )
            logger.info("Created GAN model")
        else:
            # For standard and variational models
            model = FMRItoImageModel(
                fmri_dim=fmri_dim,
                latent_dim=args.latent_dim,
                img_channels=3,
                img_size=args.img_size,
                hidden_dims=args.hidden_dims,
                dropout_rate=args.dropout_rate,
                use_residual=args.use_residual,
                variational=(args.model_type == 'variational')
            )
            logger.info(f"Created {'variational' if args.model_type == 'variational' else 'standard'} model")
    else:
        logger.info(f"Loading model from: {args.model_path}")
        
        # Determine model type from file or arguments
        if args.model_type == 'gan':
            model = FMRItoImageGAN(
                fmri_dim=fmri_dim,
                latent_dim=args.latent_dim,
                img_channels=3,
                img_size=args.img_size,
                hidden_dims=args.hidden_dims,
                dropout_rate=args.dropout_rate
            )
        else:
            model = FMRItoImageModel(
                fmri_dim=fmri_dim,
                latent_dim=args.latent_dim,
                img_channels=3,
                img_size=args.img_size,
                hidden_dims=args.hidden_dims,
                dropout_rate=args.dropout_rate,
                use_residual=args.use_residual,
                variational=(args.model_type == 'variational')
            )
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Move model to device
    model = model.to(device)
    
    # Run in specified mode
    if args.mode == 'train':
        logger.info("Starting training...")
        
        # Configure training
        train_config = {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'early_stopping_patience': args.early_stopping_patience,
            'checkpoint_dir': args.checkpoint_dir,
            'use_gan': args.model_type == 'gan',
            'use_perceptual_loss': args.use_perceptual_loss,
            'use_ssim_loss': args.use_ssim_loss,
            'kl_weight': args.kl_weight
        }
        
        # Train model
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=train_config
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir
        )
        
        logger.info(f"Test metrics: {test_metrics}")
    
    elif args.mode == 'evaluate':
        logger.info("Evaluating model...")
        
        # Evaluate on test set
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir
        )
        
        logger.info(f"Test metrics: {test_metrics}")
    
    elif args.mode == 'interpolate':
        logger.info("Interpolating between samples...")
        
        # Get samples for interpolation
        idx1, idx2 = args.interpolate_samples
        
        # Make sure indices are valid
        if idx1 >= len(test_dataset) or idx2 >= len(test_dataset):
            logger.error(f"Invalid sample indices. Dataset size: {len(test_dataset)}")
            return
        
        # Get fMRI data for interpolation
        fmri1, _ = test_dataset[idx1]
        fmri2, _ = test_dataset[idx2]
        
        # Add batch dimension
        fmri1 = fmri1.unsqueeze(0)
        fmri2 = fmri2.unsqueeze(0)
        
        # Interpolate
        interpolated_images = interpolate_latent_space(
            model=model,
            fmri_data1=fmri1,
            fmri_data2=fmri2,
            device=device,
            steps=args.interpolate_steps
        )
        
        # Visualize interpolation
        if interpolated_images:
            visualize_interpolation(
                interpolated_images=interpolated_images,
                output_dir=args.output_dir,
                filename=f"interpolation_{idx1}_{idx2}.png"
            )
            logger.info(f"Interpolation saved to {args.output_dir}/interpolation_{idx1}_{idx2}.png")
        else:
            logger.error("Interpolation failed. Model may not support latent space operations.")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()