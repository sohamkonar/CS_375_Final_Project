import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import gc

# Configure logging
logger = logging.getLogger("bold5000_dataset")

class Bold5000Dataset(Dataset):
    """
    Dataset for the BOLD5000 fMRI dataset
    
    Args:
        data_dir: Directory containing the BOLD5000 dataset
        subjects: List of subjects to include (e.g., ['CSI1', 'CSI2'])
        img_size: Size to resize images to (square)
        augmentation: Dictionary of augmentation options
    """
    def __init__(self, 
                 data_dir: str, 
                 subjects: List[str] = None,
                 img_size: int = 128,
                 augmentation: Dict[str, bool] = None):
        
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Default subjects if none provided
        if subjects is None:
            subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
        self.subjects = subjects
        
        # Set up paths
        self.fmri_dir = os.path.join(data_dir, 'GLMbetas')
        self.image_dir = os.path.join(data_dir, 'stimuli')
        self.metadata_path = os.path.join(data_dir, 'trials_metadata.csv')
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Filter by subjects
        if subjects:
            self.metadata = self.metadata[self.metadata['subject'].isin(subjects)]
        
        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)
        
        logger.info(f"Loaded metadata with {len(self.metadata)} samples")
        
        # Set up image transforms
        self.transform = self._get_transforms(img_size, augmentation)
        
        # Cache for fMRI data
        self.fmri_cache = {}
        
        # Load first fMRI sample to get dimensionality
        first_subject = self.metadata.iloc[0]['subject']
        first_session = self.metadata.iloc[0]['session']
        first_trial = self.metadata.iloc[0]['trial']
        
        first_fmri_path = os.path.join(
            self.fmri_dir, 
            first_subject, 
            f"sess-{first_session}_trial-{first_trial}.npy"
        )
        
        if os.path.exists(first_fmri_path):
            first_fmri = np.load(first_fmri_path)
            self.fmri_dim = first_fmri.shape[0]
            logger.info(f"fMRI dimension: {self.fmri_dim}")
        else:
            raise FileNotFoundError(f"Could not find fMRI file: {first_fmri_path}")
    
    def _get_transforms(self, img_size: int, augmentation: Dict[str, bool] = None) -> Callable:
        """
        Get image transforms based on configuration
        
        Args:
            img_size: Size to resize images to
            augmentation: Dictionary of augmentation options
            
        Returns:
            Transforms function
        """
        # Base transforms that are always applied
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
        
        # Add augmentations if specified
        if augmentation:
            aug_transforms = []
            
            if augmentation.get('horizontal_flip', False):
                aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if augmentation.get('rotation', False):
                aug_transforms.append(transforms.RandomRotation(10))
            
            if augmentation.get('color_jitter', False):
                aug_transforms.append(
                    transforms.ColorJitter(
                        brightness=0.1, 
                        contrast=0.1, 
                        saturation=0.1, 
                        hue=0.1
                    )
                )
            
            # Combine base transforms with augmentations
            transform = transforms.Compose(aug_transforms + base_transforms)
            logger.info(f"Using augmented transforms: {augmentation}")
        else:
            transform = transforms.Compose(base_transforms)
            logger.info("Using base transforms without augmentation")
        
        return transform
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (fMRI data, image)
        """
        # Get metadata for this sample
        sample_meta = self.metadata.iloc[idx]
        subject = sample_meta['subject']
        session = sample_meta['session']
        trial = sample_meta['trial']
        image_name = sample_meta['stimulus']
        
        # Load fMRI data
        fmri_path = os.path.join(
            self.fmri_dir, 
            subject, 
            f"sess-{session}_trial-{trial}.npy"
        )
        
        # Check cache first
        if fmri_path in self.fmri_cache:
            fmri_data = self.fmri_cache[fmri_path]
        else:
            try:
                fmri_data = np.load(fmri_path)
                # Cache the data
                self.fmri_cache[fmri_path] = fmri_data
            except Exception as e:
                logger.error(f"Error loading fMRI data from {fmri_path}: {e}")
                # Return a zero tensor as fallback
                fmri_data = np.zeros(self.fmri_dim, dtype=np.float32)
        
        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            # Return a zero tensor as fallback
            image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        
        # Convert fMRI data to tensor
        fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
        
        return fmri_tensor, image
    
    def get_fmri_dim(self) -> int:
        """
        Get the dimensionality of the fMRI data
        
        Returns:
            Dimension of fMRI data
        """
        return self.fmri_dim
    
    def get_subject_ids(self) -> List[str]:
        """
        Get the list of subject IDs in the dataset
        
        Returns:
            List of subject IDs
        """
        return self.subjects
    
    def clear_cache(self) -> None:
        """Clear the fMRI data cache to free memory"""
        self.fmri_cache.clear()
        gc.collect()

class AdaptiveBatchSampler(Sampler):
    """
    Adaptive batch sampler that adjusts batch size based on available GPU memory
    
    Args:
        dataset: Dataset to sample from
        base_batch_size: Initial batch size to try
        max_memory_usage: Maximum fraction of GPU memory to use (0.0-1.0)
    """
    def __init__(self, 
                dataset: Dataset, 
                base_batch_size: int = 32, 
                max_memory_usage: float = 0.8):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.max_memory_usage = max_memory_usage
        self.batch_size = self._find_optimal_batch_size()
        self.num_samples = len(dataset)
        
        # Calculate number of batches
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
    def _find_optimal_batch_size(self) -> int:
        """
        Find the optimal batch size based on available GPU memory
        
        Returns:
            Optimal batch size
        """
        # Start with base batch size
        batch_size = self.base_batch_size
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.info(f"CUDA not available, using base batch size: {batch_size}")
            return batch_size
        
        try:
            # Try to allocate a batch and see if it fits in memory
            sample_fmri, sample_img = self.dataset[0]
            
            # Get memory stats
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_allowed_memory = total_memory * self.max_memory_usage
            
            # Estimate memory per sample
            fmri_memory = sample_fmri.element_size() * sample_fmri.nelement()
            img_memory = sample_img.element_size() * sample_img.nelement()
            
            # Add some overhead for model parameters and intermediate activations
            memory_per_sample = (fmri_memory + img_memory) * 3
            
            # Calculate max batch size
            max_batch_size = int(max_allowed_memory / memory_per_sample)
            
            # Limit batch size to reasonable range
            batch_size = min(max_batch_size, self.base_batch_size * 4)
            batch_size = max(batch_size, 1)
            
            logger.info(f"Adaptive batch size: {batch_size} (base: {self.base_batch_size})")
            
        except Exception as e:
            logger.warning(f"Error determining optimal batch size: {e}")
            logger.info(f"Using base batch size: {batch_size}")
        
        return batch_size
    
    def __iter__(self):
        # Create batches
        indices = torch.randperm(self.num_samples).tolist()
        
        # Yield batches
        for i in range(0, self.num_samples, self.batch_size):
            batch = indices[i:i + self.batch_size]
            yield batch
    
    def __len__(self):
        return self.num_batches