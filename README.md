# CS 375 Final Project: fMRI-to-Image Reconstruction using BOLD5000

This project implements a neural decoder that reconstructs visual stimuli from fMRI brain activity using the complete BOLD5000 dataset.

## About BOLD5000

[BOLD5000](https://bold5000.github.io/) is a large-scale fMRI dataset containing brain responses to 5,000 distinct images from 4 subjects across multiple sessions. This implementation uses Release 2.0 of the dataset, which includes optimized GLM beta maps.

## Features

- Automatic download and setup of the complete BOLD5000 dataset
- Efficient memory handling for processing large fMRI data files
- Deep learning model for fMRI-to-image reconstruction
- Multi-subject and multi-session support
- Training and evaluation pipelines with GPU acceleration
- Visualization tools for model results

## Installation

1. Clone this repository:
```
git clone https://github.com/sohamkonar/CS_375_Final_Project.git
cd CS_375_Final_Project
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. (Recommended) If you have a CUDA-capable GPU, ensure you have the appropriate CUDA toolkit installed.

## Project Structure

- `download_dataset.py`: Downloads and extracts the complete BOLD5000 dataset
- `dataset.py`: Contains the PyTorch dataset class with memory-efficient data loading
- `model.py`: Defines the neural network architecture for fMRI-to-image reconstruction
- `train.py`: Contains the training loop and related functions
- `evaluate.py`: Provides functions for evaluating and visualizing model results
- `main.py`: Main script that coordinates the entire pipeline

## Usage

### Basic Usage

To download the complete dataset and run the full experiment:

```
python main.py
```

**Note**: The full dataset is approximately 200 GB and requires significant time to download and process.

### Command Line Arguments

- `--download`: Force download of the dataset even if files exist
- `--subjects`: List of subjects to use (default: all subjects)
  - Example: `--subjects CSI1 CSI3` to use only subjects CSI1 and CSI3
- `--sessions`: List of sessions to include (default: all sessions)
  - Example: `--sessions ses01 ses02` to use only sessions 1 and 2
- `--subset_size`: Limit dataset size (default: 0, which uses the full dataset)
- `--memory_mapping`: Use memory mapping for large datasets (default: True)
- `--latent_dim`: Size of latent space dimension (default: 256)
- `--eval_only`: Skip training and only evaluate an existing model
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--cpu`: Force CPU usage even if GPU is available
- `--num_workers`: Number of data loading workers (default: 4)

### Examples

Train using only subject CSI1, all sessions:
```
python main.py --subjects CSI1
```

Use a specific subset of sessions:
```
python main.py --subjects CSI1 CSI2 --sessions ses01 ses02 ses03
```

Use a smaller subset of the data for faster experimentation:
```
python main.py --subset_size 1000
```

## Hardware Requirements

Processing the full BOLD5000 dataset requires:

- At least 16GB RAM (32GB recommended)
- 250GB+ disk space
- CUDA-capable GPU with 8GB+ VRAM for efficient training
- High-speed internet connection for downloading the dataset

## Data Processing Details

This implementation:
- Uses BOLD5000 Release 2.0 TYPED-FITHRF-GLMDENOISE-RR beta maps (recommended version)
- Applies proper z-score normalization as specified in the BOLD5000 documentation
- Implements efficient memory mapping for handling large datasets
- Supports parallel data loading for improved training speed

## Performance Optimizations

For large datasets, this implementation includes:
- Memory mapping for efficient data access without loading everything into RAM
- Parallel data loading with multiple worker processes
- Automatic batch size adjustment for GPU memory constraints
- Selective subject/session loading for experimenting with subsets of data

## Citations

If you use this code or the BOLD5000 dataset, please cite:

```
@article{chang2019bold5000,
  title={BOLD5000, a public fMRI dataset while viewing 5000 visual images},
  author={Chang, Nadine and Pyles, John A and Marcus, Austin and Gupta, Abhinav and Tarr, Michael J and Aminoff, Elissa M},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={49},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## Troubleshooting

### Storage Space

The complete dataset requires approximately 200GB of storage. If space is limited:
1. Use `--subjects` and `--sessions` to download only specific portions
2. Use a smaller subset with `--subset_size`

### Memory Usage

If you encounter memory errors:
1. Ensure `--memory_mapping` is enabled (it's on by default)
2. Reduce batch size with `--batch_size`
3. Process fewer subjects/sessions at once

### Training Time

The full dataset will take several days to train on consumer hardware. To speed up:
1. Use a powerful GPU
2. Start with a smaller data subset to validate your approach
3. Increase the number of workers with `--num_workers` if you have many CPU cores
```