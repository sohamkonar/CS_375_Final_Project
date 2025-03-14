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

### Downloading the Dataset

#### Download the Complete Dataset

To download the entire BOLD5000 dataset (approximately 200 GB):

```
python download_dataset.py
```

This will:
1. Create a `data/` directory in the project folder
2. Download all subject data from the BOLD5000 repository
3. Extract and organize files by subject and session
4. Verify the integrity of downloaded files

#### Download a Subset of the Dataset

To download data for specific subjects only:

```
python download_dataset.py --subjects CSI1 CSI3
```

To download specific sessions for selected subjects:

```
python download_dataset.py --subjects CSI1 CSI3 --sessions ses01 ses03 ses05
```

To download a limited number of samples (useful for testing):

```
python download_dataset.py --subset_size 500
```

This will download only the first 500 samples across the specified subjects/sessions.

### Running Experiments

#### Full Experiment

To run a complete training and evaluation experiment using all available data:

```
python main.py
```

This will:
1. Check if the dataset exists (downloading if necessary)
2. Preprocess the fMRI data
3. Train the reconstruction model
4. Evaluate performance and generate visualizations
5. Save the trained model and results

#### Quick Test Experiment

For a quick test to verify your setup:

```
python main.py --subjects CSI1 --sessions ses01 --subset_size 100 --epochs 5
```

This runs a small experiment using only 100 samples from subject CSI1's first session with just 5 training epochs.

#### Cross-Subject Experiment

To train on some subjects and test on others:

```
python main.py --train_subjects CSI1 CSI2 --test_subjects CSI3 --epochs 30
```

#### Hyperparameter Tuning Example

To experiment with different model configurations:

```
python main.py --subjects CSI1 CSI2 --latent_dim 512 --batch_size 32 --lr 5e-5
```

#### Evaluation Only

To evaluate a previously trained model without retraining:

```
python main.py --eval_only --model_path ./models/my_trained_model.pth
```

### Advanced Usage Examples

#### Memory-Efficient Processing

For machines with limited RAM:

```
python main.py --memory_mapping --batch_size 8 --num_workers 2
```

#### High-Performance Setup

For machines with powerful GPUs and plenty of RAM:

```
python main.py --batch_size 64 --num_workers 8
```

#### Resuming Training

To continue training from a checkpoint:

```
python main.py --resume --checkpoint_path ./checkpoints/checkpoint_epoch20.pth
```

#### Custom Data Split

To specify a custom train/validation/test split:

```
python main.py --train_split 0.7 --val_split 0.15 --test_split 0.15
```

### Visualizing Results

To generate visualizations of reconstructed images:

```
python visualize.py --model_path ./models/trained_model.pth --num_samples 20
```

This will create a grid of original vs. reconstructed images for the specified number of test samples.

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

## Typical Workflow Examples

### Example 1: First-time Complete Setup

```bash
# Clone the repository
git clone https://github.com/sohamkonar/CS_375_Final_Project.git
cd CS_375_Final_Project

# Install dependencies
pip install -r requirements.txt

# Download the complete dataset (this will take several hours)
python download_dataset.py

# Run the full experiment (this will take several days on consumer hardware)
python main.py
```

### Example 2: Quick Pilot Study

```bash
# Download a small subset of the data
python download_dataset.py --subjects CSI1 --sessions ses01 --subset_size 200

# Run a quick experiment to test your setup
python main.py --subjects CSI1 --sessions ses01 --subset_size 200 --epochs 10 --batch_size 16
```

### Example 3: Incremental Approach

```bash
# Start with one subject
python download_dataset.py --subjects CSI1

# Train on the first subject
python main.py --subjects CSI1 --epochs 20

# Download additional subjects
python download_dataset.py --subjects CSI2 CSI3

# Train on all downloaded subjects
python main.py --subjects CSI1 CSI2 CSI3 --epochs 30
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

### Common Errors and Solutions

#### Dataset Download Errors

If you encounter download interruptions:
```
python download_dataset.py --resume --verify
```
This will resume downloading and verify existing files.

#### CUDA Out of Memory

If you see "CUDA out of memory" errors:
```
python main.py --batch_size 8 --memory_mapping
```
Try progressively smaller batch sizes until it works.

#### Slow Training

If training is too slow:
```
python main.py --subjects CSI1 --sessions ses01 ses02 --num_workers 8
```
Start with a smaller dataset and optimize data loading parameters.

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