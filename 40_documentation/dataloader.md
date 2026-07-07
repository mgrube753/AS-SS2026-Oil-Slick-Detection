# [dataloader.py](../30_experiments/dataloader.py)

Creates DataLoaders for training, validation, and testing with Sentinel-1 SAR-specific z-score normalization and augmentation transforms.

## Overview

A comprehensive data loading module that orchestrates the complete data pipeline for training and evaluation. Implements custom SAR-specific normalization (z-score with nodata masking and percentile clipping), augmentation transforms, and provides factory functions for creating PyTorch DataLoaders with proper train/validation/test splits.

## Core Components

### Custom Transforms

* **`SARzScore`** Custom transform class for SAR-specific normalization:
  - **Nodata masking**: Pixels ≤ −50 dB or NaN are identified and set to 0
  - **Percentile clipping**: Valid pixels clipped to 2nd–98th percentile range
  - **Z-score normalization**: (x − mean) / (std + 1e−6) using training set statistics
  - **Prevents data leakage**: Statistics derived from training set only
  - Applied per-channel (VV and VH bands independently)

* **`get_train_transform(train_stats)`** Composition of training transforms:
  1. Resize to 224×224 with antialiasing
  2. SARzScore normalization
  3. Random horizontal flip (p=0.5)
  4. Random vertical flip (p=0.5)
  5. Random 90° rotations: 0°, 90°, 180°, 270° with equal probability
  **Returns:** `transforms.Compose` object for training data

* **`get_val_test_transform(train_stats)`** Composition of validation/test transforms:
  1. Resize to 224×224 with antialiasing
  2. SARzScore normalization (no augmentation)
  **Returns:** `transforms.Compose` object for validation/test data

### Rotation Transforms

* **`rotate0(x)`** No rotation (0°) - identity transform
* **`rotate90(x)`** 90° counterclockwise rotation
* **`rotate180(x)`** 180° rotation
* **`rotate270(x)`** 270° counterclockwise rotation (or 90° clockwise)

## Core Functions

* **`load_split_stats(split_type)`** Loads pre-computed training set statistics (mean, std per channel) from `../20_data_analysis/split_stats.json`. **Returns:** dict with per-channel statistics for normalization.

* **`get_train_val_loaders(data_root, batch_size, split_type, seed)`** Creates training and validation DataLoaders:
  1. Loads training set statistics
  2. Creates train and validation datasets with appropriate transforms
  3. Initializes DataLoaders with batch_size, shuffling, and multi-worker support
  - **Parameters**: data_root, batch_size (default 16), split_type ("random"/"geographic"), seed (default 42)
  - **Returns:** tuple of (train_loader, val_loader, train_dataset)

* **`get_test_loader(data_root, batch_size, split_type, seed)`** Creates test DataLoader:
  1. Loads training set statistics
  2. Creates test dataset with validation transforms (no augmentation)
  3. Initializes DataLoader with batch_size and multi-worker support
  - **Parameters**: data_root, batch_size (default 16), split_type ("random"/"geographic"), seed (default 42)
  - **Returns:** tuple of (test_loader, test_dataset)

## Transform Pipeline

| Stage | Training | Validation/Test | Purpose |
|-------|----------|-----------------|---------|
| **1. Resize** | 224×224 | 224×224 | Standardize input dimensions |
| **2. SAR Normalization** | Percentile clip + Z-score | Percentile clip + Z-score | SAR-specific preprocessing |
| **3. Horizontal Flip** | p=0.5 | None | Data augmentation |
| **4. Vertical Flip** | p=0.5 | None | Data augmentation |
| **5. Random Rotation** | 0°/90°/180°/270° | None | Rotational invariance |

## DataLoader Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 16 | Balanced memory usage |
| Shuffle (Train) | True | Randomize batch order |
| Shuffle (Val/Test) | False | Consistent evaluation |
| Num Workers | 8 | Parallel data loading |
| Pin Memory | True (GPU) / False (CPU) | Memory efficiency |
| Generator | Seeded (seed=42) | Reproducibility |

## Split Data Sizes

### Random Split

| Subset | Samples | Batches (size=16) |
|--------|---------|------------------|
| Train | 793 | 50 |
| Validation | 127 | 8 |
| Test | 266 | 17 |

### Geographic Split

| Subset | Samples | Batches (size=16) |
|--------|---------|------------------|
| Train | 749 | 47 |
| Validation | 189 | 12 |
| Test | 141 | 9 |

## SAR Normalization Details

### Nodata Handling
- Pixels marked as nodata (≤ −50 dB or NaN) are excluded from statistics
- Nodata pixels set to 0 after normalization

### Percentile Clipping
- Valid pixel range: 2nd to 98th percentile
- Reduces impact of outliers/speckle noise in SAR data

### Z-Score Normalization
- Formula: `z = (x - mean) / (std + 1e-6)`
- Small epsilon (1e-6) prevents division by zero
- Training set statistics prevent data leakage

## Dependencies

* **Internal:** `dataset`, `config`
* **External:** `torch`, `torchvision.transforms`, `numpy`, `json`, `os`

## Usage Example

```python
from dataloader import get_train_val_loaders, get_test_loader
from config import Config

# Create training and validation loaders
train_loader, val_loader, train_ds = get_train_val_loaders(
    data_root=Config.DATA_ROOT,
    batch_size=Config.BATCH_SIZE,
    split_type="random",
    seed=Config.SEED
)

# Create test loader
test_loader, test_ds = get_test_loader(
    data_root=Config.DATA_ROOT,
    batch_size=Config.BATCH_SIZE,
    split_type="random",
    seed=Config.SEED
)

# Iterate through batches
for images, labels in train_loader:
    # images: [16, 2, 224, 224]
    # labels: [16]
    pass
```

## Key Features

- **SAR-specific preprocessing**: Handles Sentinel-1 characteristics (dB scale, speckle)
- **Data augmentation**: Rotation invariance for satellite imagery
- **Reproducibility**: Seeded generators for consistent splits
- **Multi-worker loading**: Efficient parallel data loading
- **Memory management**: Pin memory for GPU acceleration
- **No data leakage**: Statistics derived from training set only
