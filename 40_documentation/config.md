# [config.py](../30_experiments/config.py)

Configuration class defining hyperparameters, data paths, and grid search values for both CNN and GFM models.

## Overview

A centralized configuration module that consolidates all hyperparameters, paths, and constants used across the training and evaluation pipeline. Manages device selection, batch sizes, learning rates, weight decays, and other training parameters for both BaselineCNN and TerraMindClassifier architectures.

## Core Components

* **`DEVICE`** Automatically selects CUDA if available, otherwise falls back to CPU. Used for all tensor operations and model training.

* **`DATA_ROOT`** Root directory path for the OilSlick dataset. Points to `../10_waterbench_data/data/OilSlick` containing GeoTIFF images and split definitions.

* **`OUTPUT_ROOT`** Root directory for saving outputs including trained models, logs, and evaluation results. Defaults to current directory.

* **`SEED`** Random seed for reproducibility across all libraries (random, numpy, torch). Set to 42.

## Training Parameters

### Early Stopping & Batch Configuration

* **`EARLY_STOPPING_PATIENCE`** Number of epochs to wait for validation loss improvement before stopping. Value: 4 epochs (applied after warmup phase).

* **`BATCH_SIZE`** Number of samples per batch during training and inference. Value: 16.

### Epochs Configuration

* **`EPOCHS_CNN`** Total training epochs for BaselineCNN. Value: 32.

* **`EPOCHS_GFM`** Total training epochs for TerraMindClassifier. Value: 16.

### Warmup Epochs

* **`WARMUP_EPOCHS_CNN`** Linear learning rate warmup epochs for CNN. Value: 3.

* **`WARMUP_EPOCHS_GFM`** Linear learning rate warmup epochs for GFM. Value: 2.

## Hyperparameter Grid Search

### CNN (BaselineCNN) Hyperparameters

* **`LEARNING_RATES_CNN`** Learning rate options for grid search. Values: `[1e-4, 5e-4, 1e-3]`.

* **`WEIGHT_DECAYS_CNN`** Weight decay options for grid search. Values: `[1e-4, 1e-3, 1e-2]`.

**Grid Size**: 3 learning rates × 3 weight decays = **9 subruns per configuration**

### GFM (TerraMindClassifier) Hyperparameters

* **`LEARNING_RATES_GFM`** Learning rate options for grid search. Values: `[5e-4, 1e-3, 3e-3]` (higher than CNN for fine-tuning pre-trained backbone).

* **`WEIGHT_DECAYS_GFM`** Weight decay options for grid search. Values: `[1e-5, 1e-4, 1e-3]` (lower than CNN for pre-trained models).

**Grid Size**: 3 learning rates × 3 weight decays = **9 subruns per configuration**

## Usage

All configuration values are accessed as class attributes throughout the training pipeline:

```python
from config import Config

device = Config.DEVICE
data_root = Config.DATA_ROOT
batch_size = Config.BATCH_SIZE
```

## Summary

| Parameter | CNN | GFM | Notes |
|-----------|-----|-----|-------|
| Epochs | 32 | 16 | GFM trains fewer epochs (pre-trained) |
| Warmup Epochs | 3 | 2 | Longer warmup for CNN from scratch |
| Batch Size | 16 | 16 | Same across both models |
| Early Stopping | 4 | 4 | Same patience threshold |
| Learning Rates | `[1e-4, 5e-4, 1e-3]` | `[5e-4, 1e-3, 3e-3]` | GFM uses higher LRs |
| Weight Decays | `[1e-4, 1e-3, 1e-2]` | `[1e-5, 1e-4, 1e-3]` | GFM uses lower decay |

## Dependencies

* **External:** `torch`

## Total Training Runs

- **CNN Configurations**: 2 splits × 9 subruns = 18 runs
- **GFM Configurations**: 2 splits × 9 subruns = 18 runs
- **Total**: 4 configurations × 9 subruns = **36 trained models**
