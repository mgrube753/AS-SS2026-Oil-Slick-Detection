# [**`30_experiments`**](../30_experiments/)

This directory contains the model training, evaluation, and analysis pipeline for oil slick detection using two different architectures: a baseline CNN trained from scratch and a pre-trained TerraMind foundation model (GFM) with a custom classification head.

## Contents

| File | Description |
|---|---|
| [`config.py`](config.py) | Hyperparameter configuration and grid search parameters |
| [`dataloader.py`](dataloader.py) | Custom DataLoader with SAR-specific z-score normalization transform |
| [`dataset.py`](dataset.py) | OilSlickDataset class for loading filtered GeoTIFF images and labels |
| [`model.py`](model.py) | Model architectures: BaselineCNN and TerraMindClassifier |
| [`train.py`](train.py) | Training loop with early stopping, MLflow logging, and checkpointing |
| [`runner.py`](runner.py) | Argument-based runner script for grid search execution |
| [`eval.py`](eval.py) | Evaluation script for test set inference and metrics computation |
| [`failure_analysis.py`](failure_analysis.py) | False positive and false negative visualization |
| [`utils.py`](utils.py) | Utility functions for metric calculation and seed management |

## Experimental Setup

The experiments follow a systematic grid search over hyperparameters for both model types and split strategies:

### Model Configurations

**BaselineCNN (CNN)**
- Small 2-layer convolutional network trained from scratch
- Input: 2-channel SAR images (VV + VH), resized to 224×224
- Output: Binary classification (oil slick / no slick)
- Kaiming weight initialization

**TerraMindClassifier (GFM)**
- Pre-trained TerraMind v1 Small backbone (frozen by default)
- Modality: Sentinel-1 GRD (S1GRD)
- Custom 2-layer classification head with dropout
- Output: Binary classification

### Hyperparameter Grid

| Parameter | CNN | GFM |
|---|---|---|
| Learning Rate | `[1e-4, 5e-4, 1e-3]` | `[5e-4, 1e-3, 3e-3]` |
| Weight Decay | `[1e-4, 1e-3, 1e-2]` | `[1e-5, 1e-4, 1e-3]` |
| Epochs | 32 | 16 |
| Warmup Epochs | 3 | 2 |
| Batch Size | 16 | 16 |
| Early Stopping Patience | 4 (after warmup) | 4 (after warmup) |

**Total runs per configuration**: 3 LR × 3 WD = **9 subruns** per (model, split) combination

### Split Types

1. **Random Split** (In-distribution)
   - Train: 793 samples
   - Val: 127 samples
   - Test: 266 samples

2. **Geographic Split** (Out-of-distribution evaluation)
   - Train: 749 samples
   - Val: 189 samples
   - Test: 141 samples

## Data Pipeline

### 1. Data Loading & Preprocessing

- Loads filtered metadata from [`../20_data_analysis/filtered_metadata.csv`](../20_data_analysis/filtered_metadata.csv)
- Filters to only valid samples (`valid_sample == True`)
- Loads 2-band GeoTIFF chips from [`../10_waterbench_data/data/OilSlick/images_s1/`](../10_waterbench_data/data/OilSlick/images_s1/)

### 2. SAR-Specific Normalization

Custom `SARzScore` transform applies:
- **Nodata masking**: Pixels ≤ −50 dB or NaN are set to 0
- **Percentile clipping**: Valid pixels clipped to 2nd–98th percentile
- **Z-score normalization**: (x − mean) / (std + 1e−6)
- **Statistics source**: Computed per-split from training set only (prevents data leakage)

### 3. Data Augmentation (Training Only)

Applied during training for BaselineCNN and TerraMindClassifier:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotations (0°, 90°, 180°, 270°)

Validation and test sets use only normalization (no augmentation).

### 4. Class Imbalance Handling

- Computed `pos_weight` from training set class distribution
- Used in `BCEWithLogitsLoss` to balance binary cross-entropy loss

## Training Workflow

### Scheduler & Optimization

1. **Warmup phase** (Linear LR ramp):
   - CNN: 3 epochs (0.1× → 1.0× learning rate)
   - GFM: 2 epochs

2. **Annealing phase** (Cosine decay):
   - Remaining epochs with cosine annealing to η_min = LR × 0.01

3. **Optimizer**: AdamW with specified learning rate and weight decay

### Early Stopping

- Monitored metric: validation loss
- Patience: 4 epochs (after warmup phase)
- Best model saved when validation loss improves

### Checkpointing

- **Best model**: Saved when validation loss reaches new minimum
- **Final model**: Saved at end of training
- Both stored in `logs/<split>/<model>/lr<LR>_wd<WD>/`

### MLflow Logging

All runs logged to `logs/mlflow/` with:
- Parameters: model, split, LR, weight decay, batch size, warmup, epochs, criterion, device
- Metrics per epoch: train/val loss, learning rate, accuracy, precision, recall, F1, AUC-ROC
- Artifacts: best and final model checkpoints

## Running Experiments

### Quick Start

```bash
# Train BaselineCNN with random split (grid search over 9 hyperparameter combinations)
python runner.py --model-name baselinecnn --split-type random

# Train TerraMind with geographic split
python runner.py --model-name terramind --split-type geographic