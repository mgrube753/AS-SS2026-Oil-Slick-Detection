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

- Small 4-layer convolutional network trained from scratch
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

Run all 4 training configurations (each performs a grid search over 9 hyperparameter combinations):

```bash
# 1. Train BaselineCNN with random split (9 subruns)
python runner.py --model-name baselinecnn --split-type random

# 2. Train BaselineCNN with geographic split (9 subruns)
python runner.py --model-name baselinecnn --split-type geographic

# 3. Train TerraMind with random split (9 subruns)
python runner.py --model-name terramind --split-type random

# 4. Train TerraMind with geographic split (9 subruns)
python runner.py --model-name terramind --split-type geographic
```

**Total**: 4 configurations × 9 subruns = **36 trained models**

### Grid Search Execution

Each command above performs grid search over the hyperparameter combinations defined in `config.py`. Training progress and metrics are logged to MLflow in `logs/mlflow/`.

### Evaluation & Analysis

By running [`eval.py`](./eval.py), the best 4 models (one per training configuration) are firstly obtained by selecting the checkpoint with the lowest validation loss. The script then evaluates these models on the designated test set, computing metrics and generating visualizations.

Afterwards, the [`failure_analysis.py`](./failure_analysis.py) script can be run to explicitly visualize false positives and false negatives, providing insight into model performance and potential areas for improvement.

```bash
# Evaluate best checkpoint per (model, split) configuration on test set
python eval.py

# Outputs:
# - Test metrics (accuracy, precision, recall, F1, AUC-ROC)
# - Confusion matrices
# - Probability histograms
# - Classification reports
# - Prediction CSVs (../50_evaluation/)

# Analyze false positives and false negatives
python failure_analysis.py

# Outputs:
# - Visualized FP/FN examples with both VV and VH channels
# - Saved to ../50_evaluation/
```

## Output Structure

After complete training and evaluation:

```bash
30_experiments/
├── config.py
├── dataloader.py
├── dataset.py
├── model.py
├── train.py
├── runner.py
├── eval.py
├── failure_analysis.py
├── utils.py
├── README.md
└── logs/
    ├── mlflow/                                  # MLflow tracking data
    │   └── <experiment-id>/<run-id>/...
    ├── random_split/
    │   ├── cnn/
    │   │   └── models/lr<LR>_wd<WD>/
    │   │       ├── best_model.pth
    │   │       └── final_model.pth
    │   └── gfm/
    │       └── models/lr<LR>_wd<WD>/
    │           ├── best_model.pth
    │           └── final_model.pth
    └── geographic_split/
        ├── cnn/
        │   └── models/lr<LR>_wd<WD>/
        │       ├── best_model.pth
        │       └── final_model.pth
        └── gfm/
            └── models/lr<LR>_wd<WD>/
                ├── best_model.pth
                └── final_model.pth
```

## Configuration Details

### Hardware & Reproducibility

- **Device**: Automatically selects CUDA if available, falls back to CPU
- **Seed**: Fixed at 42 for reproducibility across all runs
- **Mixed Precision**: Not used; full float32 precision
- **Number of Workers**: 8 for data loading

### Key Dependencies

- PyTorch with CUDA support
- Torchvision (transforms, models)
- TerraTorch (pre-trained backbone registry)
- scikit-learn (metrics)
- MLflow (experiment tracking)
- tifffile (GeoTIFF reading)
- pandas (metadata handling)

## Results & Interpretation

### Metrics Computed

For each model on the test set:

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy (minimize false alarms)
- **Recall**: True positive rate (minimize missed detections)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating curve
- **Confusion Matrix**: TP, TN, FP, FN breakdown

### Failure Analysis

For each configuration, false positives and false negatives are visualized to identify:

- Misclassified ambiguous regions
- Model-specific weaknesses
- Potential dataset annotation issues

## Notes

- All statistics (mean, std) derived from **training set only** to prevent data leakage
- Validation set used for early stopping; not for hyperparameter selection
- Test set completely held out until final evaluation
- Geographic split serves as an out-of-distribution evaluation benchmark
