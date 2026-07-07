# [train.py](../30_experiments/train.py)

Model- and split-specific training script with epoch loops for training and validation, early stopping, model checkpointing, and MLflow logging.

## Overview

A comprehensive training module orchestrating the complete training workflow for both BaselineCNN and TerraMindClassifier models across random and geographic splits. Implements learning rate scheduling with warmup, early stopping, checkpoint management, and MLflow experiment tracking for all hyperparameter configurations.

## Core Functions

* **`run_training(model_name, split_type, lr, wd, total_runs, counter)`** Orchestrates complete training pipeline: initializes model → sets up optimizer and scheduler → runs epoch loop with early stopping → saves checkpoints and logs to MLflow. Supports both "baselinecnn" and "terramind" model types. **Returns:** None (artifacts saved to disk and MLflow).

* **`train_epoch(model, loader, optimizer, criterion, device)`** Executes one training epoch: forward passes → compute loss → backward pass → optimizer step. Accumulates batch losses and predictions for metric calculation. **Returns:** tuple of (epoch_loss, metrics_dict) with accuracy, precision, recall, F1, and AUC-ROC.

* **`validate_epoch(model, loader, criterion, device)`** Executes one validation epoch: forward passes in eval mode → compute loss without gradients. Accumulates predictions for metric calculation. **Returns:** tuple of (epoch_loss, metrics_dict).

## Key Features

### Learning Rate Scheduling

- **Warmup Phase**: Linear LR ramp from 0.1× to 1.0× learning rate
  - CNN: 3 epochs
  - GFM: 2 epochs
- **Annealing Phase**: Cosine annealing decay to η_min = LR × 0.01
  - CNN: 29 remaining epochs
  - GFM: 14 remaining epochs
- **Optimizer**: AdamW with specified learning rate and weight decay

### Early Stopping

- **Monitored Metric**: Validation loss
- **Patience**: 4 epochs (applied after warmup phase only)
- **Best Model**: Automatically saved when validation loss reaches new minimum

### Checkpointing

- **Best Model**: Saved to `best_model.pth` when validation loss improves
- **Final Model**: Saved to `final_model.pth` at training end
- Both include: epoch, model state dict, optimizer state dict, validation loss

### MLflow Logging

All training runs logged to `logs/mlflow/` with:
- **Parameters**: model, split, learning rate, weight decay, batch size, warmup epochs, total epochs, pos_weight, criterion, device
- **Metrics per Epoch**: train/val loss, learning rate, accuracy, precision, recall, F1, AUC-ROC
- **Artifacts**: best and final model checkpoints
- **Tags**: early stopping reason (if applicable)

## Dependencies

* **Internal:** `dataloader`, `model`, `config`, `utils`
* **External:** `torch`, `torch.nn`, `torch.optim`, `mlflow`

## Output Structure

Checkpoints saved to:
```
logs/<split_folder>/<model_folder>/models/lr<LR>_wd<WD>/
├── best_model.pth
└── final_model.pth
```

Where:
- `<split_folder>`: "random_split" or "geographic_split"
- `<model_folder>`: "cnn" or "gfm"
- `<LR>`: Learning rate value
- `<WD>`: Weight decay value

MLflow experiments organized by model and split type in `logs/mlflow/`.
