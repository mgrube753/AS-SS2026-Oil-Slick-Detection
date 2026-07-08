# [utils.py](../30_experiments/utils.py)

Utility module providing helper functions for training/validation loops, metric calculation, seed setting, and class imbalance handling.

## Overview

A collection of essential utility functions for the oil slick detection pipeline. This module encompasses seed management for reproducibility, metric computation for model evaluation, epoch-level training and validation loops, and class imbalance mitigation through positive weight calculation.

## Core Functions

* **`set_seed(seed=42)`** Sets random seeds across Python, NumPy, PyTorch CPU and CUDA to ensure reproducibility. Configures cuDNN for deterministic behavior. **Parameters:** seed (int, default=42). **Returns:** None.

* **`calc_metrics(probs, labels)`** Computes classification metrics from predicted probabilities and true labels. Applies 0.5 threshold for binary predictions. **Returns:** dict with accuracy, precision, recall, F1 score, and AUC-ROC. Handles edge cases (e.g., zero_division in precision/recall, single-class datasets in AUC-ROC).

* **`compute_pos_weight(dataset, device)`** Calculates positive class weight for handling class imbalance in binary cross-entropy loss. Computes the ratio of negative to positive samples. **Parameters:** dataset (OilSlickDataset), device (torch device). **Returns:** torch.Tensor with positive weight value.

* **`train_epoch(model, loader, optimizer, criterion, device)`** Executes a complete training epoch: forward passes, loss computation, backpropagation, and optimizer updates. Collects predictions and labels for metric computation. **Parameters:** model, DataLoader, optimizer, loss criterion, device. **Returns:** tuple of (epoch_loss, metrics_dict).

* **`validate_epoch(model, loader, criterion, device)`** Executes a complete validation epoch in evaluation mode (no gradients). Computes loss and classification metrics on the validation set. **Parameters:** model, DataLoader, loss criterion, device. **Returns:** tuple of (epoch_loss, metrics_dict).

## Dependencies

* **Internal:** None
* **External:** `torch`, `numpy`, `random`, `tqdm`, `sklearn.metrics`

