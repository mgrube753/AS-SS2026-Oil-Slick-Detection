
# Documentation

In this directory, you will find the technical documentation for the Python modules used in the oil slick detection pipeline. Each Markdown file corresponds to a specific module, providing information on its functions and interactions within the pipeline. The modules are stored in the [`../30_experiments/`](../30_experiments/) directory and organized as follows:

## Structure Overview

### Core Pipeline

- **[`config.md`](config.md)** - Configuration class defining hyperparameters, data paths, and grid search values for both CNN and GFM models.
- **[`dataset.md`](dataset.md)** - `OilSlickDataset` class for proper loading of Sentinel-1 SAR images (VV/VH bands) with their according binary labels.
- **[`dataloader.md`](dataloader.md)** - Creates Torch DataLoaders for training, validation, and testing, including training set-based z-score normalization and augmentation transforms (random flips and rotations).
- **[`model.md`](model.md)** - Two model architectures are presented: `BaselineCNN` (lightweight CNN trained from scratch) and `TerraMindClassifier` (pre-trained GFM backbone with a custom classification head on top).

### Training and Execution

- **[`runner.md`](runner.md)** - Argparse-based script for running grid search training for both model and both split combinations.
- **[`train.md`](train.md)** - Training loop using AdamW as optimizer of choice, cosine annealing LR scheduler with linear warmup to the desired learning rate, early stopping, model checkpointing (best and final models), and MLflow experiment logging.
- **[`utils.md`](utils.md)** - Seed setting, training/validation epoch loops, metric computation (accuracy, precision, recall, F1, AUC-ROC), and class imbalance handling (lower positive weight calculation, since the positive class is overrepresented).

### Evaluation and Analysis

- **[`eval.md`](eval.md)** - Evaluation script that finds the best checkpoint per configuration from MLflow logs (1 CNN on random/geographic splits, 1 GFM on random/geographic splits, 4 in total), runs test set-based inference, and computes classification metrics (accuracy, precision, recall, F1, AUC, confusion matrix).
- **[`failure_analysis.md`](failure_analysis.md)** - Failure analysis pipeline that analyses these 4 models and identifies false positives and false negatives from test set predictions and saves the corresponding SAR image patches for further qualitative inspection.
