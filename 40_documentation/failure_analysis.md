# [failure_analysis.py](../30_experiments/failure_analysis.py)

Failure analysis script that identifies and visualizes false positives and false negatives from the test set predictions of the trained models.

## Overview

A comprehensive failure analysis module for analyzing the 4 best-performing models (BaselineCNN and TerraMindClassifier on random and geographic splits). It orchestrates test set inference, identifies misclassifications, and saves visualization examples of false positives and false negatives for qualitative analysis.

## Core Functions

* **`get_failure_indices(all_labels, all_preds)`** Identifies indices of false positives (predicted 1, true 0) and false negatives (predicted 0, true 1) by comparing ground truth labels with model predictions. **Returns:** tuple of (fp_indices, fn_indices) lists.

* **`save_examples(indices, dataset, save_dir, prefix)`** Generates and saves visualization plots for a set of samples. Each plot displays the VV and VH Sentinel-1 SAR bands side-by-side as grayscale images. **Creates:** individual PNG files per sample with format `{prefix.lower()}_{idx}.png`.

* **`run_failure_analysis(split_type, model_type)`** Orchestrates complete failure analysis pipeline: finds best checkpoint → runs test inference → identifies failures → generates visualizations. Supports both "random" and "geographic" split types and both "cnn" and "gfm" model types. **Outputs:** PNG visualizations organized in `false_positives/` and `false_negatives/` subdirectories.

## Dependencies

* **Internal:** `eval` (imports `run_test_inference`, `find_best_checkpoint`)
* **External:** `os`, `matplotlib.pyplot`

## Execution

The script runs failure analysis for all 4 model configurations when executed as main:
- Geographic CNN
- Random CNN
- Geographic GFM
- Random GFM

Output structure:
```
50_evaluation/
├── random_split/
│   ├── cnn/results/failure_analysis/
│   │   ├── false_positives/
│   │   └── false_negatives/
│   └── gfm/results/failure_analysis/
│       ├── false_positives/
│       └── false_negatives/
└── geographic_split/
    ├── cnn/results/failure_analysis/
    │   ├── false_positives/
    │   └── false_negatives/
    └── gfm/results/failure_analysis/
        ├── false_positives/
        └── false_negatives/
```
