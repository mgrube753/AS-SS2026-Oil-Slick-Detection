# [runner.py](../30_experiments/runner.py)

Argparse-based runner script to execute training with different models and split type combinations for grid search over hyperparameters.

## Overview

A command-line interface for launching the complete grid search training pipeline. Manages 4 training configurations (2 model types × 2 split types) and orchestrates parallel execution of 9 hyperpara[...] 

## Core Functions

* **`main(model_name, split_type)`** Parses command-line arguments and initializes grid search over learning rates and weight decays for the specified model and split type. Iterates through all hy[...]

## Command-Line Arguments

* **`--model-name`** Model architecture to train. Choices: `["baselinecnn", "terramind"]`. Default: "baselinecnn"

* **`--split-type`** Data split strategy. Choices: `["random", "geographic"]`. Default: "random"

## Hyperparameter Grid

The grid search automatically selects hyperparameters based on model type:

**BaselineCNN (CNN)**
- Learning Rates: `[1e-4, 5e-4, 1e-3]`
- Weight Decays: `[1e-4, 1e-3, 1e-2]`
- Combinations: 3 × 3 = 9 subruns

**TerraMindClassifier (GFM)**
- Learning Rates: `[5e-4, 1e-3, 3e-3]`
- Weight Decays: `[1e-5, 1e-4, 1e-3]`
- Combinations: 3 × 3 = 9 subruns

## Complete Grid Search Execution

To train all 4 model configurations:

```bash
# Configuration 1: BaselineCNN + Random Split
python runner.py --model-name baselinecnn --split-type random

# Configuration 2: BaselineCNN + Geographic Split
python runner.py --model-name baselinecnn --split-type geographic

# Configuration 3: TerraMind + Random Split
python runner.py --model-name terramind --split-type random

# Configuration 4: TerraMind + Geographic Split
python runner.py --model-name terramind --split-type geographic
```

**Total Training:** 4 configurations × 9 subruns = 36 trained models

## Dependencies

* **Internal:** `train`, `config`
* **External:** `argparse`

## Output

Each subrun generates:
- **Checkpoints**: Saved to `logs/<split_folder>/<model_folder>/models/lr<LR>_wd<WD>/`
- **MLflow Logs**: Tracked in `logs/mlflow/` with experiment names `{model_name}-{split_type}`
- **Console Output**: Progress tracking with epoch metrics and loss values

All best models are later retrieved by `eval.py` for test set evaluation based on lowest validation loss.
