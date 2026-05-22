## Directory structure for the future

```tree
30_experiments/
├── dataloader.py              [already there]
├── dataset.py                 [already there]
├── model.py                   [already there]
├── inference.py               [already there]
│
├── config.py                  [Configuration of hyperparams (e.g. optimizer, learning rate, early stopping), paths, MLflow logging for all runs]
├── train.py                   [Overall training script with regularization and MLflow (+ train/val metrics logging)]
├── eval.py                    [Evaluation metrics calculation on both splits and both models on test set (MLflow logging)]
├── failure_analysis.py        [Confusion matrices, lookalike analysis (also MLflow possible?)]
├── (utils.py)                 [Helper functions if needed]
│
├── run_random_cnn.py          [Runner for Random Split + ResNet50] # 4 runner scripts needed?
├── run_random_gfm.py          [Runner for Random Split + TerraMind]
├── run_geographic_cnn.py      [Runner for Geographic Split + ResNet50]
├── run_geographic_gfm.py      [Runner for Geographic Split + TerraMind]
│
├── random_split/
│   ├── .gitkeep
│   ├── cnn/                   [ResNet50 Outputs]
│   │   ├── models/            [Model checkpoints]
│   │   ├── logs/              [MLflow Outputs]
│   │   └── results/           [Metrics, eval outputs]
│   └── gfm/                   [TerraMind Outputs]
│       ├── models/
│       ├── logs/
│       └── results/
│
└── geographic_split/
    ├── .gitkeep
    ├── cnn/                   [ResNet50 Outputs]
    │   ├── models/
    │   ├── logs/
    │   └── results/
    └── gfm/                   [TerraMind Outputs]
        ├── models/
        ├── logs/
        └── results/
```
