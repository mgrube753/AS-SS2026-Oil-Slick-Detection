import torch

"""
Config class with grid search parameters for both models, as well as other constants.
missing: mlflow logging (train/val), so we can better track progress/fails
"""


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = "../10_waterbench_data/data/OilSlick"
    OUTPUT_ROOT = "."
    SEED = 42
    #########
    EARLY_STOPPING_PATIENCE = 4
    BATCH_SIZE = 16
    EPOCHS_CNN = 32
    EPOCHS_GFM = 16
    #########
    WARMUP_EPOCHS_CNN = 3
    WARMUP_EPOCHS_GFM = 2
    #########
    LEARNING_RATES_CNN = [1e-4, 5e-4, 1e-3]
    WEIGHT_DECAYS_CNN = [1e-4, 1e-3, 1e-2]
    LEARNING_RATES_GFM = [5e-4, 1e-3, 3e-3]  # > CNN
    WEIGHT_DECAYS_GFM = [1e-5, 1e-4, 1e-3]  # < CNN
    #########
