import torch

"""
Examplory config class with adjustable hyperparameters
missing: mlflow logging (train/val), so we can better track progress/fails
useful addition: learning rate schedulers
"""


class Config:
    DATA_ROOT = "../10_waterbench_data/data/OilSlick"
    OUTPUT_ROOT = "."

    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 16  # todo: check if this is enough for convergence
    EARLY_STOPPING_PATIENCE = 4  # todo: implement soon
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # todo: talk about LR and WD values --> Schedulers? let's discuss
    LEARNING_RATE_CNN = 5e-4
    WEIGHT_DECAY_CNN = 5e-2
    LEARNING_RATE_GFM = 3e-4
    WEIGHT_DECAY_GFM = 1e-2

    NUM_CLASSES = 2
    IN_CHANNELS = 2

    CLASS_WEIGHTS = [1.0, 2.0]
