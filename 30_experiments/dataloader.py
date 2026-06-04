import torch
import numpy as np
from scipy.stats import zscore
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import OilSlickDataset

"""
Creates dataloaders for training, validation and testing.
Also includes a custom transform for z-scoring the SAR data.
"""


class SARzScore:
    def __call__(self, x):
        x_np = x.numpy()
        for c in range(x_np.shape[0]):
            if np.std(x_np[c]) > 1e-7:
                x_np[c] = zscore(x_np[c], axis=None)
            else:
                x_np[c] = x_np[c] - np.mean(x_np[c])
        return torch.from_numpy(x_np)


# Training transform (with augmentation)
def get_train_transform():
    return transforms.Compose([
        SARzScore(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize((224, 224), antialias=True),
    ])


# Validation/Test transform (no augmentation)
def get_val_transform():
    return transforms.Compose([
        SARzScore(),
        transforms.Resize((224, 224), antialias=True),
    ])


def get_train_val_loaders(data_root, batch_size=16):
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_ds = OilSlickDataset(data_root,split="train", transform=train_transform)
    val_ds = OilSlickDataset(data_root, split="val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds


def get_test_loader(data_root, batch_size=16):
    test_transform = get_val_transform()

    test_ds = OilSlickDataset(data_root, split="test", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_loader, test_ds
