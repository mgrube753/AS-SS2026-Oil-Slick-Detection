import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import OilSlickDataset

"""
Creates dataloaders for training, validation and testing.
Also includes a custom transform for z-scoring the SAR data.
"""

TRAIN_STATS_RANDOM = {
    "mean_vv": -17.87,
    "std_vv": 26.34,
    "mean_vh": -25.61,
    "std_vh": 19.91,
}

TRAIN_STATS_GEO = {
    "mean_vv": -19.42,
    "std_vv": 22.68,
    "mean_vh": -26.84,
    "std_vh": 17.84,
}


class SARzScore:
    def __init__(self, split_type="random"):
        self.split_type = split_type
        self.train_stats = (
            TRAIN_STATS_RANDOM if split_type == "random" else TRAIN_STATS_GEO
        )

    def __call__(self, x):
        x_np = x.numpy()
        channel_keys = [("mean_vv", "std_vv"), ("mean_vh", "std_vh")]

        for c in range(x_np.shape[0]):
            mean_key, std_key = channel_keys[c]
            mean = self.train_stats[mean_key]
            std = self.train_stats[std_key]

            nodata = (x_np[c] <= -50) | np.isnan(x_np[c])
            valid = ~nodata

            if valid.any():
                valid_pixels = x_np[c][valid]
                p_low = np.percentile(valid_pixels, 2)
                p_high = np.percentile(valid_pixels, 98)
                x_np[c][valid] = np.clip(x_np[c][valid], p_low, p_high)
                x_np[c][valid] = (x_np[c][valid] - mean) / (std + 1e-6)

            x_np[c][nodata] = 0.0

        return torch.from_numpy(x_np)


def get_train_transform(split_type="random"):
    return transforms.Compose(
        [
            SARzScore(split_type=split_type),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.Resize((224, 224), antialias=True),
        ]
    )


def get_val_test_transform(split_type="random"):
    return transforms.Compose(
        [
            SARzScore(split_type=split_type),
            transforms.Resize((224, 224), antialias=True),
        ]
    )


def get_train_val_loaders(data_root, batch_size=16, split_type="random"):
    train_transform = get_train_transform(split_type=split_type)
    val_transform = get_val_test_transform(split_type=split_type)

    train_ds = OilSlickDataset(data_root, split="train", transform=train_transform)
    val_ds = OilSlickDataset(data_root, split="val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ds


def get_test_loader(data_root, batch_size=16, split_type="random"):
    test_transform = get_val_test_transform(split_type=split_type)

    test_ds = OilSlickDataset(data_root, split="test", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_loader, test_ds
