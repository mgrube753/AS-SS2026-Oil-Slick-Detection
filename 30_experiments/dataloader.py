import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import OilSlickDataset

"""
Creates dataloaders for training, validation and testing.
Also includes a custom transform for z-scoring the SAR data.
"""


def rotate0(x):
    return torch.rot90(x, 0, [-1, -2])


def rotate90(x):
    return torch.rot90(x, 1, [-1, -2])


def rotate180(x):
    return torch.rot90(x, 2, [-1, -2])


def rotate270(x):
    return torch.rot90(x, 3, [-1, -2])


class SARzScore:
    def __init__(self, train_stats):
        self.train_stats = train_stats

    def __call__(self, x):
        x_np = x.numpy()
        for c in range(x_np.shape[0]):
            c_str = str(c)
            mean = self.train_stats["means"][c_str]
            std = self.train_stats["stds"][c_str]

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


def get_train_transform(train_stats):
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                antialias=True,
            ),
            SARzScore(train_stats=train_stats),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomChoice(
                [
                    transforms.Lambda(rotate0),
                    transforms.Lambda(rotate90),
                    transforms.Lambda(rotate180),
                    transforms.Lambda(rotate270),
                ]
            ),
        ]
    )


def get_val_test_transform(train_stats):
    return transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            SARzScore(train_stats=train_stats),
        ]
    )


def load_split_stats(split_type):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(exp_dir, "..", "20_data_analysis", "split_stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats[split_type]


def get_train_val_loaders(data_root, batch_size=16, split_type="random", seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    train_stats = load_split_stats(split_type)

    train_transform = get_train_transform(train_stats)
    val_transform = get_val_test_transform(train_stats)

    train_ds = OilSlickDataset(
        data_root, split_type=split_type, split="train", transform=train_transform
    )
    val_ds = OilSlickDataset(
        data_root, split_type=split_type, split="val", transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g,
    )

    return train_loader, val_loader, train_ds


def get_test_loader(data_root, batch_size=16, split_type="random", seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    train_stats = load_split_stats(split_type)
    test_transform = get_val_test_transform(train_stats)

    test_ds = OilSlickDataset(
        data_root, split_type=split_type, split="test", transform=test_transform
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
        generator=g,
    )

    return test_loader, test_ds
