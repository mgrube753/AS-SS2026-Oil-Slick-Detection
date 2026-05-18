import os
import pandas as pd
import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Defines the OilSlickDataset class for loading and 
preprocessing the SAR images and their labels.
"""


class OilSlickDataset(Dataset):
    def __init__(self, data_root, split_type="random", split="train", transform=None):
        self.image_dir = os.path.join(data_root, "images_s1")

        df = pd.read_csv(os.path.join(data_root, "metadata.csv"))
        self.labels = dict(zip(df["sample_id"], df["label"]))

        with open(
            os.path.join(data_root, "splits", split_type, f"{split}.txt"), "r"
        ) as f:
            all_ids = [line.strip() for line in f.readlines()]

        self.image_ids = [
            i
            for i in all_ids
            if os.path.exists(os.path.join(self.image_dir, f"{i}_s1.tif"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_data = tifffile.imread(os.path.join(self.image_dir, f"{img_id}_s1.tif"))
        img_tensor = torch.from_numpy(img_data.astype(np.float32))

        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3 and img_tensor.shape[-1] <= 3:
            img_tensor = img_tensor.permute(2, 0, 1)

        label = int(self.labels.get(img_id, 0))

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label, dtype=torch.long)
