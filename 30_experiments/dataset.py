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

        exp_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(exp_dir, "filtered_metadata.csv"))
        df = df[df["valid_sample"] == True]

        self.labels = dict(zip(df["sample_id"], df["label"]))

        with open(
            os.path.join(data_root, "splits", split_type, f"{split}.txt"), "r"
        ) as f:
            all_ids = [line.strip() for line in f.readlines()]

        self.image_ids = [
            i
            for i in all_ids
            if i in self.labels
            and os.path.exists(os.path.join(self.image_dir, f"{i}_s1.tif"))
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

        # temporary visualization to check if the data looks correct
        # RUN THIS by using inference.py
        # todo BEFORE UNCOMMENTING: set num_workers=1 in dataloader.py
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # fig.suptitle(f"Image: {img_id}_s1.tif", fontsize=14)
        # axes[0].imshow(img_tensor[0].numpy(), cmap="gray")
        # axes[0].set_title(f"VV - Label: {label}")
        # axes[1].imshow(img_tensor[1].numpy(), cmap="gray")
        # print(img_tensor[0].numpy())
        # print(img_tensor[1].numpy())
        # axes[1].set_title(f"VH - Label: {label}")
        # plt.tight_layout()
        # plt.show()

        return img_tensor, torch.tensor(label, dtype=torch.long)
