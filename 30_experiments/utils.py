import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

"""
Utils for training/validation loops, metric calculation,
seed setting, and class imbalance calculation.
"""


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_metrics(probs, labels):
    probs = probs.flatten()
    preds = (probs > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = 0.5
    return metrics


def compute_pos_weight(dataset, device):
    num_pos = sum(dataset.labels[i] for i in dataset.image_ids)
    if num_pos == 0:
        return torch.tensor([1.0], device=device, dtype=torch.float)

    pos_weight_val = (len(dataset) - num_pos) / num_pos
    return torch.tensor([pos_weight_val], device=device, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_acc = 0.0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        loss_acc += loss.item()
        all_probs.append(torch.sigmoid(outputs).detach())
        all_labels.append(labels)

    epoch_loss = loss_acc / len(loader)
    metrics = calc_metrics(
        torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    )
    return epoch_loss, metrics


def validate_epoch(model, loader, criterion, device):
    model.eval()
    loss_acc = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            loss_acc += loss.item()
            all_probs.append(torch.sigmoid(outputs))
            all_labels.append(labels)

    epoch_loss = loss_acc / len(loader)
    metrics = calc_metrics(
        torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    )
    return epoch_loss, metrics
