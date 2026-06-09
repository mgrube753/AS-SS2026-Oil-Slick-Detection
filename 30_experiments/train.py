import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataloader import get_train_val_loaders
from model import ResNet50Classifier, TerraMindClassifier
from config import Config

"""
Model- and split-specific training script including
training and validation loops, metric calculations,
and a first learning rate scheduler.

This pipeline has to be revised properly, since currently
- no stable training for both models is achieved by testing out
  LR schedulers might be the improvement!
- no seeding is included
- no early stopping is implemented
- no mlflow logging is implemented
- no checkpoint saving is implemented
"""


def calc_metrics(probs, labels):
    probs = probs.flatten()
    preds = (probs > 0.5).astype(int)
    auc_probs = probs

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(labels, auc_probs)
    except ValueError:
        metrics["auc_roc"] = 0.5
    return metrics


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
        all_probs.append(outputs.detach())
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
            all_probs.append(outputs)
            all_labels.append(labels)

    epoch_loss = loss_acc / len(loader)
    metrics = calc_metrics(
        torch.cat(all_probs).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    )
    return epoch_loss, metrics


def run_training(model_name, split_type):
    print(f"Model: {model_name} | Split: {split_type}")

    train_loader, val_loader, _ = get_train_val_loaders(
        data_root=Config.DATA_ROOT, batch_size=Config.BATCH_SIZE, split_type=split_type
    )

    if model_name == "resnet50":
        model = ResNet50Classifier(
            num_classes=Config.NUM_CLASSES, in_channels=Config.IN_CHANNELS
        )
    elif model_name == "terramind":
        model = TerraMindClassifier(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(Config.DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=(
            Config.LEARNING_RATE_CNN
            if model_name == "resnet50"
            else Config.LEARNING_RATE_GFM
        ),
        weight_decay=(
            Config.WEIGHT_DECAY_CNN
            if model_name == "resnet50"
            else Config.WEIGHT_DECAY_GFM
        ),
    )

    # todo: need to set up proper scheduler and LR/WD values
    # useful: warmup phase for both models to do essential changes to heads' weights, then switch to cosine annealing to step-wise reduce LR
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float).to(Config.DEVICE)
    criterion = (
        nn.BCELoss()
    )  # todo: think about class weighting based on positive/negative ratio in training set
    criterion.to(Config.DEVICE)

    for epoch in range(Config.EPOCHS):
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, v_m = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        print(
            f"Validation | Acc: {v_m['accuracy']:.4f}, F1: {v_m['f1']:.4f}, AUC: {v_m['auc_roc']:.4f}"
        )

        scheduler.step()
