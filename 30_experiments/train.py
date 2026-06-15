import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
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
from model import BaselineCNN, TerraMindClassifier
from config import Config

"""
Model- and split-specific training script including
training and validation loops, metric calculations,
and a first learning rate scheduler.

This pipeline has to be revised properly, since currently
- no stable training for both models is achieved by testing out
  LR schedulers might be the improvement!
- no early stopping is implemented
- no mlflow logging is implemented
- no checkpoint saving is implemented
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


def run_training(model_name, split_type):
    set_seed(Config.SEED)
    print(f"Model: {model_name} | Split: {split_type}")

    train_loader, val_loader, _ = get_train_val_loaders(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        split_type=split_type,
        seed=Config.SEED,
    )

    if model_name == "baselinecnn":
        model = BaselineCNN(
            num_classes=Config.NUM_CLASSES, in_channels=Config.IN_CHANNELS
        )
    elif model_name == "terramind":
        model = TerraMindClassifier(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(Config.DEVICE)
    
    if model_name == "baselinecnn":
        model_folder = "cnn"
    else:
        model_folder = "gfm"
    
    if split_type == "random":
        split_folder = "random_split"
    else:
        split_folder = "geographic_split"
    checkpoint_dir = os.path.join(
        Config.OUTPUT_ROOT,
        split_folder,
        model_folder,
        "models",
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(
        checkpoint_dir,
        "best_model.pth"
    )

    final_model_path = os.path.join(
        checkpoint_dir,
        "final_model.pth"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=(
            Config.LEARNING_RATE_CNN
            if model_name == "baselinecnn"
            else Config.LEARNING_RATE_GFM
        ),
        weight_decay=(
            Config.WEIGHT_DECAY_CNN
            if model_name == "baselinecnn"
            else Config.WEIGHT_DECAY_GFM
        ),
    )

    # todo: need to set up proper scheduler and LR/WD values
    # useful: warmup phase for both models to do essential changes to heads' weights, then switch to cosine annealing to step-wise reduce LR
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # pos_weight = torch.tensor(Config.CLASS_WEIGHTS[1], dtype=torch.float).to(
    # Config.DEVICE
    # )
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=pos_weight
    # )
    criterion = nn.BCEWithLogitsLoss()
    # todo: think about class weighting based on positive/negative ratio in training set instead of fixed values
    criterion.to(Config.DEVICE)
    
    best_val_loss = float("inf")
    patience_counter = 0

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_model_path,
            )

            print(f"Best model saved (Val Loss = {val_loss:.4f})")
        
        else:
            patience_counter += 1

            print(f"No improvement for {patience_counter}/{Config.EARLY_STOPPING_PATIENCE} epochs" )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
        scheduler.step()
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        final_model_path,
    )

    print(f"Final model saved to {final_model_path}")
