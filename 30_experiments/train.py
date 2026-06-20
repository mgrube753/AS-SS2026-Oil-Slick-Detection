import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow

from dataloader import get_train_val_loaders
from model import BaselineCNN, TerraMindClassifier
from config import Config
import utils

"""
Model- and split-specific training script including
a loop over epochs for training and validation,
early stopping, model checkpointing, and MLflow logging.
"""

DARK_GREEN = "\033[32;1m"
RESET = "\033[0m"


def run_training(
    model_name, split_type, lr=None, wd=None, total_runs=None, counter=None
):
    utils.set_seed(Config.SEED)
    epochs = Config.EPOCHS_CNN if model_name == "baselinecnn" else Config.EPOCHS_GFM
    warmup = (
        Config.WARMUP_EPOCHS_CNN
        if model_name == "baselinecnn"
        else Config.WARMUP_EPOCHS_GFM
    )

    print(
        f"{DARK_GREEN}Config {counter}/{total_runs} for Model: {model_name} | Split: {split_type} | LR: {lr} | WD: {wd}{RESET}"
    )

    train_loader, val_loader, train_ds = get_train_val_loaders(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        split_type=split_type,
        seed=Config.SEED,
    )

    if model_name == "baselinecnn":
        model = BaselineCNN(num_classes=2, in_channels=2)
    elif model_name == "terramind":
        model = TerraMindClassifier(num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(Config.DEVICE)

    model_folder = "cnn" if model_name == "baselinecnn" else "gfm"
    split_folder = "random_split" if split_type == "random" else "geographic_split"

    exp_base_dir = os.path.join(Config.OUTPUT_ROOT, "logs", split_folder, model_folder)
    mlflow_path = os.path.abspath(os.path.join(Config.OUTPUT_ROOT, "logs", "mlflow"))

    os.makedirs(mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_path}")
    mlflow.set_experiment(f"{model_name}-{split_type}")

    checkpoint_dir = os.path.join(exp_base_dir, "models", f"lr{lr}_wd{wd}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    s1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    s2 = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup])

    pos_weight = utils.compute_pos_weight(train_ds, Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Computed pos_weight: {pos_weight.item():.4f}")

    with mlflow.start_run(run_name=f"lr{lr}_wd{wd}"):
        mlflow.log_params(
            {
                "model": model_name,
                "split": split_type,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": Config.BATCH_SIZE,
                "warmup_epochs": warmup,
                "total_epochs": epochs,
                "pos_weight": pos_weight.item(),
                "criterion": criterion.__class__.__name__,
                "device": str(Config.DEVICE),
            }
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, t_m = utils.train_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE
            )
            val_loss, v_m = utils.validate_epoch(
                model, val_loader, criterion, Config.DEVICE
            )

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            for k, v in t_m.items():
                mlflow.log_metric(f"train_{k}", v, step=epoch)
            for k, v in v_m.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
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
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                print(f"Best model saved (Val Loss = {val_loss:.4f})")

            elif epoch >= warmup:
                patience_counter += 1
                print(
                    f"No improvement for {patience_counter}/{Config.EARLY_STOPPING_PATIENCE} epochs"
                )

                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    mlflow.set_tag("stopping_reason", "early_stopping")
                    print("Early stopping occurred.")
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

        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(final_model_path)
        print(f"Final model and MLflow artifacts saved to {final_model_path}")
