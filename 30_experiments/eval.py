"""
Evaluation on Test Sets
"""
import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import Config
from model import BaselineCNN, TerraMindClassifier
from dataloader import get_test_loader

def find_best_checkpoint(split_type, model_type):

    mlflow_dir = "mlflow_extracted/mlflow"

    best_checkpoint = None
    best_val_loss = float("inf")

    model_map = {
        "cnn": "baselinecnn",
        "gfm": "terramind",
    }

    target_model = model_map[model_type]

    for experiment_id in os.listdir(mlflow_dir):

        exp_path = os.path.join(
            mlflow_dir,
            experiment_id,
        )

        if not os.path.isdir(exp_path):
            continue

        for run_id in os.listdir(exp_path):

            run_path = os.path.join(
                exp_path,
                run_id,
            )

            params_dir = os.path.join(
                run_path,
                "params",
            )

            metrics_dir = os.path.join(
                run_path,
                "metrics",
            )

            try:

                with open(
                    os.path.join(params_dir, "model")
                ) as f:
                    model = f.read().strip()

                with open(
                    os.path.join(params_dir, "split")
                ) as f:
                    split = f.read().strip()

                if (
                    split != split_type
                    or model != target_model
                ):
                    continue

                with open(
                    os.path.join(
                        metrics_dir,
                        "best_val_loss",
                    )
                ) as f:
                    last_line = (
                        f.readlines()[-1]
                        .strip()
                        .split()
                    )

                    val_loss = float(last_line[1])

                checkpoint_path = os.path.join(
                    run_path,
                    "artifacts",
                    "best_model.pth",
                )

                if val_loss < best_val_loss:

                    best_val_loss = val_loss
                    best_checkpoint = checkpoint_path

            except Exception:
                continue

    return best_checkpoint

def run_test_inference(checkpoint_path,split_type,model_type,):

    model = load_model(checkpoint_path,model_type,)

    test_loader, test_ds = get_test_loader(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        split_type=split_type,
        seed=Config.SEED,
    )

    labels, preds, probs = run_model_inference(
        model,
        test_loader,
    )

    return {
        "labels": labels,
        "preds": preds,
        "probs": probs,
        "dataset": test_ds,
    }

def compute_metrics(labels,preds,probs,):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "cm": confusion_matrix(labels, preds),
    }

    return metrics

def evaluate_model_checkpoint(checkpoint_path,split_type,model_type,):
    results = run_test_inference(checkpoint_path,split_type,model_type,)
    metrics = compute_metrics(results["labels"],results["preds"],results["probs"], )

    return {
    "metrics": metrics,
    "labels": results["labels"],
    "preds": results["preds"],
    "probs": results["probs"],
    "dataset": results["dataset"],
}


def load_model(checkpoint_path,model_type,):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=Config.DEVICE,
    )

    if model_type == "cnn":
        model = BaselineCNN(
            num_classes=2,
            in_channels=2,
        )

    elif model_type == "gfm":
        model = TerraMindClassifier(
            num_classes=2,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(Config.DEVICE)
    model.eval()

    return model

def run_model_inference(model, test_loader):

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).int()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    

    return (all_labels, all_preds, all_probs,)


        
def save_confusion_matrix(cm,output_dir,):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png",
        )
    )
    plt.close()
    
if __name__ == "__main__":
    
    configs = [
    ("geographic", "cnn"),
    ("random", "cnn"),
    ("geographic", "gfm"),
    ("random", "gfm"),
]

    for split_type, model_type in configs:
        checkpoint_path = find_best_checkpoint(
        split_type,
        model_type,
    )

        print(f"\nEvaluating {split_type} {model_type}")
        print(f"Checkpoint: {checkpoint_path}")

        results = evaluate_model_checkpoint(
            checkpoint_path=checkpoint_path,
            split_type=split_type,
            model_type=model_type,
        )

        metrics = results["metrics"]

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"AUC ROC  : {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['cm']}")
        
