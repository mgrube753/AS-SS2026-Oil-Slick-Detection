import os
import torch
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from config import Config
from model import BaselineCNN, TerraMindClassifier
from dataloader import get_test_loader

"""
Evaluation script for best 4 oil slick detection models.
For each of the configs, 9 subruns were trained using
different hyperparameters. This module finds the best
checkpoint per configuration (split & model) from MLflow logs,
runs inference on the test set, and computes classification metrics.
"""


def find_best_checkpoint(split_type, model_type):
    mlflow_dir = "logs/mlflow"

    best_checkpoint = None
    best_val_loss = float("inf")
    best_run_id = None

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
                with open(os.path.join(params_dir, "model")) as f:
                    model = f.read().strip()
                with open(os.path.join(params_dir, "split")) as f:
                    split = f.read().strip()
                if split != split_type or model != target_model:
                    continue

                with open(
                    os.path.join(
                        metrics_dir,
                        "best_val_loss",
                    )
                ) as f:
                    last_line = f.readlines()[-1].strip().split()
                    val_loss = float(last_line[1])

                checkpoint_path = os.path.join(
                    run_path,
                    "artifacts",
                    "best_model.pth",
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = checkpoint_path
                    best_run_id = run_id

            except Exception:
                continue

    return best_checkpoint, best_val_loss, best_run_id


def run_test_inference(
    checkpoint_path,
    split_type,
    model_type,
):

    model = load_model(
        checkpoint_path,
        model_type,
    )

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


def compute_metrics(
    labels,
    preds,
    probs,
):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "cm": confusion_matrix(labels, preds),
    }

    return metrics


def evaluate_model_checkpoint(
    checkpoint_path,
    split_type,
    model_type,
):
    results = run_test_inference(
        checkpoint_path,
        split_type,
        model_type,
    )
    metrics = compute_metrics(
        results["labels"],
        results["preds"],
        results["probs"],
    )

    return {
        "metrics": metrics,
        "labels": results["labels"],
        "preds": results["preds"],
        "probs": results["probs"],
        "dataset": results["dataset"],
    }


def load_model(
    checkpoint_path,
    model_type,
):
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
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
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

    return (
        all_labels,
        all_preds,
        all_probs,
    )


def save_confusion_matrix(
    cm,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    disp.plot(text_kw={"fontsize": 20})
    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png",
        )
    )
    plt.close()


def save_probability_histogram(
    labels,
    probs,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(
        [probs[i] for i in range(len(labels)) if labels[i] == 0],
        bins=30,
        alpha=0.6,
        label="Negative (class 0)",
    )
    plt.hist(
        [probs[i] for i in range(len(labels)) if labels[i] == 1],
        bins=30,
        alpha=0.6,
        label="Positive (class 1)",
    )
    plt.axvline(0.5, color="red", linestyle="--", label="Threshold=0.5")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Prediction probability distribution by true class")

    plt.savefig(
        os.path.join(
            output_dir,
            "probability_histogram.png",
        )
    )
    plt.close()


def save_classification_report(
    labels,
    preds,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    report = classification_report(
        labels,
        preds,
        target_names=["No Slick (0)", "Oil Slick (1)"],
        digits=4,
    )
    print(report)

    report_path = os.path.join(
        output_dir,
        "classification_report.txt",
    )
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Classification report saved to: {os.path.abspath(report_path)}")


def save_predictions_csv(
    labels,
    preds,
    probs,
    split_type,
    model_type,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        output_dir,
        "test_predictions.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "true_label",
                "pred_label",
                "probability",
                "correct",
            ]
        )

        for i, (y_true, y_pred, prob) in enumerate(zip(labels, preds, probs)):
            writer.writerow(
                [
                    i,
                    y_true,
                    y_pred,
                    f"{prob:.6f}",
                    int(y_true == y_pred),
                ]
            )

    print(f"Predictions CSV saved to: {os.path.abspath(csv_path)}")


def save_combined_confusion_matrices(cm_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vmin = min(cm.min() for cm, _, _ in cm_list)
    vmax = max(cm.max() for cm, _, _ in cm_list)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 13),
        constrained_layout=True,
        gridspec_kw={"hspace": 0.05},
    )
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        cm, title, class_names = cm_list[idx]
        im = ax.imshow(cm, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")

        norm_val = (cm - vmin) / (vmax - vmin + 1e-9)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                rgba = plt.cm.viridis(norm_val[i, j])
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_c = "white" if luminance < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color=text_c,
                )

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=18)
        ax.set_yticklabels(class_names, fontsize=18)
        ax.set_xlabel("Predicted label", fontsize=18)
        ax.set_ylabel("True label", fontsize=18)
        ax.set_title(title, fontsize=20, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.03)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined confusion matrices saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    configs = [
        ("random", "cnn"),
        ("random", "gfm"),
        ("geographic", "cnn"),
        ("geographic", "gfm"),
    ]

    title_map = {
        ("random", "cnn"): "(a) Baseline CNN: Random Split",
        ("random", "gfm"): "(b) Terramind: Random Split",
        ("geographic", "cnn"): "(c) Baseline CNN: Geographic Split",
        ("geographic", "gfm"): "(d) Terramind: Geographic Split",
    }
    class_names = ["0", "1"]
    combined_cm_list = []

    for split_type, model_type in configs:
        checkpoint_path, best_val_loss, best_run_id = find_best_checkpoint(
            split_type,
            model_type,
        )

        print(f"\nEvaluating {split_type} {model_type}")
        print(f"Best run ID : {best_run_id}")
        print(f"Checkpoint  : {checkpoint_path}")
        print(f"Best val loss: {best_val_loss:.6f}")

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

        fp = metrics["cm"][0, 1]
        fn = metrics["cm"][1, 0]
        tp = metrics["cm"][1, 1]
        tn = metrics["cm"][0, 0]
        print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        split_folder = "random_split" if split_type == "random" else "geographic_split"
        output_dir = os.path.join(
            "..",
            "50_evaluation",
            split_folder,
            model_type,
            "results",
        )
        save_confusion_matrix(
            metrics["cm"],
            output_dir,
        )
        combined_cm_list.append(
            (
                metrics["cm"],
                title_map[(split_type, model_type)],
                class_names,
            )
        )
        save_probability_histogram(
            results["labels"],
            results["probs"],
            output_dir,
        )
        save_classification_report(
            results["labels"],
            results["preds"],
            output_dir,
        )
        save_predictions_csv(
            results["labels"],
            results["preds"],
            results["probs"],
            split_type,
            model_type,
            output_dir,
        )

    combined_output_path = os.path.join(
        "..",
        "50_evaluation",
        "combined_confusion_matrices.png",
    )
    save_combined_confusion_matrices(combined_cm_list, combined_output_path)
