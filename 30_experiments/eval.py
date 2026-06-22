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

    final_models = [
        {
            "name": "geographic_cnn",
            "checkpoint": "logs/geographic_split/cnn/models/lr0.001_wd0.0001/best_model.pth",
            "split_type": "geographic",
            "model_type": "cnn",
        },
        {
            "name": "random_cnn",
            "checkpoint": "logs/random_split/cnn/models/lr0.001_wd0.0001/best_model.pth",
            "split_type": "random",
            "model_type": "cnn",
        },
        {
            "name": "geographic_gfm",
            "checkpoint": "logs/geographic_split/gfm/models/lr0.003_wd0.0001/best_model.pth",
            "split_type": "geographic",
            "model_type": "gfm",
        },
        {
            "name": "random_gfm",
            "checkpoint": "logs/random_split/gfm/models/lr0.003_wd0.0001/best_model.pth",
            "split_type": "random",
            "model_type": "gfm",
        },
    ]

    for model_info in final_models:

        print(f"Evaluating: {model_info['name']}")
        
        results = evaluate_model_checkpoint(
            checkpoint_path=model_info["checkpoint"],
            split_type=model_info["split_type"],
            model_type=model_info["model_type"],
        )

        metrics = results["metrics"]
        output_dir = os.path.join(
            "..",
            "50_evaluation",
            model_info["split_type"] + "_split",
            model_info["model_type"],
            "results",
        )

        save_confusion_matrix(
            metrics["cm"],
            output_dir,
        )
        

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print(f"AUC ROC  : {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['cm']}")
        
