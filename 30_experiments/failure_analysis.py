import os
import torch
import matplotlib.pyplot as plt

"""
Failure analysis script to identify and visualize
false positives and false negatives from the test
set predictions of the individual trained model.
"""

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
from dataloader import get_test_loader
from model import BaselineCNN

def load_model(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=Config.DEVICE,
    )

    model = BaselineCNN(
        num_classes=2,
        in_channels=2,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(Config.DEVICE)
    model.eval()

    return model
    
def evaluate_model(model, test_loader):

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

    metrics = {
        "accuracy": accuracy_score(
            all_labels,
            all_preds,
        ),
        "precision": precision_score(
            all_labels,
            all_preds,
        ),
        "recall": recall_score(
            all_labels,
            all_preds,
        ),
        "f1": f1_score(
            all_labels,
            all_preds,
        ),
        "auc": roc_auc_score(
            all_labels,
            all_probs,
        ),
        "cm": confusion_matrix(
            all_labels,
            all_preds,
        ),
    }

    return metrics, all_labels, all_preds

def get_failure_indices(all_labels,all_preds):
    fp_indices = []
    fn_indices = []

    for i, (y_true, y_pred) in enumerate(
        zip(all_labels, all_preds) ):

        if y_true == 0 and y_pred == 1:
            fp_indices.append(i)

        elif y_true == 1 and y_pred == 0:
            fn_indices.append(i)

    return fp_indices, fn_indices

def save_examples(indices, dataset, save_dir, prefix):
    os.makedirs( save_dir, exist_ok=True,)
    
    print(f"Saving {prefix} examples to:")
    print(os.path.abspath(save_dir))

    for idx in indices:

        image, label = dataset[idx]

        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow( image[0], cmap="gray",)
        plt.title( f"{prefix} | VV | idx={idx}" )
        
        plt.subplot(1, 2, 2)
        plt.imshow(image[1], cmap="gray",)
        plt.title("VH")
        
        plt.tight_layout()
        filename = os.path.join(save_dir, f"{prefix.lower()}_{idx}.png")
        plt.savefig(filename)
        plt.close()
        
    print(f"Saved {len(indices)} {prefix} examples to {save_dir}")
        
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
        
def run_failure_analysis(split_type, lr=0.0001, wd=0.0001):
    print(f"\nRunning Failure Analysis: {split_type} | LR: {lr} | WD: {wd}")

    split_folder = "random_split" if split_type == "random" else "geographic_split"

    checkpoint_path = os.path.join(
        "logs",
        split_folder,
        "cnn",
        "models",
        f"lr{lr}_wd{wd}",
        "best_model.pth",
    )

    model = load_model(checkpoint_path)

    test_loader, test_ds = get_test_loader(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        split_type=split_type,
        seed=Config.SEED,
    )

    metrics, labels, preds = evaluate_model( model, test_loader, )
    fp_indices, fn_indices = get_failure_indices(labels,preds,)

    print(f"Accuracy : {metrics['accuracy']:.4f}"  )
    print(f"Precision: {metrics['precision']:.4f}" )
    print(f"Recall   : {metrics['recall']:.4f}" )
    print(f"F1 Score : {metrics['f1']:.4f}" )
    print( f"AUC ROC  : {metrics['auc']:.4f}" )
    print(f"False Positives: {len(fp_indices)}" )
    print( f"False Negatives: {len(fn_indices)}" )

    output_dir = os.path.join(
        "..",
        "50_evaluation",
        split_folder,
        "cnn",
        f"lr{lr}_wd{wd}",
        "results",
        "failure_analysis",
    )
    print("Output directory:")
    print(os.path.abspath(output_dir)) 

    save_confusion_matrix(
        metrics["cm"],
        output_dir,
    )

    save_examples(
        fp_indices,
        test_ds,
        os.path.join(
            output_dir,
            "false_positives",
        ),
        "FP",
    )

    save_examples(
        fn_indices,
        test_ds,
        os.path.join(
            output_dir,
            "false_negatives",
        ),
        "FN",
    )

if __name__ == "__main__":
    # 2 sample runs which have to be done before executing this script
    run_failure_analysis(split_type="random", lr=0.0001, wd=0.0001)
    # run_failure_analysis(split_type="geographic", lr=0.0001, wd=0.0001)
