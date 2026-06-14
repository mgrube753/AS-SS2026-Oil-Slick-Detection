import torch
from dataloader import get_train_val_loaders, get_test_loader
from model import BaselineCNN, TerraMindClassifier

"""
Temporal file for running some inference to
check the modules' functionality.
"""


def evaluate(model_name="baselinecnn", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = "../10_waterbench_data/data/OilSlick"

    loader, _, ds = get_train_val_loaders(
        data_root, batch_size=batch_size, split_type="random"
    )
    # _, loader, ds = get_train_val_loaders(
    #     data_root, batch_size=batch_size, split_type="random"
    # )
    # loader, ds = get_test_loader(data_root, batch_size=batch_size, split_type="random")

    # loader, _, ds = get_train_val_loaders(
    #     data_root, batch_size=batch_size, split_type="geographic"
    # )
    # _, loader, ds = get_train_val_loaders(
    #     data_root, batch_size=batch_size, split_type="geographic"
    # )
    # loader, ds = get_test_loader(
    #     data_root, batch_size=batch_size, split_type="geographic"
    # )

    if model_name == "baselinecnn":
        model = BaselineCNN(num_classes=2, in_channels=2)
    elif model_name == "terramind":
        model = TerraMindClassifier(num_classes=2)
    else:
        raise ValueError(f"Invalid model: {model_name}")

    model.to(device)
    model.eval()

    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0
    print(f"Testing {model_name} on {len(ds)} samples...")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if imgs.shape[1] > 2:
                imgs = imgs[:, :2, :, :]
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs).flatten() > 0.5).int()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"Acc: {acc:.4f} ({correct}/{total}) - Prec: {precision:.4f} - Rec: {recall:.4f} - F1: {f1:.4f}"
    )


if __name__ == "__main__":
    evaluate()

