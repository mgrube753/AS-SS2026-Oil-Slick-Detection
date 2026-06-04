"""
We need to find out what is needed
- stick to structure.md for further file organization
- use dataloader, dataset and model files
- we need calls like this:

train_loader, val_loader, train_ds = get_train_val_loaders(
    data_root="path",
    split_type="random", # or geographic
    batch_size=16
)

- it should be an initial file, which gets updated step by step
- already done: data augmentation and Dropout Layers in both model classification heads
- initially, we need Adam or AdamW for L2 regularization
- add first metrics (accuracy, precision, recall, f1, auc-roc) during training and validation
- later on, add early stopping
- logging via mlflow comes later for train/val
- make it somewhen possible to properly run all 4 different trainings (random/geographic split and resnet/terramind model)
- so, in the end we should have another runner file (or 4, so for each split type and model)
"""
