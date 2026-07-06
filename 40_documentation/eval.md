# [eval.py](../30_experiments/eval.py)

Evaluation script that finds the best checkpoint per configuration from MLflow logs, runs test set inference, and computes classification metrics, visualizations, and result exports.

## Overview

A comprehensive evaluation module for the 4 best-performing models (BaselineCNN and TerraMindClassifier on random and geographic splits). It orchestrates test set inference, computes classification metrics, generates visualizations (confusion matrix, probability histogram), and exports results.

## Core Functions

* **`find_best_checkpoint(split_type, model_type)`** Searches MLflow logs directory to locate the best model checkpoint based on lowest validation loss for a given (model, split) configuration. **Returns:** tuple of (checkpoint_path, best_val_loss, best_run_id).

* **`run_test_inference(checkpoint_path, split_type, model_type)`** Loads model from checkpoint, creates test loader, and runs inference to collect predictions, probabilities, and true labels. **Returns:** dict with labels, preds, probs, and dataset.

* **`compute_metrics(labels, preds, probs)`** Computes classification metrics: accuracy, precision, recall, F1 score, AUC-ROC, and confusion matrix. **Returns:** dict with all metrics.

* **`evaluate_model_checkpoint(checkpoint_path, split_type, model_type)`** Orchestrates full evaluation pipeline: inference → metric computation. **Returns:** dict containing metrics, labels, predictions, probabilities, and dataset.

* **`load_model(checkpoint_path, model_type)`** Instantiates correct model class (`BaselineCNN` or `TerraMindClassifier`), loads state dict from checkpoint, and sets to eval mode. **Returns:** model object on device.

* **`run_model_inference(model, test_loader)`** Executes forward passes on test loader with no gradient computation. Applies sigmoid activation and 0.5 threshold for binary predictions. **Returns:** tuple of (all_labels, all_preds, all_probs).

* **`save_confusion_matrix(cm, output_dir)`** Generates confusion matrix plot using scikit-learn's `ConfusionMatrixDisplay` and saves as `confusion_matrix.png`.

* **`save_probability_histogram(labels, probs, output_dir)`** Creates dual histogram showing predicted probability distributions for negative and positive samples with decision threshold line at 0.5. Saves as `probability_histogram.png`.

* **`save_classification_report(labels, preds, output_dir)`** Generates formatted classification report with precision, recall, and F1 per class. Saves as `classification_report.txt`.

* **`save_predictions_csv(labels, preds, probs, split_type, model_type, output_dir)`** Exports per-sample predictions to CSV with columns: index, true_label, pred_label, probability, correct. Saves as `test_predictions.csv`.

## Dependencies

* **Internal:** `config`, `model`, `dataloader`
* **External:** `torch`,
*               `sklearn.metrics`
*               `matplotlib.pyplot`
