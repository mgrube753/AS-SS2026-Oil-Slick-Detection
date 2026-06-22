import os

MLFLOW_DIR = "mlflow_extracted/mlflow"

best_models = {}

for experiment_id in os.listdir(MLFLOW_DIR):

    exp_path = os.path.join(MLFLOW_DIR, experiment_id)

    if not os.path.isdir(exp_path):
        continue

    for run_id in os.listdir(exp_path):

        run_path = os.path.join(exp_path, run_id)

        params_dir = os.path.join(run_path, "params")
        metrics_dir = os.path.join(run_path, "metrics")

        if not os.path.exists(params_dir):
            continue

        try:
            with open(os.path.join(params_dir, "model")) as f:
                model = f.read().strip()

            with open(os.path.join(params_dir, "split")) as f:
                split = f.read().strip()

            with open(os.path.join(params_dir, "lr")) as f:
                lr = f.read().strip()

            with open(os.path.join(params_dir, "weight_decay")) as f:
                wd = f.read().strip()

            with open(os.path.join(metrics_dir, "best_val_loss")) as f:
                last_line = f.readlines()[-1].strip().split()
                val_loss = float(last_line[1])
        except Exception:
            continue

        key = (split, model)

        if key not in best_models:
            best_models[key] = {
                "val_loss": val_loss,
                "lr": lr,
                "wd": wd,
                "run_id": run_id,
            }

        elif val_loss < best_models[key]["val_loss"]:
            best_models[key] = {
                "val_loss": val_loss,
                "lr": lr,
                "wd": wd,
                "run_id": run_id,
            }

print("\nBest models:\n")

for key, info in best_models.items():

    split, model = key

    print(
        f"{split} | {model} | "
        f"lr={info['lr']} | "
        f"wd={info['wd']} | "
        f"val_loss={info['val_loss']:.6f}"
    )