from train import run_training
import argparse
from config import Config

"""
Argparse-based runner script to execute training
with different models and split types (4 combinations).
These combinations are used for grid search over
learning rates and weight decays.
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
    choices=["baselinecnn", "terramind"],
    default="baselinecnn",
    help="Name of the model to train",
)
parser.add_argument(
    "--split-type",
    type=str,
    choices=["random", "geographic"],
    default="random",
    help="Type of split to use",
)

args = parser.parse_args()

if __name__ == "__main__":
    lrs = (
        Config.LEARNING_RATES_CNN
        if args.model_name == "baselinecnn"
        else Config.LEARNING_RATES_GFM
    )
    wds = (
        Config.WEIGHT_DECAYS_CNN
        if args.model_name == "baselinecnn"
        else Config.WEIGHT_DECAYS_GFM
    )

    total_runs = len(lrs) * len(wds)
    counter = 0

    print(f"Grid Search for {args.model_name}...")
    for lr in lrs:
        for wd in wds:
            counter += 1
            run_training(
                model_name=args.model_name,
                split_type=args.split_type,
                lr=lr,
                wd=wd,
                total_runs=total_runs,
                counter=counter,
            )
