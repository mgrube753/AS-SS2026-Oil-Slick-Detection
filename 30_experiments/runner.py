from train import run_training
import argparse

"""
Argparse-based runner script to execute training
with different models and split types (4 combinations).
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name",
    type=str,
    choices=["resnet50", "terramind"],
    default="resnet50",
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
    run_training(model_name=args.model_name, split_type=args.split_type)
