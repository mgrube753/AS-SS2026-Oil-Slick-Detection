import os
import matplotlib.pyplot as plt
from eval import (
    run_test_inference,
    find_best_checkpoint,
)

"""
Failure analysis script to identify and visualize
false positives and false negatives from the test
set predictions of the individual trained model.
"""


def get_failure_indices(all_labels, all_preds):
    fp_indices = []
    fn_indices = []

    for i, (y_true, y_pred) in enumerate(zip(all_labels, all_preds)):
        if y_true == 0 and y_pred == 1:
            fp_indices.append(i)
        elif y_true == 1 and y_pred == 0:
            fn_indices.append(i)

    return fp_indices, fn_indices


def save_examples(indices, dataset, save_dir, prefix):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    print(f"Saving {prefix} examples to:")
    print(os.path.abspath(save_dir))

    for idx in indices:
        image, _ = dataset[idx]

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(
            image[0],
            cmap="gray",
        )
        plt.title(f"{prefix} | VV | idx={idx}")

        plt.subplot(1, 2, 2)
        plt.imshow(
            image[1],
            cmap="gray",
        )
        plt.title("VH")

        plt.tight_layout()
        filename = os.path.join(save_dir, f"{prefix.lower()}_{idx}.png")
        plt.savefig(filename)
        plt.close()

    print(f"Saved {len(indices)} {prefix} examples to {save_dir}")


def run_failure_analysis(
    split_type,
    model_type,
):

    split_folder = "random_split" if split_type == "random" else "geographic_split"

    checkpoint_path = find_best_checkpoint(
        split_type,
        model_type,
    )
    results = run_test_inference(
        checkpoint_path,
        split_type,
        model_type,
    )

    labels = results["labels"]
    preds = results["preds"]
    test_ds = results["dataset"]

    fp_indices, fn_indices = get_failure_indices(
        labels,
        preds,
    )

    output_dir = os.path.join(
        "..",
        "50_evaluation",
        split_folder,
        model_type,
        "results",
        "failure_analysis",
    )
    print("Output directory:")
    print(os.path.abspath(output_dir))

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
    # Geographic CNN
    run_failure_analysis(
        split_type="geographic",
        model_type="cnn",
    )

    # Random CNN
    run_failure_analysis(
        split_type="random",
        model_type="cnn",
    )

    # Geographic GFM
    run_failure_analysis(
        split_type="geographic",
        model_type="gfm",
    )

    # Random GFM
    run_failure_analysis(
        split_type="random",
        model_type="gfm",
    )
