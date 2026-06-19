# Oil Slick Detection in Sentinel-1 SAR Imagery

[![Python 3.12.4](https://img.shields.io/badge/python-3.12.4-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Repo Size](https://img.shields.io/github/repo-size/mgrube753/AS-SS2026-Oil-Slick-Detection.svg)](https://github.com/mgrube753/AS-SS2026-Oil-Slick-Detection)
[![University of Rostock](https://img.shields.io/badge/Institution-University_of_Rostock-003D7A.svg)](https://www.uni-rostock.de/)

This repository contains the code, data, and documentation for the Area Seminar "Deep Learning for Maritime Vision Applications" (summer semester 2026) at the University of Rostock. The project deals with the detection of oil slicks in Sentinel-1 SAR imagery. Therefore, a subset of the WaterBench dataset is provided, named OilSlick, which includes Sentinel-1 SAR chips for the project task.

The research questions of the project are:

- **RQ1:** How well can deep learning detect oil slicks in Sentinel-1 SAR imagery, and how does performance change between an in-distribution random split and a geographic out-of-distribution split (Mediterranean)?
- **RQ2:** (Optional:) Can a Geospatial Foundation Model (GFM) improve over a Convolutional Neural Network (CNN) baseline, and what Explainable AI (XAI) methods reveal about the features which drive the predictions.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/mgrube753/AS-SS2026-Oil-Slick-Detection.git
    cd AS-SS2026-Oil-Slick-Detection
    ```

2. Create a virtual environment for e.g. Python 3.12.4 (recommended):

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. In your environment, install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Download & Extract Data

While being in your environment, download the OilSlick subset from the WaterBench dataset (on [Hugging Face](https://huggingface.co/datasets/ayushprd/WaterBench)) by running the [`setup_data.sh`](./setup_data.sh) script from the repository root directory:

```bash
./setup_data.sh
```

The progress bar may appear delayed, but the download is progressing in the background. Please wait until the download is complete. Then, the script will automatically extract the downloaded archive and move it to the [`10_waterbench_data/`](10_waterbench_data/) directory.

## Exploration of Data

The [`20_data_analysis/`](20_data_analysis/) directory contains a Jupyter Notebook thats provides an exploratory data analysis of the dataset, incuding dataset information, statistics and according visualizations, plots of sample SAR chips, and a filtering process to enhance the validity of the OilSlick dataset. The filtered dataset and the resulting metadata (stored in [`20_data_analysis/`](20_data_analysis/) after execution) are then used for further experimenting.

## Run Experiments & Monitor Results

1. All experiments are overall located in the [`30_experiments/`](30_experiments/) directory. Each experiment is run by executing the [`runner.py`](30_experiments/runner.py) file in this directory. It includes 4 experimental configurations, based on the model (Baseline CNN vs. Geo Foundation Model Terramind) and the split configuration (random split vs. geographic split)

    Run the respective experimental configuration of interest by executing the proper command in [`30_experiments/`](30_experiments/):

    ```bash
    python runner.py --model-name baselinecnn --split-type random

    python runner.py --model-name baselinecnn --split-type geographic

    python runner.py --model-name terramind --split-type random

    python runner.py --model-name terramind --split-type geographic
    ```

    Each of the 4 runs executes grid search over model-specific hyperparameter set combinations. This includes 3 learning rates and 3 weight decays for each model, resulting in 4x3x3=36 model trainings.
2. While the experiments are running, you can monitor the training progress and loggings using MLflow. Run the respective command in [`30_experiments/`](30_experiments/) to start the MLflow UI:

    Linux (and probably MacOS):

    ```bash
    export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file:///$(pwd)/logs/mlflow
    ```

    Windows:

    ```bash
    set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file:///%cd%/logs/mlflow
    ```

    The UI can be accessed by opening a web browser for [`http://127.0.0.1:5000`](http://127.0.0.1:5000). There, you can see the 4 experimental runs, and in each run, you can see the 9 model trainings with their individual hyperparameter combinations and loggings.

For further experimental details, please refer to the [`30_experiments/`](30_experiments/) directory and its [`README.md`](30_experiments/README.md) file. For each Python file, a documentation file is also provided in [`40_documentation/`](40_documentation/) to understand the task, the classes and the functions.

## Evaluation, Failure Analysis & XAI

(This step is unfinished, needs update soon)

1. After running the experiments and obtaining the best model checkpoints for the 4 experimental configurations, evaluation is crucial. Run [`eval.py`](30_experiments/eval.py) and [`failure_analysis.py`](30_experiments/failure_analysis.py) in [`30_experiments/`](30_experiments/) as follows:

    ```bash
    python eval.py --model-name baselinecnn --split-type random
    python eval.py --model-name baselinecnn --split-type geographic
    python eval.py --model-name terramind --split-type random
    python eval.py --model-name terramind --split-type geographic

    python failure_analysis.py --model-name baselinecnn --split-type random
    python failure_analysis.py --model-name baselinecnn --split-type geographic
    python failure_analysis.py --model-name terramind --split-type random
    python failure_analysis.py --model-name terramind --split-type geographic
    ```

    By this, results are obtained and visualized for the checkpoints, which are then saved in [`50_evaluation/`](50_evaluation/) respectively. The evaluation includes inference on the test set (just the respective one, or maybe both test split sets for each model?), calculation of evaluation metrics, and also creating confusion matrices.

2. For the CNN baseline, Grad-CAM is applied as an XAI method to visualize the respective features which drive the model's predictions. The visualized heatmaps are also saved in [`50_evaluation/`](50_evaluation/) for the respective split types. Run the following command for Grad-CAM:

    ```bash
    python gradcam.py --model-name baselinecnn --split-type random
    python gradcam.py --model-name baselinecnn --split-type geographic
    ```

    For the Geo Foundation Model, no XAI method is applied, due to time constraints. For future work, it would be interesting to apply XAI to the GFM as a comparison to the CNN baseline.
