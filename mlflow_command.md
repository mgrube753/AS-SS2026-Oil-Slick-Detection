Run from 30_experiments/
All 4 experimental runner configuration loggings will be available by running the MLflow UI with the following command:

Linux:
export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file:///$(pwd)/logs/mlflow

Windows:
set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file:///%cd%/logs/mlflow
