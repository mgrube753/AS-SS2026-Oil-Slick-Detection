Run from 30_experiments/
Depending on which runner configuration you are using, start the mlflow logging with the respective command below in a separate terminal.

linux:
export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://$(pwd)/random_split/cnn/logs
export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://$(pwd)/geographic_split/cnn/logs
export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://$(pwd)/random_split/gfm/logs
export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://$(pwd)/geographic_split/gfm/logs

windows:
set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://%cd%/random_split/cnn/logs
set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://%cd%/geographic_split/cnn/logs
set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://%cd%/random_split/gfm/logs
set MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri file://%cd%/geographic_split/gfm/logs
