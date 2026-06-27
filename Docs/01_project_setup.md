# Project Setup

This guide explains how to run the monthly credit-risk ML pipeline locally with Docker, Airflow, Spark, XGBoost/logistic-regression models, and Streamlit.

## 1. Project Components

```text
Airflow DAGs   -> orchestrate data processing, training, inference, monitoring, and simulation
Python scripts -> build bronze/silver/gold data and run ML lifecycle logic
Streamlit app  -> displays monitoring, predictions, model registry, performance, and EDA
JupyterLab     -> optional interactive development service
```

Important files:

| Area | File |
| --- | --- |
| Historical DAG | `dags/dag.py` |
| Simulation DAG | `dags/simulation_inference_dag.py` |
| Dashboard | `dashboard.py` |
| Synthetic generator | `scripts/utils/generate_synthetic_gold_data.py` |
| Model training | `scripts/ML_model/model_train.py` |
| Model inference | `scripts/ML_model/model_inference.py` |
| Model monitoring | `scripts/ML_model/model_monitoring.py` |
| Prediction evaluation | `scripts/ML_model/model_performance.py` |
| Registry reconciliation | `scripts/ML_model/model_registry.py` |

## 2. Docker Mounts And Outputs

Docker mounts the local project folders into the Airflow container:

```text
./dags       -> /opt/airflow/dags
./scripts    -> /opt/airflow/scripts
./data       -> /opt/airflow/data
./datamart   -> /opt/airflow/datamart
./model_bank -> /opt/airflow/model_bank
```

Main output locations:

```text
datamart/bronze/
datamart/silver/
datamart/gold/feature_store/
datamart/gold/label_store/
datamart/gold/model_predictions/
datamart/gold/model_predictions_csv/
datamart/gold/model_monitoring/
datamart/gold/model_performance/
model_bank/model_log.csv
model_bank/*.pkl
```

## 3. Recommended Hardware

Minimum:

```text
CPU: 4 cores
RAM: 8 GB
Disk: 20 GB free
```

Recommended:

```text
CPU: 8 cores
RAM: 16 GB
Disk: 40 GB free
```

The compose file uses Airflow `LocalExecutor`:

```yaml
AIRFLOW__CORE__EXECUTOR: LocalExecutor
AIRFLOW__CORE__PARALLELISM: 8
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG: 8
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG: 1
```

If your machine is memory constrained, reduce `PARALLELISM` and `MAX_ACTIVE_TASKS_PER_DAG` in `docker-compose.yaml`.

## 4. Start Airflow

Start the webserver and scheduler:

```bash
docker compose up airflow-webserver airflow-scheduler
```

Open Airflow:

```text
http://localhost:8080
username: admin
password: admin
```

If DAG changes do not appear:

```bash
docker compose restart airflow-scheduler airflow-webserver
```

## 5. Optional JupyterLab

Start JupyterLab:

```bash
docker compose up jupyter
```

Open:

```text
http://localhost:8888
```

The project root is mounted at `/app` inside the Jupyter container.

## 6. Run The Historical DAG

The main DAG is named:

```text
dag
```

It runs monthly from `2023-01-01` through `2024-12-01` with catchup enabled:

```text
source CSVs
  -> bronze layer
  -> silver layer
  -> gold feature and label stores
  -> scheduled model training
  -> model inference
  -> model monitoring
  -> monitoring-triggered retraining
```

Trigger it from the terminal:

```bash
docker compose run --rm airflow-webserver airflow dags trigger dag
```

You can also unpause `dag` in the Airflow UI and let the historical catchup run.

## 7. Main DAG Flow

The DAG first builds label and feature data tracks:

```text
setup_directories
  -> dep_check_source_label_data
  -> bronze_label_store
  -> silver_label_store
  -> gold_label_store
  -> label_store_completed

setup_directories
  -> dep_check_source_data_bronze_1
  -> bronze_feature_store
  -> silver_feature_store
  -> gold_feature_store
  -> feature_store_completed
```

Then it runs the model lifecycle:

```text
label_store_completed + feature_store_completed
  -> scheduled_training_start
      -> scheduled_xgboost_training OR skip_scheduled_xgboost_training
      -> scheduled_log_reg_training OR skip_scheduled_log_reg_training
  -> scheduled_training_completed
  -> model_inference_start
      -> model_xgboost_inference
      -> model_log_reg_inference
  -> model_inference_completed
  -> model_monitor_start
      -> model_xgboost_monitor
      -> model_log_reg_monitor
  -> model_monitor_completed
  -> model_automl_start
      -> model_xgboost_automl OR skip_xgboost_retraining
      -> model_log_reg_automl OR skip_log_reg_retraining
  -> model_automl_completed
```

Pink skipped tasks are expected when a branch is not selected.

## 8. Bronze, Silver, And Gold Layers

```text
Bronze -> land monthly source data close to raw format
Silver -> clean and standardize data
Gold   -> create model-ready labels and features
```

Gold feature creation includes derived fields such as:

```text
debt_to_income_ratio
emi_to_salary_ratio
high_credit_utilization_flag
Credit_History_Age_Months
```

The label maturity rule is:

```text
feature snapshot month M -> label month M + 6
```

## 9. Model Training

Training starts from:

```text
FIRST_TRAINING_DATE = 2024-09-01
```

This is the first month with enough matured-label history. Scheduled training runs every 3 months after the latest training date recorded in `model_bank/model_log.csv`.

Both model families are trained when scheduled training is due:

```text
xgboost
log_reg
```

Training uses:

```text
12 months train/test window
2 months out-of-time window
25-iteration stochastic hyperparameter search
numeric/categorical preprocessing with imputation
sigmoid probability calibration
threshold selection by Youden's J on the validation split
model registration in model_bank/model_log.csv
```

Model artifacts are saved as:

```text
model_bank/credit_model_<model_type>_YYYY_MM_DD_vN.pkl
```

The registry stores model type, train date, AUC/Gini/Brier/log-loss metrics, calibration details, threshold details, and champion/challenger flags.

## 10. Inference, Monitoring, And Retraining

Inference defaults to the champion model for each model type and writes only missing prediction months:

```text
datamart/gold/model_predictions/<model_version>/<model_version>_predictions_YYYY_MM_DD.parquet
datamart/gold/model_predictions_csv/<model_version>/<model_version>_predictions_YYYY_MM_DD.csv
```

Monitoring compares the current feature snapshot against the selected model's training feature window:

```text
PSI -> model score drift
CSI -> feature drift
PSI >= 0.25 -> retrain_required
CSI -> informational diagnostic signal
```

Monitoring outputs are written to:

```text
datamart/gold/model_monitoring/<model_version>/
```

If PSI requires retraining, the DAG branches to the matching model family's AutoML retraining task.

## 11. Start Streamlit

Run the dashboard from the repo root:

```bash
streamlit run dashboard.py
```

Open:

```text
http://localhost:8501
```

The dashboard is read-only. It reads files created by Airflow and the ML scripts; it does not trigger training, inference, monitoring, simulation, or evaluation.

## 12. Clean Setup Checklist

1. Start Airflow with Docker Compose.
2. Open `http://localhost:8080`.
3. Unpause or trigger the main `dag`.
4. Let the historical backfill complete through `2024-12-01`.
5. Confirm `model_bank/model_log.csv` and prediction/monitoring outputs exist.
6. Start Streamlit and review Monitoring, Predictions, and EDA.
7. Run `simulation_inference_dag` if future synthetic months are needed.
