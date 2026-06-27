# Project Setup

This guide explains how to set up and run the credit risk ML pipeline locally with Docker, Airflow, Spark/XGBoost scripts, and Streamlit.

## 1. Project Components

The project has three main execution layers:

```text
Airflow DAGs      -> orchestrate data processing, training, inference, monitoring, and simulation
Python scripts    -> perform bronze/silver/gold processing and ML logic
Streamlit app     -> displays model monitoring, predictions, performance, and EDA
```

Important files:

| Area | File |
| --- | --- |
| Main historical DAG | `dags/dag.py` |
| Simulation DAG | `dags/simulation_inference_dag.py` |
| Dashboard | `dashboard.py` |
| Synthetic generator | `scripts/utils/generate_synthetic_gold_data.py` |
| Model training | `scripts/ML_model/model_train.py` |
| Model inference | `scripts/ML_model/model_inference.py` |
| Model monitoring | `scripts/ML_model/model_monitoring.py` |
| Prediction evaluation | `scripts/ML_model/model_performance.py` |

## 2. Folder Mounts

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
datamart/gold/model_monitoring/
datamart/gold/model_performance/
model_bank/model_log.csv
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

Better for repeated training and simulation:

```text
CPU: 8-12 cores
RAM: 32 GB
Disk: 80 GB free
```

The current Airflow configuration is designed for a laptop or small workstation:

```yaml
AIRFLOW__CORE__EXECUTOR: LocalExecutor
AIRFLOW__CORE__PARALLELISM: 8
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG: 8
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG: 1
```

If the machine has limited memory, reduce parallelism:

```yaml
AIRFLOW__CORE__PARALLELISM: 4
AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG: 4
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG: 1
```

## 4. Start Airflow

Start the Airflow webserver and scheduler:

```bash
docker compose up airflow-webserver airflow-scheduler
```

Open Airflow:

```text
http://localhost:8080
```

Default login:

```text
username: admin
password: admin
```

If DAG changes do not appear:

```bash
docker compose restart airflow-scheduler airflow-webserver
```

## 5. Run The Main Historical DAG

The main DAG is:

```text
dag
```

It processes the real historical source data from January 2023 to December 2024:

```text
source data
  -> bronze layer
  -> silver layer
  -> gold feature and label stores
  -> scheduled model training
  -> model inference
  -> model monitoring
  -> monitoring-triggered retraining
```

To trigger the main DAG from the terminal:

```bash
docker compose run --rm airflow-webserver airflow dags trigger dag
```

Usually, you can also unpause `dag` in the Airflow UI and let catchup run all monthly periods.

## 6. Main DAG Flow

The DAG runs two data tracks before ML starts:

```text
setup_directories
  -> bronze_label_store
  -> silver_label_store
  -> gold_label_store
  -> label_store_completed

setup_directories
  -> bronze_feature_store
  -> silver_feature_store
  -> gold_feature_store
  -> feature_store_completed
```

Then it runs the model lifecycle:

```text
label_store_completed + feature_store_completed
  -> scheduled_training_start
      -> scheduled_xgboost_training
      -> scheduled_log_reg_training
      -> skip_scheduled_model_training
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
      -> model_xgboost_automl
      -> model_log_reg_automl
      -> skip_model_retraining
  -> model_automl_completed
```

Pink skipped tasks in Airflow are expected when a branch is not selected.

## 7. Bronze, Silver, And Gold Layers

The medallion layers separate the data preparation lifecycle:

```text
Bronze -> land monthly source data
Silver -> clean and standardize data
Gold   -> create model-ready labels and features
```

Bronze keeps monthly source extracts close to the raw format. Silver standardizes types, cleans invalid values, and prepares reliable inputs. Gold creates the final feature and label stores used by training, inference, monitoring, and simulation.

Gold feature creation includes derived fields such as:

```text
debt_to_income_ratio
emi_to_salary_ratio
high_credit_utilization_flag
Credit_History_Age_Months
```

## 8. Model Training Setup

Training starts from:

```text
FIRST_TRAINING_DATE = 2024-09-01
```

This is the earliest point where enough matured labels exist. The label maturity rule is:

```text
feature snapshot month M -> label month M + 6
```

Scheduled model training runs every 3 months from the last training date. Training uses:

```text
12 months train/test window
2 months out-of-time window
sigmoid probability calibration
threshold selection on validation/OOT metrics
model registration in model_bank/model_log.csv
```

## 9. Start Streamlit

Run the dashboard:

```bash
streamlit run dashboard.py
```

Open:

```text
http://localhost:8501
```

The dashboard is read-only. It does not train, infer, monitor, or evaluate by itself. It reads the files created by Airflow and the scripts.

## 10. Setup Checklist

Use this order for a clean setup:

1. Start Airflow with Docker Compose.
2. Open `http://localhost:8080`.
3. Unpause or trigger the main `dag`.
4. Let the historical backfill complete through December 2024.
5. Confirm outputs exist under `datamart/gold/` and `model_bank/`.
6. Start Streamlit with `streamlit run dashboard.py`.
7. Use `Docs/02_simulation_pipeline.md` when you want to extend the pipeline with synthetic future data.
