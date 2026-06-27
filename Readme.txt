# Assignment 2: Monthly Credit-Risk ML Pipeline

This project runs a monthly Apache Airflow pipeline for loan default prediction.
It builds bronze, silver, and gold datamart layers, trains and versions an
XGBoost model, runs backfilled inference, and includes a notebook prototype for
model monitoring.

## Quick Start

```bash
docker compose up airflow-webserver airflow-scheduler
```

Airflow UI:

```text
http://localhost:8080
username: admin
password: admin
```

JupyterLab:

```bash
docker compose up jupyter
```

```text
http://localhost:8888
```

## Key Docs

- `Docs/first_time_setup.md` - first-time Docker and Airflow setup
- `Docs/dag_documentation.md` - DAG task graph, constants, validation, and skips
- `Docs/model_train_documentation.md` - XGBoost training, artifacts, and champion logic
- `Docs/model_registry_logic.md` - registry reconciliation and champion recovery
- `Docs/ml_model_logic.md` - learning guide for training, inference, and registry logic
- `Docs/logic log.md` - end-to-end project status and open gaps

## Main Entrypoints

- `dags/dag.py` - Airflow DAG
- `scripts/utils/data_processing_bronze_table.py` - bronze ingestion
- `scripts/utils/data_processing_silver_table.py` - silver transformations
- `scripts/utils/data_processing_gold_table.py` - gold labels and features
- `scripts/ML_model/model_train.py` - model training
- `scripts/ML_model/model_inference.py` - model inference
- `scripts/ML_model/model_monitoring.py` - PSI/CSI monitoring and retraining flag
- `scripts/ML_model/model_registry.py` - model-bank reconciliation
