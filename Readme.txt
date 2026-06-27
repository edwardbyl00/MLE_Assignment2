Git Link: https://github.com/edwardbyl00/MLE_Assignment2.git

## Quick Start

Start Airflow:

```bash
docker compose up airflow-webserver airflow-scheduler
```

Airflow UI:

```text
http://localhost:8080
username: admin
password: admin
```

Start Streamlit:

```bash
streamlit run dashboard.py
```

```text
http://localhost:8501
```

Optional JupyterLab:

```bash
docker compose up jupyter
```

```text
http://localhost:8888
```

## Key Docs

- `Docs/01_project_setup.md` - local setup, Airflow, DAG flow, model lifecycle, and dashboard startup
- `Docs/02_simulation_pipeline.md` - synthetic future gold data, simulation DAG parameters, inference, monitoring, and evaluation
- `Docs/03_user_guide_after_deployment.md` - operating guide for Airflow, Streamlit, registry, troubleshooting, and daily checks

## Main Entrypoints

- `dags/dag.py` - historical Airflow DAG
- `dags/simulation_inference_dag.py` - synthetic future simulation DAG
- `dashboard.py` - Streamlit dashboard
- `scripts/utils/data_processing_bronze_table.py` - bronze ingestion
- `scripts/utils/data_processing_silver_table.py` - silver transformations
- `scripts/utils/data_processing_gold_table.py` - gold labels and features
- `scripts/utils/generate_synthetic_gold_data.py` - synthetic gold partition generation
- `scripts/ML_model/model_train.py` - model training for XGBoost and logistic regression
- `scripts/ML_model/model_inference.py` - champion/model-specific inference
- `scripts/ML_model/model_monitoring.py` - PSI/CSI monitoring and retraining flag
- `scripts/ML_model/model_performance.py` - matured-label prediction evaluation
- `scripts/ML_model/model_registry.py` - model-bank reconciliation
