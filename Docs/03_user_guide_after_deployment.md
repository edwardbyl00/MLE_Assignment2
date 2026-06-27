# User Guide After Deployment

This guide explains how to operate the project after Airflow and Streamlit are running.

## 1. Main Links

Airflow UI:

```text
http://localhost:8080
```

Streamlit dashboard:

```text
http://localhost:8501
```

Start Airflow:

```bash
docker compose up airflow-webserver airflow-scheduler
```

Start Streamlit from the repo root:

```bash
streamlit run dashboard.py
```

Optional JupyterLab:

```bash
docker compose up jupyter
```

```text
http://localhost:8888
```

## 2. What Runs Where

| Tool | Purpose |
| --- | --- |
| Airflow | Runs the historical `dag` and `simulation_inference_dag`. |
| Python scripts | Process data, train models, infer predictions, monitor drift, and evaluate performance. |
| Streamlit | Reads generated files and displays monitoring, predictions, registry, performance, and EDA. |
| JupyterLab | Optional notebook-style exploration and debugging. |

The dashboard is read-only. It does not trigger training, inference, monitoring, simulation, or performance evaluation.

## 3. Normal Operating Flow

Historical run:

```text
Start Airflow
  -> run or unpause dag
  -> wait for historical backfill through 2024-12-01
  -> review outputs in Streamlit
```

Future simulation:

```text
Start Airflow
  -> trigger simulation_inference_dag
  -> wait for synthetic generation, inference, monitoring, and evaluation
  -> refresh Streamlit
```

Dashboard review:

```text
Start Streamlit
  -> choose model version
  -> review Monitoring, Predictions, and EDA pages
```

## 4. Airflow DAGs

| DAG | When to use |
| --- | --- |
| `dag` | Historical production-style pipeline from `2023-01-01` to `2024-12-01`. |
| `simulation_inference_dag` | Manual synthetic future simulation after the historical DAG has created gold data and models. |

Trigger the historical DAG:

```bash
docker compose run --rm airflow-webserver airflow dags trigger dag
```

Trigger the simulation DAG:

```bash
docker compose run --rm airflow-webserver airflow dags trigger simulation_inference_dag
```

Trigger simulation to June 2026:

```bash
docker compose run --rm airflow-webserver airflow dags trigger simulation_inference_dag \
  --conf '{"synthetic_end_date":"2026-06-01","max_snapshotdate":"2026-06-01","performance_evaluation_date":"2026-06-01"}'
```

## 5. Reading Airflow

Green tasks mean success. Red tasks mean failure and should be checked from the task log.

Pink skipped tasks are usually expected because both DAGs use branches. Common expected skips:

```text
scheduled training skipped because the 3-month interval has not passed
XGBoost or logistic-regression retraining skipped because PSI did not require retraining
simulation inference skipped because predictions already exist
simulation monitoring skipped because monitoring was disabled or no eligible snapshots exist
simulation evaluation skipped because labels have not matured yet
```

The simulation DAG has separate XGBoost and logistic-regression inference, monitoring, and evaluation branches. It is normal to see one branch family succeed or skip independently of the other.

## 6. Streamlit Dashboard Navigation

The dashboard has three top-level pages:

```text
Monitoring
Predictions
EDA
```

### Monitoring

Use this page to answer:

```text
Which model version is selected?
Is model score drift healthy?
Which features are drifting?
Does the model need retraining?
How has PSI/CSI changed over time?
How does matured-label performance look?
```

Important tabs:

| Tab | Purpose |
| --- | --- |
| Drift Overview | PSI trend, max CSI trend, latest drift status. |
| Feature Drift | CSI by feature for a selected snapshot. |
| Model Registry | Champion/challenger records from `model_bank/model_log.csv`. |
| Model Performance | Metrics after predictions are compared with matured labels. |

PSI interpretation:

```text
PSI < 0.10    -> healthy
0.10 to 0.25 -> warning
PSI >= 0.25  -> retrain_required
```

CSI interpretation:

```text
CSI < 0.10    -> healthy
0.10 to 0.25 -> warning
CSI >= 0.25  -> material feature drift
```

PSI is the retraining trigger in the current setup. CSI is diagnostic and helps explain what moved.

### Predictions

Use this page to answer:

```text
Which snapshot dates were scored?
How many customers were scored?
What probability distribution did the model produce?
What threshold was used?
What final predicted labels were assigned?
How does performance look once labels mature?
```

Important fields:

| Field | Meaning |
| --- | --- |
| `model_predictions` | Calibrated model probability. |
| `prediction_threshold` | Probability cutoff selected during training or supplied as an override. |
| `label` | Model-predicted binary class in prediction output files. |
| `predicted_label` | Renamed prediction label used inside performance evaluation. |
| `actual_label` | Matured label joined from the label store during performance evaluation. |

Prediction outputs are stored under:

```text
datamart/gold/model_predictions/<model_version>/
datamart/gold/model_predictions_csv/<model_version>/
```

### EDA

Use this page to inspect pipeline outputs:

```text
bronze partitions
silver partitions
gold feature store
gold label store
row counts
column summaries
missing values
```

EDA is useful when drift or performance changes may be caused by data movement rather than model behavior.

## 7. Model Registry

The model registry is stored in:

```text
model_bank/model_log.csv
```

It records:

```text
model_version
model_type
training date
AUC/Gini/Brier/log-loss metrics
calibration method
selected threshold
threshold selection metric
champion/challenger status
```

Model artifacts are stored as:

```text
model_bank/credit_model_<model_type>_YYYY_MM_DD_vN.pkl
```

Champion model:

```text
the active model selected by default for inference, monitoring, and evaluation
```

Challenger model:

```text
the best non-champion model by registry reconciliation, or a newly staged model that beats the champion on test and OOT AUC
```

The registry can be reconciled from valid `.pkl` artifacts by `scripts/ML_model/model_registry.py`. Reconciliation preserves one valid existing champion when possible; otherwise it promotes the best available artifact by OOT, test, and train AUC.

## 8. Model Families

The historical DAG trains, infers, monitors, and retrains both:

```text
xgboost
log_reg
```

Inference, monitoring, and performance evaluation are model-type aware in both the historical DAG and the simulation DAG. When a specific `model_name` is supplied, it must match the expected model type for that task.

## 9. Prediction Performance

Prediction performance uses the 6-month label maturity rule:

```text
prediction month + 6 months = label month
```

Example:

```text
prediction 2025-12-01 -> evaluated against label 2026-06-01
```

Performance output is written to:

```text
datamart/gold/model_performance/<model_version>/
```

Use the Model Performance tab to review:

```text
AUC
Gini
Brier score
Log loss
Accuracy
Precision
Recall
F1
score-band calibration
monthly matured prediction performance
```

The evaluator writes one overall row for all matured predictions and one row per matured prediction snapshot.

## 10. Refreshing Data

After a DAG run finishes:

1. Go to Streamlit.
2. Click refresh if available.
3. Re-select the model version or snapshot date.
4. Confirm new output dates appear.

If new dates do not appear, check that files exist under:

```text
datamart/gold/model_predictions/
datamart/gold/model_monitoring/
datamart/gold/model_performance/
```

## 11. Common Troubleshooting

### DAG changes do not appear

Restart Airflow:

```bash
docker compose restart airflow-scheduler airflow-webserver
```

### Historical inference has no model

Check:

```text
gold feature and label stores exist
FIRST_TRAINING_DATE has enough historical data
model_bank exists as a directory
model_bank/model_log.csv can be reconciled
```

If no champion exists, inference can attempt bootstrap training for `2024-09-01`.

### Simulation did not infer new months

Likely causes:

```text
prediction files already exist for the selected model version
synthetic gold feature partitions were not created
max_snapshotdate is too early
infer_through_latest_gold changed the effective upper bound
no valid champion or requested model exists
the supplied model_name does not match the model family branch
```

### Dashboard shows only one snapshot date

Likely causes:

```text
only one prediction or monitoring output exists
Streamlit cache needs refresh
the selected model version only has one output month
simulation monitoring was run for one snapshot only
```

### Feature drift is high

Check:

```text
whether synthetic data was regenerated with the latest logic
whether ratio fields were recalculated and capped
whether categorical categories shifted
whether the feature has many unseen values
whether the gold feature distribution changed upstream
```

### Performance evaluation is empty

Check:

```text
labels have matured by the evaluation date
prediction files exist for the selected model
label partitions exist for prediction month + 6
Customer_ID values overlap between predictions and labels
```

## 12. Daily User Checklist

1. Open Airflow and confirm DAG runs are successful.
2. Open Streamlit and select the relevant model version.
3. Review Drift Overview for PSI status.
4. Review Feature Drift for high-CSI features.
5. Review Predictions for scored volume and probability distribution.
6. Review Model Performance when labels have matured.
7. Use EDA if drift or performance changes need data-level investigation.
