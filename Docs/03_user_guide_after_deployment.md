# User Guide After Deployment

This guide explains how to use the project after the pipeline has been deployed and Airflow/Streamlit are running.

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

Start Streamlit:

```bash
streamlit run dashboard.py
```

## 2. What Runs Where

| Tool | Purpose |
| --- | --- |
| Airflow | Runs the main historical DAG and simulation DAG. |
| Python scripts | Process data, train models, infer predictions, monitor drift, and evaluate performance. |
| Streamlit | Displays results after Airflow/scripts have created output files. |

The dashboard is read-only. It does not trigger training, inference, monitoring, or simulation.

## 3. Normal Operating Flow

For historical pipeline execution:

```text
Start Airflow
  -> run or unpause dag
  -> wait for historical backfill to complete
  -> review outputs in Streamlit
```

For future simulation:

```text
Start Airflow
  -> trigger simulation_inference_dag
  -> wait for synthetic generation, inference, monitoring, and evaluation
  -> refresh Streamlit
```

For dashboard review:

```text
Start Streamlit
  -> choose model version
  -> review Monitoring, Predictions, and EDA pages
```

## 4. Airflow DAGs

There are two main DAGs:

| DAG | When to use |
| --- | --- |
| `dag` | Historical production-style pipeline from January 2023 to December 2024. |
| `simulation_inference_dag` | Manual synthetic future simulation after the main DAG has created gold data and a model. |

Trigger the main DAG:

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

Green tasks mean success.

Pink skipped tasks are usually expected because the DAG uses branches. For example, if training is not due, the training task may be skipped and the skip branch succeeds.

Red tasks mean failure and should be checked from the task log.

Common expected skips:

```text
scheduled training skipped because the 3-month interval has not passed
inference skipped because predictions already exist
monitoring skipped because no eligible monitoring snapshot exists
retraining skipped because PSI did not require retraining
```

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
Which model is active?
Is model score drift healthy?
Which features are drifting?
Does the model need retraining?
How has PSI/CSI changed over time?
```

Important tabs:

| Tab | Purpose |
| --- | --- |
| Drift Overview | PSI trend, max CSI trend, current drift status. |
| Feature Drift | CSI by feature for a selected snapshot. |
| Model Registry | Champion/challenger model records from `model_log.csv`. |
| Model Performance | Metrics after predictions are compared with matured labels. |

PSI interpretation:

```text
PSI < 0.10       -> healthy
0.10 to 0.25    -> warning
PSI >= 0.25     -> retrain_required
```

CSI interpretation:

```text
CSI < 0.10       -> healthy
0.10 to 0.25    -> warning
CSI >= 0.25     -> material feature drift
```

CSI helps explain what moved. PSI is the retraining trigger in the current setup.

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
| `prediction_threshold` | Probability cutoff selected during training. |
| `label` | Model-predicted binary class, not the actual future label. |

### EDA

Use this page to inspect raw pipeline outputs:

```text
bronze partitions
silver partitions
gold feature store
gold label store
row counts
column summaries
missing values
```

This is useful when drift or performance changes might be caused by data changes rather than model behavior.

## 7. Model Registry

The model registry is stored in:

```text
model_bank/model_log.csv
```

It records model metadata such as:

```text
model name
training date
train/test/OOT periods
selected threshold
calibration method
champion/challenger status
performance metrics
```

Champion model:

```text
the active model used by default for inference
```

Challenger model:

```text
a valid model candidate that did not replace the champion
```

## 8. Prediction Performance

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
```

## 9. Refreshing Data

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

## 10. Common Troubleshooting

### DAG changes do not appear

Restart Airflow:

```bash
docker compose restart airflow-scheduler airflow-webserver
```

### Simulation did not infer new months

Likely causes:

```text
prediction files already exist
synthetic gold feature partitions were not created
max_snapshotdate is too early
no champion model exists
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

## 11. Daily User Checklist

1. Open Airflow and confirm DAG runs are successful.
2. Open Streamlit and select the champion model.
3. Review Drift Overview for PSI status.
4. Review Feature Drift for high-CSI features.
5. Review Predictions for scored volume and probability distribution.
6. Review Model Performance when labels have matured.
7. Use EDA if drift or performance changes need data-level investigation.
