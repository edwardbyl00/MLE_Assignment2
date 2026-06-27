import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from scripts.ML_model.model_inference import select_model_name
    from scripts.ML_model.model_registry import ensure_directory
except ModuleNotFoundError:
    from model_inference import select_model_name
    from model_registry import ensure_directory


label_mob_months = 6
default_bins = 10

# Resolve paths for both Airflow containers and local terminal runs. Inside
# Airflow the project root is /opt/airflow; locally it is the current repo.
base_directory = "/opt/airflow" if os.path.isdir("/opt/airflow") else "."
default_model_bank_directory = os.path.join(base_directory, "model_bank")
default_prediction_directory = os.path.join(
    base_directory,
    "datamart/gold/model_predictions",
)
default_label_store_directory = os.path.join(
    base_directory,
    "datamart/gold/label_store",
)
default_output_directory = os.path.join(
    base_directory,
    "datamart/gold/model_performance",
)


def _safe_metric(metric_function, y_true, values, default=np.nan):
    # AUC/log-loss are undefined when matured labels contain only one class.
    if len(pd.Series(y_true).dropna().unique()) < 2:
        return default
    try:
        return metric_function(y_true, values)
    except ValueError:
        return default


def _classification_metrics(y_true, y_pred):
    # Use a fixed [0, 1] label order so tn/fp/fn/tp are stable even when one
    # prediction class is absent.
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _prediction_files(prediction_directory, model_version):
    # Prediction outputs are partitioned by model version and snapshot month.
    model_directory = os.path.join(prediction_directory, model_version)
    return sorted(glob.glob(os.path.join(model_directory, "*.parquet")))


def _load_predictions(prediction_directory, model_version):
    # Load every monthly prediction partition for the selected model version.
    files = _prediction_files(prediction_directory, model_version)
    if not files:
        raise FileNotFoundError(
            "No prediction parquet files found for model version: "
            f"{model_version}"
        )

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["prediction_file"] = os.path.basename(path)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)

    required_columns = {"Customer_ID", "snapshot_date", "model_predictions"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")

    predictions["prediction_snapshot_date"] = pd.to_datetime(
        predictions["snapshot_date"],
    )
    # Labels mature after the model's performance window. A prediction made at
    # month M can only be evaluated once the M+6 label partition exists.
    predictions["label_snapshot_date"] = (
        predictions["prediction_snapshot_date"]
        + pd.DateOffset(months=label_mob_months)
    )
    predictions["model_version"] = model_version
    if "prediction_threshold" not in predictions.columns:
        # Older prediction files did not persist the threshold. Preserve
        # backwards compatibility by falling back to the historical default.
        predictions["prediction_threshold"] = 0.5
    if "label" in predictions.columns:
        predictions = predictions.rename(columns={"label": "predicted_label"})
    else:
        predictions["predicted_label"] = (
            predictions["model_predictions"] >= predictions["prediction_threshold"]
        ).astype(int)

    return predictions


def _load_label_store(label_store_directory, min_date, max_date):
    # Only read labels that could match the matured prediction horizon.
    label_files = sorted(glob.glob(os.path.join(label_store_directory, "*.parquet")))
    if not label_files:
        raise FileNotFoundError(f"No label parquet files found: {label_store_directory}")

    frames = []
    min_date = pd.Timestamp(min_date)
    max_date = pd.Timestamp(max_date)
    for path in label_files:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        frame = frame[
            (frame["snapshot_date"] >= min_date)
            & (frame["snapshot_date"] <= max_date)
        ]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["Customer_ID", "label", "label_def", "snapshot_date"],
        )

    labels = pd.concat(frames, ignore_index=True)
    required_columns = {"Customer_ID", "label", "snapshot_date"}
    missing = required_columns.difference(labels.columns)
    if missing:
        raise ValueError(f"Label columns missing: {sorted(missing)}")

    labels = labels.rename(
        columns={
            "snapshot_date": "label_snapshot_date",
            "label": "actual_label",
        }
    )
    aggregation = {"actual_label": "max"}
    if "label_def" in labels.columns:
        aggregation["label_def"] = "first"
    # Customer-level predictions can match multiple loans in the label store.
    # If any loan defaults for that customer/month, the customer is counted as
    # defaulted for performance evaluation.
    labels = (
        labels
        .groupby(["Customer_ID", "label_snapshot_date"], as_index=False)
        .agg(aggregation)
    )
    return labels


def _join_predictions_to_labels(predictions, labels):
    # Predictions at month M are evaluated against labels at M + label_mob_months.
    if labels.empty:
        return pd.DataFrame()

    joined = predictions.merge(
        labels,
        on=["Customer_ID", "label_snapshot_date"],
        how="inner",
    )
    return joined


def _score_band_detail(scored, bins):
    # Score bands make calibration visible: average score vs actual default rate.
    if scored.empty:
        return pd.DataFrame()

    scored = scored.copy()
    try:
        # Quantile bins keep each band roughly comparable in row count. When
        # too many scores are tied, qcut may collapse duplicate bin edges.
        scored["score_band"] = pd.qcut(
            scored["model_predictions"],
            q=bins,
            duplicates="drop",
        )
    except ValueError:
        scored["score_band"] = "all_scores"
    if scored["score_band"].isna().all():
        scored["score_band"] = "all_scores"
    detail = (
        scored
        .groupby("score_band", observed=True)
        .agg(
            row_count=("actual_label", "size"),
            min_score=("model_predictions", "min"),
            max_score=("model_predictions", "max"),
            avg_score=("model_predictions", "mean"),
            actual_default_rate=("actual_label", "mean"),
            predicted_default_rate=("predicted_label", "mean"),
        )
        .reset_index()
    )
    detail["score_band"] = detail["score_band"].astype(str)
    return detail


def _summary_for_group(scored, model_version, evaluation_date, group_name):
    # Each summary row can represent either all matured predictions or one
    # original prediction snapshot month.
    y_true = scored["actual_label"].astype(int)
    y_score = scored["model_predictions"].astype(float)
    y_pred = scored["predicted_label"].astype(int)
    classification = _classification_metrics(y_true, y_pred)
    auc = _safe_metric(roc_auc_score, y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    logloss = _safe_metric(log_loss, y_true, y_score)

    row = {
        "model_version": model_version,
        "evaluation_date": evaluation_date,
        "performance_group": group_name,
        "prediction_start_date": str(scored["prediction_snapshot_date"].min().date()),
        "prediction_end_date": str(scored["prediction_snapshot_date"].max().date()),
        "label_start_date": str(scored["label_snapshot_date"].min().date()),
        "label_end_date": str(scored["label_snapshot_date"].max().date()),
        "row_count": int(len(scored)),
        "actual_default_rate": float(y_true.mean()),
        "avg_model_prediction": float(y_score.mean()),
        "prediction_threshold": float(scored["prediction_threshold"].median()),
        "predicted_default_rate": float(y_pred.mean()),
        "auc": auc,
        "gini": np.nan if pd.isna(auc) else round(2 * auc - 1, 3),
        "brier": brier,
        "log_loss": logloss,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    row.update(classification)
    return row


def _write_outputs(output_directory, model_version, evaluation_date, summary, detail):
    # Keep outputs under the model version so performance histories do not mix
    # results from different artifacts.
    output_directory = ensure_directory(output_directory)
    model_directory = ensure_directory(os.path.join(output_directory, model_version))
    date_clean = evaluation_date.replace("-", "_")

    history_path = os.path.join(
        model_directory,
        f"{model_version}_performance_history.parquet",
    )
    history_csv_path = os.path.join(
        model_directory,
        f"{model_version}_performance_history.csv",
    )
    detail_path = os.path.join(
        model_directory,
        f"{model_version}_performance_detail_{date_clean}.parquet",
    )
    detail_csv_path = os.path.join(
        model_directory,
        f"{model_version}_performance_detail_{date_clean}.csv",
    )

    new_history = pd.DataFrame(summary)
    if os.path.exists(history_path):
        # Re-running the same evaluation date replaces the prior summary rows.
        old_history = pd.read_parquet(history_path)
        history = pd.concat([old_history, new_history], ignore_index=True)
        history = history.drop_duplicates(
            subset=["model_version", "evaluation_date", "performance_group"],
            keep="last",
        )
    else:
        history = new_history

    history = history.sort_values(
        ["evaluation_date", "model_version", "performance_group"],
    ).reset_index(drop=True)
    history.to_parquet(history_path, index=False)
    history.to_csv(history_csv_path, index=False)
    detail.to_parquet(detail_path, index=False)
    detail.to_csv(detail_csv_path, index=False)

    return {
        "history_path": history_path,
        "history_csv_path": history_csv_path,
        "detail_path": detail_path,
        "detail_csv_path": detail_csv_path,
    }


def main(
    evaluationdate,
    modelname=None,
    model_type=None,
    model_bank_directory=default_model_bank_directory,
    prediction_directory=default_prediction_directory,
    label_store_directory=default_label_store_directory,
    output_directory=default_output_directory,
    bins=default_bins,
):
    # Select champion by default, matching inference and monitoring behavior.
    model_name = select_model_name(modelname, model_bank_directory, model_type)
    model_version = os.path.splitext(model_name)[0]
    evaluation_date = pd.Timestamp(evaluationdate)

    predictions = _load_predictions(prediction_directory, model_version)
    # Only evaluate predictions whose future labels should exist by the chosen
    # evaluation date. Newer predictions remain unscored until their labels mature.
    matured_predictions = predictions[
        predictions["label_snapshot_date"] <= evaluation_date
    ].copy()
    if matured_predictions.empty:
        raise ValueError(
            "No predictions have matured labels by evaluation date "
            f"{evaluationdate}. Earliest label date needed is "
            f"{predictions['label_snapshot_date'].min().date()}."
        )

    labels = _load_label_store(
        label_store_directory,
        matured_predictions["label_snapshot_date"].min(),
        matured_predictions["label_snapshot_date"].max(),
    )
    scored = _join_predictions_to_labels(matured_predictions, labels)
    if scored.empty:
        raise ValueError(
            "No matured labels matched predictions. Check Customer_ID and "
            f"{label_mob_months}-month label snapshot offset."
        )

    summary = [
        # Overall row: one KPI set for all prediction months that have matured.
        _summary_for_group(
            scored,
            model_version,
            evaluationdate,
            "all_matured_predictions",
        )
    ]
    for prediction_date, group in scored.groupby("prediction_snapshot_date"):
        # Monthly rows: keep each original prediction snapshot visible so
        # performance degradation can be spotted over time.
        summary.append(
            _summary_for_group(
                group,
                model_version,
                evaluationdate,
                prediction_date.strftime("prediction_%Y_%m_%d"),
            )
        )

    detail = _score_band_detail(scored, bins)
    detail.insert(0, "model_version", model_version)
    detail.insert(1, "evaluation_date", evaluationdate)
    output_paths = _write_outputs(
        output_directory,
        model_version,
        evaluationdate,
        summary,
        detail,
    )

    result = summary[0].copy()
    result.update(output_paths)

    print("model performance evaluation completed")
    print("model_version:", model_version)
    print("evaluation_date:", evaluationdate)
    print("row_count:", result["row_count"])
    print("auc:", result["auc"])
    print("gini:", result["gini"])
    print("brier:", result["brier"])
    print("log_loss:", result["log_loss"])
    print("history_path:", output_paths["history_path"])
    print("detail_path:", output_paths["detail_path"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against matured labels."
    )
    parser.add_argument("--evaluationdate", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--modelname",
        default=None,
        help="Optional model filename/version. Defaults to champion model.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["xgboost", "logistic_regression"],
        help="Optional model algorithm filter.",
    )
    parser.add_argument("--model-bank", default=default_model_bank_directory)
    parser.add_argument("--prediction-dir", default=default_prediction_directory)
    parser.add_argument("--label-store", default=default_label_store_directory)
    parser.add_argument("--output-dir", default=default_output_directory)
    parser.add_argument("--bins", type=int, default=default_bins)
    args = parser.parse_args()

    main(
        evaluationdate=args.evaluationdate,
        modelname=args.modelname,
        model_type=args.model_type,
        model_bank_directory=args.model_bank,
        prediction_directory=args.prediction_dir,
        label_store_directory=args.label_store,
        output_directory=args.output_dir,
        bins=args.bins,
    )
