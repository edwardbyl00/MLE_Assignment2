import argparse
import glob
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scripts.ML_model.model_inference import select_model_name
    from scripts.ML_model.model_registry import ensure_directory
except ModuleNotFoundError:
    from model_inference import select_model_name
    from model_registry import ensure_directory


"""Default stability thresholds. Warning values are informational>> retrain values
are used to set retrain_required=True."""

default_psi_warning_threshold = 0.10
default_psi_retrain_threshold = 0.25
default_csi_warning_threshold = 0.10
default_csi_retrain_threshold = 0.25
default_bins = 10
default_categorical_top_n = 5

# Small floor value to avoid divide-by-zero and log(0) in PSI/CSI calculations.
epsilon = 1e-6

# Use Airflow paths inside the container and repo-relative paths for local runs.
base_directory = "/opt/airflow" if os.path.isdir("/opt/airflow") else "."
default_model_bank_directory = os.path.join(base_directory, "model_bank")
default_feature_store_directory = os.path.join(
    base_directory,
    "datamart/gold/feature_store",
)
default_output_directory = os.path.join(
    base_directory,
    "datamart/gold/model_monitoring",
)


def _month_starts(start_value, end_value):
    # Normalize any date-like inputs to month starts so partition names line up.
    start = pd.Timestamp(start_value).replace(day=1)
    end = pd.Timestamp(end_value).replace(day=1)
    return pd.date_range(start=start, end=end, freq="MS")


def _load_feature_partitions(feature_store_directory, start_value, end_value):
    # Load the model's reference population from the feature window used in training.
    frames = []
    missing = []
    for snapshot_date in _month_starts(start_value, end_value):
        date_clean = snapshot_date.strftime("%Y_%m_%d")
        path = os.path.join(
            feature_store_directory,
            f"gold_feature_store_{date_clean}.parquet",
        )
        if not os.path.exists(path):
            missing.append(path)
            continue
        frames.append(pd.read_parquet(path))

    if missing:
        raise FileNotFoundError(
            "Missing model monitoring reference feature-store partitions. "
            "Monitoring compares the current snapshot against the feature "
            "window used to train the selected model, so those historical gold "
            "feature partitions must exist before monitoring can run. Missing "
            "partitions: " + ", ".join(missing)
        )
    if not frames:
        raise ValueError("No feature-store partitions loaded.")

    return pd.concat(frames, ignore_index=True)


def _load_snapshot_features(feature_store_directory, snapshotdate):
    # Load the current population whose drift we want to monitor.
    date_clean = snapshotdate.replace("-", "_")
    path = os.path.join(
        feature_store_directory,
        f"gold_feature_store_{date_clean}.parquet",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature-store partition not found: {path}")
    return pd.read_parquet(path)


def _score_features(model_artifact, features_pdf):
    # Reuse the artifact's locked feature list and fitted preprocessing pipeline.
    feature_cols = model_artifact.get("selected_features")
    if not feature_cols:
        feature_cols = [
            column for column in features_pdf.columns
            if column.startswith("fe_")
        ]
    if not feature_cols:
        raise ValueError("No model feature columns available for monitoring.")

    missing_features = sorted(set(feature_cols).difference(features_pdf.columns))
    if missing_features:
        raise ValueError(f"Feature-store columns missing: {missing_features}")

    x_values = features_pdf[feature_cols]
    transformers = model_artifact["preprocessing_transformers"]
    if "preprocessor" in transformers:
        # Current artifacts store the full numeric/categorical ColumnTransformer.
        x_values = transformers["preprocessor"].transform(x_values)
    else:
        # Older artifacts only stored a scaler, so they can only handle numeric inputs.
        non_numeric = [
            column for column in feature_cols
            if not pd.api.types.is_numeric_dtype(features_pdf[column])
        ]
        if non_numeric:
            raise ValueError(
                "Legacy scaler artifact can only monitor numeric selected "
                f"features. Non-numeric columns found: {non_numeric}"
            )
        x_values = transformers["stdscaler"].transform(x_values)

    scores = model_artifact["model"].predict_proba(x_values)[:, 1]
    return pd.Series(scores, name="model_predictions"), feature_cols


def _stability_index(expected_counts, actual_counts):
    # PSI and CSI use the same stability-index formula over bucket distributions.
    expected = np.asarray(expected_counts, dtype=float)
    actual = np.asarray(actual_counts, dtype=float)

    expected_pct = expected / max(expected.sum(), epsilon)
    actual_pct = actual / max(actual.sum(), epsilon)
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _numeric_edges(reference_series, bins):
    # Build quantile buckets from the reference distribution, then apply them to both samples.
    reference = pd.to_numeric(reference_series, errors="coerce").dropna()
    if reference.empty:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(reference, quantiles))
    if len(edges) < 2:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bucket_numeric(series, edges):
    # Missing numeric values get their own bucket instead of being dropped.
    if edges is None:
        return pd.Series("__missing__", index=series.index, dtype="object")
    buckets = pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).astype("object")
    return buckets.where(pd.notna(buckets), "__missing__").astype(str)


def _top_categorical_levels(reference_series, top_n):
    # Keep only the dominant reference categories so high-cardinality strings
    # do not create thousands of tiny CSI buckets.
    reference_values = (
        reference_series
        .astype("object")
        .where(pd.notna(reference_series), "__missing__")
        .astype(str)
    )
    levels = reference_values.value_counts().head(top_n).index.tolist()
    if "__missing__" in set(reference_values.unique()) and "__missing__" not in levels:
        levels.append("__missing__")
    return set(levels)


def _bucket_categorical(series, reference_levels):
    # Rare categories and categories unseen in training are grouped as "__other__".
    values = series.astype("object").where(pd.notna(series), "__missing__").astype(str)
    return values.where(values.isin(reference_levels), "__other__")


def _score_psi(reference_scores, current_scores, bins):
    # PSI measures drift in the final model score distribution.
    edges = _numeric_edges(reference_scores, bins)
    reference_buckets = _bucket_numeric(reference_scores, edges)
    current_buckets = _bucket_numeric(current_scores, edges)
    categories = sorted(set(reference_buckets).union(set(current_buckets)))
    reference_counts = reference_buckets.value_counts().reindex(categories, fill_value=0)
    current_counts = current_buckets.value_counts().reindex(categories, fill_value=0)

    return _stability_index(reference_counts.values, current_counts.values), len(categories)


def _feature_csi(reference_pdf, current_pdf, feature_cols, bins, categorical_top_n):
    # CSI measures drift one input feature at a time.
    rows = []
    for feature_name in feature_cols:
        reference_series = reference_pdf[feature_name]
        current_series = current_pdf[feature_name]

        if pd.api.types.is_numeric_dtype(reference_series):
            feature_type = "numeric"
            edges = _numeric_edges(reference_series, bins)
            reference_buckets = _bucket_numeric(reference_series, edges)
            current_buckets = _bucket_numeric(current_series, edges)
        else:
            feature_type = "categorical"
            reference_levels = _top_categorical_levels(
                reference_series,
                categorical_top_n,
            )
            reference_buckets = _bucket_categorical(reference_series, reference_levels)
            current_buckets = _bucket_categorical(current_series, reference_levels)

        categories = sorted(set(reference_buckets).union(set(current_buckets)))
        reference_counts = reference_buckets.value_counts().reindex(
            categories,
            fill_value=0,
        )
        current_counts = current_buckets.value_counts().reindex(
            categories,
            fill_value=0,
        )
        csi = _stability_index(reference_counts.values, current_counts.values)

        rows.append(
            {
                "feature_name": feature_name,
                "feature_type": feature_type,
                "csi": csi,
                "reference_count": int(reference_counts.sum()),
                "current_count": int(current_counts.sum()),
                "bucket_count": len(categories),
                "categorical_top_n": (
                    categorical_top_n if feature_type == "categorical" else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


def _status(value, warning_threshold, retrain_threshold):
    # Convert each numeric drift value into a simple monitoring status.
    if value >= retrain_threshold:
        return "retrain_required"
    if value >= warning_threshold:
        return "warning"
    return "healthy"


def _write_outputs(
    output_directory,
    model_version,
    summary_row,
    csi_detail_df,
):
    # Write both machine-friendly parquet and easy-to-inspect CSV outputs.
    output_directory = ensure_directory(output_directory)
    model_directory = ensure_directory(os.path.join(output_directory, model_version))

    date_clean = summary_row["snapshot_date_clean"]
    history_path = os.path.join(
        model_directory,
        f"{model_version}_monitoring_history.parquet",
    )
    history_csv_path = os.path.join(
        model_directory,
        f"{model_version}_monitoring_history.csv",
    )
    detail_path = os.path.join(
        model_directory,
        f"{model_version}_csi_detail_{date_clean}.parquet",
    )
    detail_csv_path = os.path.join(
        model_directory,
        f"{model_version}_csi_detail_{date_clean}.csv",
    )

    new_history = pd.DataFrame([summary_row])
    if os.path.exists(history_path):
        # Keep monitoring history idempotent for a rerun of the same model/date.
        old_history = pd.read_parquet(history_path)
        history = pd.concat([old_history, new_history], ignore_index=True)
        history = history.drop_duplicates(
            subset=["model_version", "snapshot_date"],
            keep="last",
        )
    else:
        history = new_history

    history = history.sort_values(["snapshot_date", "model_version"]).reset_index(drop=True)
    history.to_parquet(history_path, index=False)
    history.to_csv(history_csv_path, index=False)
    csi_detail_df.to_parquet(detail_path, index=False)
    csi_detail_df.to_csv(detail_csv_path, index=False)

    return {
        "history_path": history_path,
        "history_csv_path": history_csv_path,
        "detail_path": detail_path,
        "detail_csv_path": detail_csv_path,
    }


def main(
    snapshotdate,
    modelname=None,
    model_type=None,
    model_bank_directory=default_model_bank_directory,
    feature_store_directory=default_feature_store_directory,
    output_directory=default_output_directory,
    bins=default_bins,
    psi_warning_threshold=default_psi_warning_threshold,
    psi_retrain_threshold=default_psi_retrain_threshold,
    csi_warning_threshold=default_csi_warning_threshold,
    csi_retrain_threshold=default_csi_retrain_threshold,
    categorical_top_n=default_categorical_top_n,
):
    # Select champion by default, or use the explicitly requested model.
    model_name = select_model_name(modelname, model_bank_directory, model_type=model_type)
    model_version = os.path.splitext(model_name)[0]
    model_path = os.path.join(model_bank_directory, model_name)

    with open(model_path, "rb") as file:
        model_artifact = pickle.load(file)

    data_dates = model_artifact.get("data_dates", {})
    feature_start_date = data_dates.get("feature_start_date")
    feature_end_date = data_dates.get("feature_end_date")
    if feature_start_date is None or feature_end_date is None:
        raise ValueError(
            "Model artifact does not contain feature_start_date and feature_end_date."
        )

    # Reference = model training feature window; current = monitored snapshot.
    reference_pdf = _load_feature_partitions(
        feature_store_directory,
        feature_start_date,
        feature_end_date,
    )
    current_pdf = _load_snapshot_features(feature_store_directory, snapshotdate)

    # Score both populations so PSI reflects model-output drift, not just raw inputs.
    reference_scores, feature_cols = _score_features(model_artifact, reference_pdf)
    current_scores, _ = _score_features(model_artifact, current_pdf)

    # PSI is score-level drift; CSI is the worst feature-level drift.
    psi, psi_bucket_count = _score_psi(reference_scores, current_scores, bins)
    csi_detail = _feature_csi(
        reference_pdf,
        current_pdf,
        feature_cols,
        bins,
        categorical_top_n,
    )

    psi_row = pd.DataFrame([{
        "feature_name": "model_score",
        "feature_type": "score",
        "csi": psi,
        "reference_count": int(len(reference_scores)),
        "current_count": int(len(current_scores)),
        "bucket_count": psi_bucket_count,
        "categorical_top_n": np.nan,
    }])
    csi_detail = pd.concat([psi_row, csi_detail], ignore_index=True)
    feature_csi_detail = csi_detail[csi_detail["feature_name"] != "model_score"]
    max_csi = (
        float(feature_csi_detail["csi"].max())
        if not feature_csi_detail.empty
        else 0.0
    )
    max_csi_feature = ""
    if not feature_csi_detail.empty:
        max_csi_feature = str(
            feature_csi_detail.sort_values("csi", ascending=False)
            .iloc[0]["feature_name"]
        )

    psi_status = _status(psi, psi_warning_threshold, psi_retrain_threshold)
    csi_status = _status(max_csi, csi_warning_threshold, csi_retrain_threshold)
    # Retraining is triggered by score drift only; CSI is informational.
    retrain_required = psi >= psi_retrain_threshold
    status = "retrain_required" if retrain_required else (
        "warning" if "warning" in {psi_status, csi_status} else "healthy"
    )

    snapshot_date_clean = snapshotdate.replace("-", "_")
    # This row is used by Airflow branching and by the persisted monitoring history.
    summary_row = {
        "model_version": model_version,
        "snapshot_date": snapshotdate,
        "snapshot_date_clean": snapshot_date_clean,
        "reference_feature_start_date": str(pd.Timestamp(feature_start_date).date()),
        "reference_feature_end_date": str(pd.Timestamp(feature_end_date).date()),
        "reference_count": int(len(reference_pdf)),
        "current_count": int(len(current_pdf)),
        "psi_score": psi,
        "psi_status": psi_status,
        "max_csi": max_csi,
        "max_csi_feature": max_csi_feature,
        "csi_status": csi_status,
        "status": status,
        "retrain_required": bool(retrain_required),
        "psi_warning_threshold": psi_warning_threshold,
        "psi_retrain_threshold": psi_retrain_threshold,
        "csi_warning_threshold": csi_warning_threshold,
        "csi_retrain_threshold": csi_retrain_threshold,
        "categorical_top_n": categorical_top_n,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_paths = _write_outputs(
        output_directory,
        model_version,
        summary_row,
        csi_detail,
    )
    summary_row.update(output_paths)

    print("model monitoring completed")
    print("model_version:", model_version)
    print("snapshot_date:", snapshotdate)
    print("psi_score:", round(psi, 6), psi_status)
    print("max_csi:", round(max_csi, 6), csi_status)
    print("max_csi_feature:", summary_row["max_csi_feature"])
    print("status:", status)
    print("retrain_required:", retrain_required)
    print("history_path:", output_paths["history_path"])
    print("detail_path:", output_paths["detail_path"])

    return summary_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor model score and feature stability with PSI and CSI."
    )
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--modelname",
        default=None,
        help="Optional model filename/version. Defaults to champion model.",
    )
    parser.add_argument("--model-bank", default=default_model_bank_directory)
    parser.add_argument(
        "--feature-store",
        default=default_feature_store_directory,
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_directory,
    )
    parser.add_argument("--bins", type=int, default=default_bins)
    parser.add_argument(
        "--psi-warning-threshold",
        type=float,
        default=default_psi_warning_threshold,
    )
    parser.add_argument(
        "--psi-retrain-threshold",
        type=float,
        default=default_psi_retrain_threshold,
    )
    parser.add_argument(
        "--csi-warning-threshold",
        type=float,
        default=default_csi_warning_threshold,
    )
    parser.add_argument(
        "--csi-retrain-threshold",
        type=float,
        default=default_csi_retrain_threshold,
    )
    parser.add_argument(
        "--categorical-top-n",
        type=int,
        default=default_categorical_top_n,
        help=(
            "Number of top reference categories kept for categorical CSI. "
            "Remaining categories are grouped as __other__."
        ),
    )
    args = parser.parse_args()

    main(
        snapshotdate=args.snapshotdate,
        modelname=args.modelname,
        model_bank_directory=args.model_bank,
        feature_store_directory=args.feature_store,
        output_directory=args.output_dir,
        bins=args.bins,
        psi_warning_threshold=args.psi_warning_threshold,
        psi_retrain_threshold=args.psi_retrain_threshold,
        csi_warning_threshold=args.csi_warning_threshold,
        csi_retrain_threshold=args.csi_retrain_threshold,
        categorical_top_n=args.categorical_top_n,
    )
