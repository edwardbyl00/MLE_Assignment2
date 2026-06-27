import argparse
import glob
import os
import pickle
import pprint
import re
from datetime import datetime

import pandas as pd
import pyspark

try:
    from scripts.ML_model.model_registry import reconcile_model_log
    import scripts.ML_model.model_train as model_train
except ModuleNotFoundError:
    from model_registry import reconcile_model_log
    import model_train


BOOTSTRAP_MODEL_TRAIN_DATE = "2024-09-01"
DEFAULT_PREDICTION_THRESHOLD = 0.5


def _parse_date_from_name(path, pattern):
    match = re.search(pattern, os.path.basename(path))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y_%m_%d").strftime("%Y-%m-%d")


def gold_feature_snapshot_dates(feature_store_directory):
    """Return snapshot dates available in the gold feature-store volume."""
    paths = glob.glob(
        os.path.join(feature_store_directory, "gold_feature_store_*.parquet")
    )
    dates = [
        _parse_date_from_name(path, r"gold_feature_store_(\d{4}_\d{2}_\d{2})\.parquet$")
        for path in paths
    ]
    return sorted(date for date in dates if date is not None)


def prediction_snapshot_dates(prediction_directory, model_version):
    """Return snapshot dates already predicted for a model version."""
    model_directory = os.path.join(prediction_directory, model_version)
    paths = glob.glob(
        os.path.join(
            model_directory,
            f"{model_version}_predictions_*.parquet",
        )
    )
    dates = [
        _parse_date_from_name(
            path,
            rf"{re.escape(model_version)}_predictions_(\d{{4}}_\d{{2}}_\d{{2}})\.parquet$",
        )
        for path in paths
    ]
    return sorted(date for date in dates if date is not None)


def unpredicted_gold_snapshot_dates(
    model_version,
    feature_store_directory="datamart/gold/feature_store",
    prediction_directory="datamart/gold/model_predictions",
    min_snapshotdate=None,
    max_snapshotdate=None,
):
    """Find gold feature partitions newer than the model's prediction outputs."""
    feature_dates = gold_feature_snapshot_dates(feature_store_directory)
    predicted_dates = set(prediction_snapshot_dates(prediction_directory, model_version))

    if min_snapshotdate:
        feature_dates = [date for date in feature_dates if date >= min_snapshotdate]
    if max_snapshotdate:
        feature_dates = [date for date in feature_dates if date <= max_snapshotdate]

    return [date for date in feature_dates if date not in predicted_dates]


def select_model_name(modelname=None, model_bank_directory="model_bank/", model_type=None):
    """Return a validated model filename, defaulting to the champion model."""
    reconcile_model_log(model_bank_directory)

    if modelname:
        model_name = os.path.basename(str(modelname).strip())
        selection_source = "user supplied"
    else:
        log_path = os.path.join(model_bank_directory, "model_log.csv")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Model log not found: {log_path}")

        log_df = pd.read_csv(log_path)
        if model_type:
            if "model_type" not in log_df.columns:
                raise ValueError("model_log.csv has no model_type column")
            model_rows = log_df[log_df["model_type"].eq(model_type)].copy()
            if model_rows.empty:
                raise ValueError(f"No {model_type} model found in model_log.csv")
            model_rows = model_rows.sort_values(
                ["auc_oot", "auc_test", "auc_train", "model_version"],
                ascending=[False, False, False, True],
            )
            model_name = str(model_rows.iloc[0]["model_version"]).strip()
            selection_source = model_type
        else:
            champion_rows = log_df[log_df["champion"] == 1]
            if champion_rows.empty:
                raise ValueError("No champion model found in model_log.csv")
            if len(champion_rows) > 1:
                raise ValueError("More than one champion model found in model_log.csv")

            model_name = str(champion_rows.iloc[0]["model_version"]).strip()
            selection_source = "champion"

    if not model_name.endswith(".pkl"):
        model_name += ".pkl"

    model_path = os.path.join(model_bank_directory, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    print(f"Using {selection_source} model: {model_name}")
    return model_name


def select_or_train_model_name(
    modelname=None,
    model_bank_directory="model_bank/",
    spark=None,
    bootstrap_train_date=BOOTSTRAP_MODEL_TRAIN_DATE,
    model_type=None,
):
    """Return a model name, training one bootstrap champion when none exists."""
    try:
        return select_model_name(modelname, model_bank_directory, model_type)
    except (FileNotFoundError, ValueError) as error:
        if modelname:
            raise
        if spark is None:
            raise

        print(
            "No usable champion model found. "
            "Checking model bank for the best available valid model."
        )
        summary = reconcile_model_log(model_bank_directory)
        if summary.get("champion") and model_type is None:
            print("Best available model selected as champion:", summary["champion"])
            return select_model_name(None, model_bank_directory, model_type)

        print(
            "No valid model artifacts found in model bank. "
            f"Attempting bootstrap training for {bootstrap_train_date}. "
            f"Original issue: {error}"
        )
        try:
            model_train.train_model(
                bootstrap_train_date,
                spark,
                model_type=model_type or "xgboost",
            )
        except Exception as training_error:
            raise RuntimeError(
                "No champion model is available, and bootstrap training could "
                f"not create one for {bootstrap_train_date}. Ensure the gold "
                "label and feature stores contain enough history before running "
                "inference."
            ) from training_error

        return select_model_name(None, model_bank_directory, model_type)


def main(
    snapshotdate,
    modelname=None,
    prediction_threshold=None,
    model_type=None,
):
    """Score one gold feature-store snapshot with the selected model artifact."""
    print("\n\n---starting job---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_inference")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        model_bank_directory = "model_bank/"
        model_name = select_or_train_model_name(
            modelname,
            model_bank_directory,
            spark,
            model_type=model_type,
        )

        config = {
            "snapshot_date_str": snapshotdate,
            "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
            "model_name": model_name,
            "model_bank_directory": model_bank_directory,
            "model_artefact_filepath": os.path.join(model_bank_directory, model_name),
        }
        pprint.pprint(config)

        with open(config["model_artefact_filepath"], "rb") as file:
            model_artefact = pickle.load(file)
        print("Model loaded successfully! " + config["model_artefact_filepath"])
        # Use the threshold selected during training unless a run explicitly
        # supplies an override for scenario testing.
        selected_prediction_threshold = (
            model_artefact.get("prediction_threshold", DEFAULT_PREDICTION_THRESHOLD)
            if prediction_threshold is None
            else prediction_threshold
        )

        snapshot_date_clean = config["snapshot_date_str"].replace("-", "_")
        feature_location = (
            "datamart/gold/feature_store/"
            f"gold_feature_store_{snapshot_date_clean}.parquet"
        )
        if not os.path.exists(feature_location):
            raise FileNotFoundError(
                f"Gold feature-store partition not found: {feature_location}"
            )

        features_pdf = spark.read.parquet(feature_location).toPandas()
        print("Loaded feature store:", feature_location, "row count:", len(features_pdf))

        selected_features = model_artefact.get("selected_features")
        if selected_features:
            feature_cols = selected_features
        else:
            # Legacy artifacts did not store selected_features, so keep the old
            # convention as a compatibility fallback.
            feature_cols = [
                column for column in features_pdf.columns
                if column.startswith("fe_")
            ]
        if not feature_cols:
            raise ValueError("No model feature columns found for inference")

        missing_features = set(feature_cols).difference(features_pdf.columns)
        if missing_features:
            raise ValueError(
                f"Feature-store columns missing: {sorted(missing_features)}"
            )

        x_inference = features_pdf[feature_cols]
        preprocessing_transformers = model_artefact["preprocessing_transformers"]
        if "preprocessor" in preprocessing_transformers:
            # Current artifacts transform numeric and categorical features with
            # the fitted training preprocessor.
            transformer = preprocessing_transformers["preprocessor"]
            x_inference = transformer.transform(x_inference)
        else:
            non_numeric_feature_cols = [
                column for column in feature_cols
                if not pd.api.types.is_numeric_dtype(features_pdf[column])
            ]
            if non_numeric_feature_cols:
                raise ValueError(
                    "Selected model features must be numeric for legacy artifacts. "
                    f"Non-numeric columns found: {non_numeric_feature_cols}"
                )
            transformer = preprocessing_transformers["stdscaler"]
            x_inference = transformer.transform(x_inference)
        print("X_inference", x_inference.shape[0])

        model = model_artefact["model"]
        predictions = model.predict_proba(x_inference)[:, 1]

        inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
        inference_pdf["model_name"] = config["model_name"]
        inference_pdf["model_predictions"] = predictions
        # Persist the cutoff beside every prediction so downstream evaluation
        # can reproduce the binary label even if model defaults change later.
        inference_pdf["prediction_threshold"] = selected_prediction_threshold
        inference_pdf["label"] = (
            inference_pdf["model_predictions"] >= selected_prediction_threshold
        ).astype(int)

        model_version = os.path.splitext(config["model_name"])[0]
        gold_directory = f"datamart/gold/model_predictions/{model_version}/"
        csv_directory = f"datamart/gold/model_predictions_csv/{model_version}/"
        os.makedirs(gold_directory, exist_ok=True)
        os.makedirs(csv_directory, exist_ok=True)

        parquet_path = os.path.join(
            gold_directory,
            f"{model_version}_predictions_{snapshot_date_clean}.parquet",
        )
        csv_path = os.path.join(
            csv_directory,
            f"{model_version}_predictions_{snapshot_date_clean}.csv",
        )
        inference_pdf.to_parquet(parquet_path, index=False)
        inference_pdf.to_csv(csv_path, index=False)
        print("saved to:", parquet_path)
        print("saved to:", csv_path)
    finally:
        spark.stop()

    print("\n\n---completed job---\n\n")


def run_new_gold_predictions(
    modelname=None,
    prediction_threshold=None,
    min_snapshotdate=None,
    max_snapshotdate=None,
    model_type=None,
    model_bank_directory="model_bank/",
    feature_store_directory="datamart/gold/feature_store",
    prediction_directory="datamart/gold/model_predictions",
):
    """Run inference only for gold feature partitions missing predictions."""
    if min_snapshotdate is None:
        min_snapshotdate = BOOTSTRAP_MODEL_TRAIN_DATE

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_inference_discovery")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        model_name = select_or_train_model_name(
            modelname,
            model_bank_directory,
            spark,
            model_type=model_type,
        )
    finally:
        spark.stop()

    model_version = os.path.splitext(model_name)[0]
    # This implements the volume trigger: new gold partitions are inferred only
    # when that model version has no prediction file for the same snapshot date.
    snapshot_dates = unpredicted_gold_snapshot_dates(
        model_version=model_version,
        feature_store_directory=feature_store_directory,
        prediction_directory=prediction_directory,
        min_snapshotdate=min_snapshotdate,
        max_snapshotdate=max_snapshotdate,
    )

    if not snapshot_dates:
        print(
            "No new gold feature partitions require inference for",
            model_version,
        )
        return {
            "model_name": model_name,
            "model_version": model_version,
            "snapshot_dates": [],
            "prediction_count": 0,
        }

    for snapshot_date in snapshot_dates:
        print(f"Running inference for new gold snapshot {snapshot_date}")
        main(snapshot_date, model_name, prediction_threshold, model_type)

    return {
        "model_name": model_name,
        "model_version": model_version,
        "snapshot_dates": snapshot_dates,
        "prediction_count": len(snapshot_dates),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--modelname",
        type=str,
        required=False,
        default=None,
        help="Optional model filename/version; defaults to the champion model",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["xgboost", "logistic_regression"],
        help="Optional model algorithm filter.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        required=False,
        default=None,
        help="Optional probability cutoff override; defaults to artifact threshold",
    )
    parser.add_argument(
        "--new-gold-only",
        action="store_true",
        help="Predict only gold feature partitions without existing predictions.",
    )
    parser.add_argument(
        "--min-snapshotdate",
        default=None,
        help="Optional lower snapshot-date bound for --new-gold-only.",
    )
    parser.add_argument(
        "--max-snapshotdate",
        default=None,
        help="Optional upper snapshot-date bound for --new-gold-only.",
    )
    args = parser.parse_args()
    if args.new_gold_only:
        run_new_gold_predictions(
            modelname=args.modelname,
            prediction_threshold=args.prediction_threshold,
            min_snapshotdate=args.min_snapshotdate,
            max_snapshotdate=args.max_snapshotdate or args.snapshotdate,
            model_type=args.model_type,
        )
    else:
        main(
            args.snapshotdate,
            args.modelname,
            args.prediction_threshold,
            args.model_type,
        )
