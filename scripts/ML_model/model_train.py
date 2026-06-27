import glob
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import add_months, col
from pyspark.sql.types import NumericType
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

try:
    from scripts.ML_model.feature_selection import select_model_feature_columns
    from scripts.ML_model.model_registry import LOG_COLUMNS, ensure_directory
except ModuleNotFoundError:
    from feature_selection import select_model_feature_columns
    from model_registry import LOG_COLUMNS, ensure_directory


# Threshold selection is separated from probability calibration. Calibration
# makes the score more probability-like; this metric chooses the binary cutoff.
THRESHOLD_SELECTION_METRIC = "youden_j"

MODEL_CONFIGS = {
    "xgboost": {
        "version_prefix": "credit_model",
        "search_iterations": 25,
        "param_dist": {
            "n_estimators": [25, 30, 35],
            "max_depth": [2, 4, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.6, 0.8],
            "colsample_bytree": [0.6, 0.8],
            "gamma": [0, 0.1],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 1.5, 2],
        },
    },
    "logistic_regression": {
        "version_prefix": "credit_model_log_reg",
        "search_iterations": 8,
        "param_dist": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        },
    },
}


def _build_candidate_model(model_type, params):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=88,
            n_jobs=-1,
            **params,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=88,
            n_jobs=-1,
            **params,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_prefit_calibrator(model, method, x_values, y_values):
    """Fit a sklearn calibrator around an already-fitted classifier."""
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )
        calibrator.fit(x_values, y_values)
        return calibrator

    try:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
    except TypeError:
        calibrator = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv="prefit",
        )
    calibrator.fit(x_values, y_values)
    return calibrator


def _score_probabilities(y_true, probabilities):
    """Return ranking and probability calibration metrics."""
    return {
        "auc": roc_auc_score(y_true, probabilities),
        "gini": round(2 * roc_auc_score(y_true, probabilities) - 1, 3),
        "brier": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
    }


def _classification_metrics_at_threshold(y_true, probabilities, threshold):
    """Return binary classification metrics at one probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    y_pred = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "youden_j": recall - false_positive_rate,
    }


def _select_prediction_threshold(y_true, probabilities):
    """Select an operating threshold from calibrated probabilities."""
    candidate_thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    threshold_results = [
        _classification_metrics_at_threshold(y_true, probabilities, threshold)
        for threshold in candidate_thresholds
    ]
    selected_result = max(
        threshold_results,
        key=lambda result: (
            result[THRESHOLD_SELECTION_METRIC],
            result["f1"],
            result["balanced_accuracy"],
        ),
    )
    return selected_result, threshold_results


def train_model(date_str, spark, model_type="xgboost"):
    """Train, evaluate, save, and register a versioned credit-risk model."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Expected one of: {sorted(MODEL_CONFIGS)}"
        )

    model_config = MODEL_CONFIGS[model_type]
    # The label is observed after a 6-month MOB window, so training joins each
    # label month back to features from six months earlier.
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8
    stochastic_search_iterations = model_config["search_iterations"]
    label_mob_months = 6

    config = {
        "model_train_date_str": date_str,
        "model_type": model_type,
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "label_mob_months": label_mob_months,
        "stochastic_search_iterations": stochastic_search_iterations,
        "model_train_date": datetime.strptime(date_str, "%Y-%m-%d"),
        "train_test_ratio": train_test_ratio,
    }
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(
        months=oot_period_months
    )
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(
        months=train_test_period_months
    )
    config["feature_start_date"] = config["train_test_start_date"] - relativedelta(
        months=label_mob_months
    )
    config["feature_end_date"] = config["oot_end_date"] - relativedelta(
        months=label_mob_months
    )
    pprint.pprint(config)

    # Read every gold partition, then filter by the calculated windows. This
    # keeps the Airflow task simple while the date logic stays in one place.
    label_files = glob.glob("/opt/airflow/datamart/gold/label_store/*")
    feature_files = glob.glob("/opt/airflow/datamart/gold/feature_store/*")
    if not label_files or not feature_files:
        raise FileNotFoundError("Gold label and feature stores are required for training.")

    label_store_sdf = spark.read.parquet(*label_files)
    features_store_sdf = spark.read.parquet(*feature_files)

    labels_sdf = (
        label_store_sdf
        .filter(
            (col("snapshot_date") >= config["train_test_start_date"])
            & (col("snapshot_date") <= config["oot_end_date"])
        )
        .withColumn(
            "feature_snapshot_date",
            add_months(col("snapshot_date"), -label_mob_months),
        )
    )
    features_sdf = (
        features_store_sdf
        .withColumnRenamed("snapshot_date", "feature_snapshot_date")
        .filter(
            (col("feature_snapshot_date") >= config["feature_start_date"])
            & (col("feature_snapshot_date") <= config["feature_end_date"])
        )
    )

    joined_sdf = labels_sdf.join(
        features_sdf,
        on=["Customer_ID", "feature_snapshot_date"],
        how="left",
    )

    # Feature selection is saved into the artifact so inference uses the same
    # training schema even if future gold partitions contain extra columns.
    feature_cols = select_model_feature_columns(joined_sdf.columns)
    if not feature_cols:
        raise ValueError("No model feature columns found after exclusions.")
    matched_feature_rows = joined_sdf.select(*feature_cols).dropna(
        how="all",
        subset=feature_cols,
    ).count()
    if matched_feature_rows == 0:
        raise ValueError(
            "No label rows matched feature rows. Check Customer_ID and the "
            f"{label_mob_months}-month label-to-feature snapshot offset."
        )

    schema_by_column = {
        field.name: field.dataType for field in joined_sdf.schema.fields
    }
    numeric_feature_cols = [
        column for column in feature_cols
        if isinstance(schema_by_column[column], NumericType)
    ]
    categorical_feature_cols = [
        column for column in feature_cols
        if column not in numeric_feature_cols
    ]
    print("numeric features:", numeric_feature_cols)
    print("categorical features:", categorical_feature_cols)

    oot_sdf = joined_sdf.filter(
        (col("snapshot_date") >= config["oot_start_date"])
        & (col("snapshot_date") <= config["oot_end_date"])
    )
    train_test_sdf = joined_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"])
        & (col("snapshot_date") <= config["train_test_end_date"])
    )

    if train_test_sdf.count() == 0:
        raise ValueError("No train/test records available.")
    if oot_sdf.count() == 0:
        raise ValueError("No OOT records available.")
    if train_test_sdf.select("label").distinct().count() < 2:
        raise ValueError("Train/test data must contain both label classes.")
    if oot_sdf.select("label").distinct().count() < 2:
        raise ValueError("OOT data must contain both label classes.")

    train_test_pdf = train_test_sdf.select("label", *feature_cols).toPandas()
    oot_pdf = oot_sdf.select("label", *feature_cols).toPandas()

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    # Stratification keeps both default/non-default classes represented in the
    # holdout set, which stabilizes AUC and threshold selection.
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols],
        train_test_pdf["label"],
        test_size=1 - train_test_ratio,
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"],
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_feature_cols),
            ("cat", categorical_transformer, categorical_feature_cols),
        ]
    )
    fitted_preprocessor = preprocessor.fit(X_train)
    X_train_processed = fitted_preprocessor.transform(X_train)
    X_test_processed = fitted_preprocessor.transform(X_test)
    X_oot_processed = fitted_preprocessor.transform(X_oot)

    best_model = None
    best_params = None
    best_search_auc = -1.0
    sampled_params = list(
        ParameterSampler(
            model_config["param_dist"],
            n_iter=stochastic_search_iterations,
            random_state=88,
        )
    )
    for index, params in enumerate(sampled_params, start=1):
        # Use a bounded random search to keep DAG runtime predictable.
        candidate_model = _build_candidate_model(model_type, params)
        candidate_model.fit(X_train_processed, y_train)
        candidate_auc = roc_auc_score(
            y_test,
            candidate_model.predict_proba(X_test_processed)[:, 1],
        )
        print(
            f"{model_type} stochastic search {index}/{len(sampled_params)} "
            f"TEST AUC={candidate_auc:.4f} params={params}"
        )
        if candidate_auc > best_search_auc:
            best_search_auc = candidate_auc
            best_model = candidate_model
            best_params = params

    selected_calibration_method = "sigmoid"
    # Sigmoid calibration is fixed by design: it is stable for smaller samples
    # and avoids isotonic overfitting on limited validation data.
    final_model = _fit_prefit_calibrator(
        best_model,
        selected_calibration_method,
        X_test_processed,
        y_test,
    )
    calibration_candidates = {
        selected_calibration_method: final_model,
    }
    calibration_results = {}
    for method, candidate_model in calibration_candidates.items():
        calibration_results[method] = {
            "train": _score_probabilities(
                y_train,
                candidate_model.predict_proba(X_train_processed)[:, 1],
            ),
            "test": _score_probabilities(
                y_test,
                candidate_model.predict_proba(X_test_processed)[:, 1],
            ),
            "oot": _score_probabilities(
                y_oot,
                candidate_model.predict_proba(X_oot_processed)[:, 1],
            ),
        }
        oot_metrics = calibration_results[method]["oot"]
        print(
            f"Calibration {method}: "
            f"OOT BRIER={oot_metrics['brier']:.4f} "
            f"LOGLOSS={oot_metrics['log_loss']:.4f} "
            f"AUC={oot_metrics['auc']:.4f}"
        )

    selected_calibration_results = calibration_results[selected_calibration_method]
    selected_test_probabilities = final_model.predict_proba(X_test_processed)[:, 1]
    selected_oot_probabilities = final_model.predict_proba(X_oot_processed)[:, 1]
    # The threshold is selected on the test split, then reported on OOT to show
    # how the chosen cutoff behaves out of time.
    selected_threshold_result, threshold_search_results = (
        _select_prediction_threshold(y_test, selected_test_probabilities)
    )
    selected_prediction_threshold = selected_threshold_result["threshold"]
    oot_threshold_result = _classification_metrics_at_threshold(
        y_oot,
        selected_oot_probabilities,
        selected_prediction_threshold,
    )
    train_auc_score = calibration_results[selected_calibration_method]["train"]["auc"]
    test_auc_score = calibration_results[selected_calibration_method]["test"]["auc"]
    oot_auc_score = calibration_results[selected_calibration_method]["oot"]["auc"]

    print("Best parameters:", best_params)
    print("Selected calibration:", selected_calibration_method)
    print(
        "Selected threshold:",
        selected_prediction_threshold,
        f"({THRESHOLD_SELECTION_METRIC}="
        f"{selected_threshold_result[THRESHOLD_SELECTION_METRIC]:.4f})",
    )
    print(f"TRAIN  AUC={train_auc_score:.4f}  GINI={round(2 * train_auc_score - 1, 3)}")
    print(f"TEST   AUC={test_auc_score:.4f}  GINI={round(2 * test_auc_score - 1, 3)}")
    print(f"OOT    AUC={oot_auc_score:.4f}  GINI={round(2 * oot_auc_score - 1, 3)}")

    model_bank_directory = ensure_directory("/opt/airflow/model_bank/")
    base_version = model_config["version_prefix"] + "_" + date_str.replace("-", "_")
    existing_versions = glob.glob(
        os.path.join(model_bank_directory, base_version + "_v*.pkl")
    )
    model_version = f"{base_version}_v{len(existing_versions) + 1}"

    model_artefact = {
        # Store both the calibrated model used in production and the raw base
        # model for diagnostics or future comparison.
        "model": final_model,
        "base_model": best_model,
        "model_type": model_type,
        "model_version": model_version,
        "preprocessing_transformers": {"preprocessor": fitted_preprocessor},
        "data_dates": config,
        "data_stats": {
            "X_train": X_train.shape[0],
            "X_test": X_test.shape[0],
            "X_oot": X_oot.shape[0],
            "y_train": round(y_train.mean(), 2),
            "y_test": round(y_test.mean(), 2),
            "y_oot": round(y_oot.mean(), 2),
        },
        "results": {
            "auc_train": train_auc_score,
            "auc_test": test_auc_score,
            "auc_oot": oot_auc_score,
            "gini_train": round(2 * train_auc_score - 1, 3),
            "gini_test": round(2 * test_auc_score - 1, 3),
            "gini_oot": round(2 * oot_auc_score - 1, 3),
            "brier_train": selected_calibration_results["train"]["brier"],
            "brier_test": selected_calibration_results["test"]["brier"],
            "brier_oot": selected_calibration_results["oot"]["brier"],
            "log_loss_train": selected_calibration_results["train"]["log_loss"],
            "log_loss_test": selected_calibration_results["test"]["log_loss"],
            "log_loss_oot": selected_calibration_results["oot"]["log_loss"],
        },
        "calibration_method": selected_calibration_method,
        "calibration_selection_metric": "fixed_sigmoid",
        "calibration_results": calibration_results,
        "prediction_threshold": selected_prediction_threshold,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "threshold_selection_results": {
            "test_selected": selected_threshold_result,
            "oot_at_selected_threshold": oot_threshold_result,
            "test_search_grid": threshold_search_results,
        },
        "selected_features": feature_cols,
        "numeric_features": numeric_feature_cols,
        "categorical_features": categorical_feature_cols,
        "feature_selection_strategy": "exclude_columns",
        "hp_search_strategy": "stochastic_holdout_search",
        "hp_search_iterations": stochastic_search_iterations,
        "hp_search_best_test_auc": best_search_auc,
        "hp_params": best_params,
    }
    pprint.pprint(model_artefact)

    file_path = os.path.join(model_bank_directory, model_version + ".pkl")
    with open(file_path, "wb") as file:
        pickle.dump(model_artefact, file)
    print(f"Model saved to {file_path}")

    log_path = os.path.join(model_bank_directory, "model_log.csv")
    new_row = {
        "model_version": model_version,
        "model_type": model_type,
        "train_date": date_str,
        "auc_train": train_auc_score,
        "auc_test": test_auc_score,
        "auc_oot": oot_auc_score,
        "gini_train": round(2 * train_auc_score - 1, 3),
        "gini_test": round(2 * test_auc_score - 1, 3),
        "gini_oot": round(2 * oot_auc_score - 1, 3),
        "brier_train": selected_calibration_results["train"]["brier"],
        "brier_test": selected_calibration_results["test"]["brier"],
        "brier_oot": selected_calibration_results["oot"]["brier"],
        "log_loss_train": selected_calibration_results["train"]["log_loss"],
        "log_loss_test": selected_calibration_results["test"]["log_loss"],
        "log_loss_oot": selected_calibration_results["oot"]["log_loss"],
        "calibration_method": selected_calibration_method,
        "calibration_selection_metric": "fixed_sigmoid",
        "prediction_threshold": selected_prediction_threshold,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "threshold_selection_score": selected_threshold_result[
            THRESHOLD_SELECTION_METRIC
        ],
        "oot_threshold_selection_score": oot_threshold_result[
            THRESHOLD_SELECTION_METRIC
        ],
        "champion": 0,
        "challenger": 0,
    }

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        champion_rows = log_df[log_df["champion"] == 1]
        if champion_rows.empty:
            # No champion exists — auto-promote new model.
            new_row["champion"] = 1
            print(f"No existing champion - new model auto-promoted to champion: {model_version}")
        else:
            champ = champion_rows.iloc[0]
            beats_test = test_auc_score > champ["auc_test"]
            beats_oot = oot_auc_score > champ["auc_oot"]
            if beats_test and beats_oot:
                # Better than champion — stage as challenger for human review.
                if "challenger" in log_df.columns:
                    log_df["challenger"] = 0
                new_row["challenger"] = 1
                print(
                    f"New challenger staged (beats champion on test and oot AUC): {model_version}"
                )
            else:
                print(
                    "New model does not beat champion on test and oot AUCs "
                    f"(test {beats_test}, oot {beats_oot}) - logged without challenger status"
                )
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_row["champion"] = 1
        log_df = pd.DataFrame([new_row])
        print(f"First model logged as champion: {model_version}")

    log_df = log_df.reindex(columns=LOG_COLUMNS)
    log_df.to_csv(log_path, index=False)
    print(f"Model log updated: {log_path}")
