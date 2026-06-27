"""Utilities for keeping model_log.csv in sync with model artifacts."""

import argparse
import glob
import os
import pickle
import re
import tempfile

import pandas as pd


LOG_COLUMNS = [
    "model_version",
    "train_date",
    "auc_train",
    "auc_test",
    "auc_oot",
    "gini_train",
    "gini_test",
    "gini_oot",
    "brier_train",
    "brier_test",
    "brier_oot",
    "log_loss_train",
    "log_loss_test",
    "log_loss_oot",
    "calibration_method",
    "calibration_selection_metric",
    "prediction_threshold",
    "threshold_selection_metric",
    "threshold_selection_score",
    "oot_threshold_selection_score",
    "champion",
    "challenger",
]


def ensure_directory(path):
    """Create path as a directory and fail clearly on invalid paths."""
    path = os.path.abspath(path)
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        pass

    if not os.path.isdir(path):
        path_type = "missing"
        if os.path.islink(path):
            path_type = "symlink"
        elif os.path.isfile(path):
            path_type = "file"
        elif os.path.exists(path):
            path_type = "non-directory"
        raise NotADirectoryError(
            f"Expected directory but found {path_type}: {path}. "
            "If this is /opt/airflow/model_bank, ensure the host ./model_bank "
            "bind-mount source exists as a directory, then restart Airflow "
            "containers so the mount is refreshed."
        )

    return path


def _train_date(artifact, model_version):
    data_dates = artifact.get("data_dates", {})
    value = data_dates.get("model_train_date_str")
    if value:
        return str(value)

    match = re.search(r"credit_model_(\d{4})_(\d{2})_(\d{2})", model_version)
    if match:
        return "-".join(match.groups())
    return ""


def _row_from_artifact(path):
    with open(path, "rb") as file:
        artifact = pickle.load(file)

    if not isinstance(artifact, dict):
        raise ValueError("artifact is not a dictionary")

    filename_version = os.path.splitext(os.path.basename(path))[0]
    model_version = str(artifact.get("model_version", "")).strip()
    if not model_version:
        raise ValueError("artifact has no model_version")
    if model_version != filename_version:
        raise ValueError(
            f"artifact model_version '{model_version}' does not match filename"
        )

    results = artifact.get("results", {})
    required_results = ("auc_train", "auc_test", "auc_oot")
    missing = [name for name in required_results if name not in results]
    if missing:
        raise ValueError(f"artifact results missing: {', '.join(missing)}")

    auc_train = float(results["auc_train"])
    auc_test = float(results["auc_test"])
    auc_oot = float(results["auc_oot"])

    return {
        "model_version": model_version,
        "train_date": _train_date(artifact, model_version),
        "auc_train": auc_train,
        "auc_test": auc_test,
        "auc_oot": auc_oot,
        "gini_train": float(results.get("gini_train", round(2 * auc_train - 1, 3))),
        "gini_test": float(results.get("gini_test", round(2 * auc_test - 1, 3))),
        "gini_oot": float(results.get("gini_oot", round(2 * auc_oot - 1, 3))),
        "brier_train": results.get("brier_train", ""),
        "brier_test": results.get("brier_test", ""),
        "brier_oot": results.get("brier_oot", ""),
        "log_loss_train": results.get("log_loss_train", ""),
        "log_loss_test": results.get("log_loss_test", ""),
        "log_loss_oot": results.get("log_loss_oot", ""),
        "calibration_method": artifact.get("calibration_method", "uncalibrated"),
        "calibration_selection_metric": artifact.get("calibration_selection_metric", ""),
        "prediction_threshold": artifact.get("prediction_threshold", ""),
        "threshold_selection_metric": artifact.get("threshold_selection_metric", ""),
        "threshold_selection_score": (
            artifact.get("threshold_selection_results", {})
            .get("test_selected", {})
            .get(artifact.get("threshold_selection_metric", ""), "")
        ),
        "oot_threshold_selection_score": (
            artifact.get("threshold_selection_results", {})
            .get("oot_at_selected_threshold", {})
            .get(artifact.get("threshold_selection_metric", ""), "")
        ),
        "champion": 0,
        "challenger": 0,
    }


def reconcile_model_log(model_bank_directory="/opt/airflow/model_bank"):
    """Synchronize model_log.csv with valid pickle artifacts."""
    model_bank_directory = ensure_directory(model_bank_directory)
    log_path = os.path.join(model_bank_directory, "model_log.csv")

    old_log = pd.DataFrame(columns=LOG_COLUMNS)
    if os.path.exists(log_path):
        try:
            old_log = pd.read_csv(log_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            old_log = pd.DataFrame(columns=LOG_COLUMNS)

    old_versions = set()
    old_champions = []
    if "model_version" in old_log.columns:
        old_versions = set(old_log["model_version"].dropna().astype(str).str.strip())
    if {"model_version", "champion"}.issubset(old_log.columns):
        champion_values = pd.to_numeric(old_log["champion"], errors="coerce").fillna(0)
        old_champions = (
            old_log.loc[champion_values.eq(1), "model_version"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    rows = []
    invalid_artifacts = {}
    for path in sorted(glob.glob(os.path.join(model_bank_directory, "*.pkl"))):
        try:
            rows.append(_row_from_artifact(path))
        except Exception as exc:
            invalid_artifacts[os.path.basename(path)] = str(exc)

    reconciled = pd.DataFrame(rows, columns=LOG_COLUMNS)
    valid_versions = set(reconciled["model_version"]) if not reconciled.empty else set()
    valid_old_champions = [
        version for version in dict.fromkeys(old_champions)
        if version in valid_versions
    ]

    selected_champion = None
    if len(valid_old_champions) == 1:
        # Preserve the existing champion.
        selected_champion = valid_old_champions[0]
    elif not reconciled.empty:
        # No champion exists — auto-promote the best model.
        selected_champion = (
            reconciled.sort_values(
                ["auc_oot", "auc_test", "auc_train", "model_version"],
                ascending=[False, False, False, True],
            )
            .iloc[0]["model_version"]
        )

    reconciled["champion"] = 0
    reconciled["challenger"] = 0
    if selected_champion:
        reconciled["champion"] = (
            reconciled["model_version"].eq(selected_champion).astype(int)
        )

    # Challenger: best non-champion by auc_oot; requires human to promote to champion.
    non_champion = reconciled[reconciled["champion"] != 1]
    if not non_champion.empty:
        selected_challenger = (
            non_champion.sort_values(
                ["auc_oot", "auc_test", "auc_train", "model_version"],
                ascending=[False, False, False, True],
            )
            .iloc[0]["model_version"]
        )
        reconciled["challenger"] = (
            reconciled["model_version"].eq(selected_challenger).astype(int)
        )

    reconciled = reconciled.sort_values("model_version").reset_index(drop=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".model_log.", suffix=".csv", dir=model_bank_directory
    )
    os.close(fd)
    try:
        reconciled.to_csv(temporary_path, index=False)
        os.replace(temporary_path, log_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    selected_challenger = None
    challenger_rows = reconciled[reconciled["challenger"] == 1]
    if not challenger_rows.empty:
        selected_challenger = challenger_rows.iloc[0]["model_version"]

    summary = {
        "log_path": log_path,
        "valid_models": len(reconciled),
        "added": sorted(valid_versions - old_versions),
        "removed": sorted(old_versions - valid_versions),
        "invalid_artifacts": invalid_artifacts,
        "champion": selected_champion,
        "challenger": selected_challenger,
    }
    print(
        "model log reconciled:",
        f"{summary['valid_models']} valid model(s),",
        f"champion={summary['champion']},",
        f"challenger={summary['challenger']}",
    )
    if summary["added"]:
        print("added:", ", ".join(summary["added"]))
    if summary["removed"]:
        print("removed:", ", ".join(summary["removed"]))
    for filename, reason in invalid_artifacts.items():
        print(f"skipped invalid artifact {filename}: {reason}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="synchronize model_log.csv with model pickle artifacts"
    )
    parser.add_argument(
        "--model-bank",
        default="/opt/airflow/model_bank",
        help="directory containing model_log.csv and model .pkl files",
    )
    args = parser.parse_args()
    reconcile_model_log(args.model_bank)
