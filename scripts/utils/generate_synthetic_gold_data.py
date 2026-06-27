"""Generate synthetic gold-layer monthly partitions from existing distributions."""

import argparse
import glob
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


base_directory = "/opt/airflow" if os.path.isdir("/opt/airflow") else "."
DEFAULT_GOLD_DIRECTORY = os.path.join(base_directory, "datamart/gold")
FEATURE_DATASET = "feature_store"
LABEL_DATASET = "label_store"

# Gaussian noise on sampled positive values can push them below zero.
# These columns are clipped to 0 after noise is applied.
NON_NEGATIVE_FEATURE_COLUMNS = {
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_Utilization_Ratio",
}
# Ratio columns are excluded from whole-row noise and instead recalculated
# from their source columns after sampling so they stay mathematically consistent
# (e.g. debt_to_income_ratio = Outstanding_Debt / Annual_Income).
RATIO_FEATURE_COLUMNS = {
    "debt_to_income_ratio",
    "emi_to_salary_ratio",
}
# 1st–99th percentile bounds prevent extreme synthetic ratios that would not
# appear in real data but can arise when source columns are noised independently.
DEFAULT_RATIO_LOWER_QUANTILE = 0.01
DEFAULT_RATIO_UPPER_QUANTILE = 0.99


def _month_start(value):
    return pd.Timestamp(value).replace(day=1).normalize()


def _parse_partition_date(path, dataset):
    pattern = rf"gold_{dataset}_(\d{{4}}_\d{{2}}_\d{{2}})\.parquet$"
    match = re.search(pattern, os.path.basename(path))
    if not match:
        return None
    return pd.Timestamp(datetime.strptime(match.group(1), "%Y_%m_%d"))


def _partition_path(directory, dataset, snapshot_date):
    date_clean = pd.Timestamp(snapshot_date).strftime("%Y_%m_%d")
    return os.path.join(directory, dataset, f"gold_{dataset}_{date_clean}.parquet")


def _partition_files(directory, dataset):
    dataset_directory = os.path.join(directory, dataset)
    return sorted(glob.glob(os.path.join(dataset_directory, f"gold_{dataset}_*.parquet")))


def _read_partitions(paths):
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No non-empty parquet partitions found to profile.")
    return pd.concat(frames, ignore_index=True)


def _profile_paths(directory, dataset, lookback_months, profile_end_date=None):
    paths = _partition_files(directory, dataset)
    dated_paths = [
        (path, _parse_partition_date(path, dataset))
        for path in paths
    ]
    dated_paths = [(path, date) for path, date in dated_paths if date is not None]
    if not dated_paths:
        raise FileNotFoundError(f"No gold {dataset} partitions found: {directory}")

    dated_paths = sorted(dated_paths, key=lambda item: item[1])
    if profile_end_date is not None:
        # When start_date is supplied we cap the profile at the month before it.
        # This prevents already-written synthetic partitions from contaminating
        # the reference distribution used to generate new ones.
        profile_end = _month_start(profile_end_date)
        dated_paths = [
            (path, date) for path, date in dated_paths
            if _month_start(date) <= profile_end
        ]
        if not dated_paths:
            raise FileNotFoundError(
                f"No gold {dataset} partitions found on or before {profile_end.date()}."
            )
    # Take only the most recent N partitions so the profile reflects current
    # distribution rather than early historical data that may have drifted.
    if lookback_months:
        dated_paths = dated_paths[-lookback_months:]
    return [path for path, _ in dated_paths], dated_paths[-1][1]


def _target_months(start_date, end_date):
    start = _month_start(start_date)
    end = _month_start(end_date)
    if start > end:
        return []
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _sample_row_count(paths, rng):
    # Sampling from actual partition sizes preserves natural month-to-month
    # volume variation instead of fixing an arbitrary constant row count.
    counts = [len(pd.read_parquet(path)) for path in paths]
    counts = [count for count in counts if count > 0]
    if not counts:
        raise ValueError("Profile partitions are empty.")
    return int(rng.choice(counts))


def _resample_series(series, row_count, rng, numeric_noise):
    values = series.dropna()
    # Preserve the observed null rate so synthetic columns have the same
    # missingness pattern as the reference data.
    missing_rate = float(series.isna().mean())
    if values.empty:
        if pd.api.types.is_numeric_dtype(series):
            return pd.Series([np.nan] * row_count, dtype=series.dtype)
        return pd.Series([pd.NA] * row_count, dtype=series.dtype)

    sampled = values.sample(
        n=row_count,
        replace=True,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)

    if pd.api.types.is_numeric_dtype(series):
        std = float(pd.to_numeric(values, errors="coerce").std())
        # Noise scale is proportional to the column's own std so the perturbation
        # is consistent across columns with very different magnitudes.
        if np.isfinite(std) and std > 0 and numeric_noise > 0:
            sampled = sampled.astype(float) + rng.normal(
                loc=0.0,
                scale=std * numeric_noise,
                size=row_count,
            )
        if pd.api.types.is_integer_dtype(series):
            sampled = sampled.round().astype(series.dtype)
        else:
            sampled = sampled.astype(series.dtype)

    if missing_rate > 0:
        missing_mask = rng.random(row_count) < missing_rate
        sampled = sampled.mask(missing_mask)

    return sampled


def _synthetic_customer_ids(source, row_count, rng):
    if "Customer_ID" not in source.columns:
        return None
    ids = source["Customer_ID"].dropna().astype(str)
    if ids.empty:
        return None
    return ids.sample(
        n=row_count,
        replace=True,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)


def _generate_frame(source, snapshot_date, row_count, rng, numeric_noise):
    frame = pd.DataFrame(index=range(row_count))
    customer_ids = _synthetic_customer_ids(source, row_count, rng)

    for column in source.columns:
        if column == "snapshot_date":
            frame[column] = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
        elif column == "Customer_ID" and customer_ids is not None:
            frame[column] = customer_ids
        else:
            frame[column] = _resample_series(
                source[column],
                row_count,
                rng,
                numeric_noise,
            )

    return frame[source.columns]


def _generate_feature_frame(source, snapshot_date, row_count, rng, numeric_noise):
    # Whole-row sampling preserves correlations between features (e.g. income and
    # debt levels from the same customer). Column-by-column sampling would break
    # these relationships and produce unrealistic feature combinations.
    frame = source.sample(
        n=row_count,
        replace=True,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True).copy()

    if "snapshot_date" in frame.columns:
        frame["snapshot_date"] = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")

    customer_ids = _synthetic_customer_ids(source, row_count, rng)
    if "Customer_ID" in frame.columns and customer_ids is not None:
        frame["Customer_ID"] = customer_ids

    for column in frame.columns:
        # Ratio columns are skipped here; they are recalculated from their source
        # columns in _recalculate_feature_fields to stay mathematically consistent.
        if column in {
            "snapshot_date",
            "Customer_ID",
            "debt_to_income_ratio",
            "emi_to_salary_ratio",
        }:
            continue
        if pd.api.types.is_numeric_dtype(source[column]):
            values = pd.to_numeric(source[column], errors="coerce").dropna()
            std = float(values.std()) if not values.empty else 0.0
            if np.isfinite(std) and std > 0 and numeric_noise > 0:
                noisy = pd.to_numeric(frame[column], errors="coerce").astype(float)
                noisy = noisy + rng.normal(
                    loc=0.0,
                    scale=std * numeric_noise,
                    size=row_count,
                )
                if pd.api.types.is_integer_dtype(source[column]):
                    frame[column] = noisy.round().astype("Int64")
                else:
                    frame[column] = noisy.astype(source[column].dtype)

    return frame[source.columns]


def _credit_history_age_to_months(series):
    values = series.astype("object").where(pd.notna(series), "").astype(str)
    years = values.str.extract(r"(\d+)\s+Years?", expand=False)
    months = values.str.extract(r"(\d+)\s+Months?", expand=False)
    has_value = values.str.strip().ne("")
    return np.where(
        has_value,
        pd.to_numeric(years, errors="coerce").fillna(0) * 12
        + pd.to_numeric(months, errors="coerce").fillna(0),
        np.nan,
    )


def _ratio_bounds(source, lower_quantile, upper_quantile):
    # Bounds are derived from the reference profile, not from each synthetic month,
    # so the cap is stable across runs and anchored to observed real data ranges.
    bounds = {}
    for column in RATIO_FEATURE_COLUMNS:
        if column not in source.columns:
            continue
        values = pd.to_numeric(source[column], errors="coerce").dropna()
        if values.empty:
            continue
        lower = float(values.quantile(lower_quantile))
        upper = float(values.quantile(upper_quantile))
        if np.isfinite(lower) and np.isfinite(upper) and lower <= upper:
            bounds[column] = (lower, upper)
    return bounds


def _recalculate_feature_fields(frame, ratio_bounds=None):
    # After whole-row sampling and noise, financial fields may be negative and
    # derived ratio fields may be inconsistent with their source columns. This
    # function restores internal consistency before the partition is written.
    frame = frame.copy()
    ratio_bounds = ratio_bounds or {}

    # Financial amounts cannot be negative; clip after noise is applied.
    for column in NON_NEGATIVE_FEATURE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower=0)

    if {"Outstanding_Debt", "Annual_Income", "debt_to_income_ratio"}.issubset(frame.columns):
        annual_income = pd.to_numeric(frame["Annual_Income"], errors="coerce")
        outstanding_debt = pd.to_numeric(frame["Outstanding_Debt"], errors="coerce")
        frame["debt_to_income_ratio"] = np.where(
            (annual_income > 0) & (outstanding_debt >= 0),
            outstanding_debt / annual_income,
            np.nan,
        )
        if "debt_to_income_ratio" in ratio_bounds:
            lower, upper = ratio_bounds["debt_to_income_ratio"]
            frame["debt_to_income_ratio"] = pd.to_numeric(
                frame["debt_to_income_ratio"],
                errors="coerce",
            ).clip(lower=lower, upper=upper)

    if {
        "Total_EMI_per_month",
        "Monthly_Inhand_Salary",
        "emi_to_salary_ratio",
    }.issubset(frame.columns):
        monthly_salary = pd.to_numeric(frame["Monthly_Inhand_Salary"], errors="coerce")
        total_emi = pd.to_numeric(frame["Total_EMI_per_month"], errors="coerce")
        frame["emi_to_salary_ratio"] = np.where(
            (monthly_salary > 0) & (total_emi >= 0),
            total_emi / monthly_salary,
            np.nan,
        )
        if "emi_to_salary_ratio" in ratio_bounds:
            lower, upper = ratio_bounds["emi_to_salary_ratio"]
            frame["emi_to_salary_ratio"] = pd.to_numeric(
                frame["emi_to_salary_ratio"],
                errors="coerce",
            ).clip(lower=lower, upper=upper)

    if {"Credit_Utilization_Ratio", "high_credit_utilization_flag"}.issubset(frame.columns):
        utilization = pd.to_numeric(frame["Credit_Utilization_Ratio"], errors="coerce")
        frame["high_credit_utilization_flag"] = np.where(utilization > 50, 1, 0)

    if {"Credit_History_Age", "Credit_History_Age_Months"}.issubset(frame.columns):
        frame["Credit_History_Age_Months"] = _credit_history_age_to_months(
            frame["Credit_History_Age"]
        )

    return frame


def _generate_label_frame(source, snapshot_date, row_count, rng, numeric_noise):
    # Label columns do not need numeric noise; the only meaningful variation is
    # in the binary label itself, so noise=0.0 is passed to _generate_frame.
    frame = _generate_frame(source, snapshot_date, row_count, rng, numeric_noise=0.0)
    if "label" in frame.columns:
        # Draw labels from a Bernoulli at the reference default rate rather than
        # resampling rows. Row resampling would bias the rate toward whichever
        # reference partitions happened to have more defaults.
        label_rate = float(pd.to_numeric(source["label"], errors="coerce").mean())
        frame["label"] = (rng.random(row_count) < label_rate).astype(int)
    if "label_def" in frame.columns:
        frame["label_def"] = source["label_def"].dropna().mode().iloc[0]
    return frame


def _write_parquet_partition(frame, path, overwrite):
    if os.path.exists(path):
        if not overwrite:
            print(f"skipped existing partition: {path}")
            return False
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write to a temp file in the same directory then atomically rename so a
    # partial write from an interrupted process never leaves a corrupt parquet.
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp.parquet",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print(f"wrote {len(frame)} row(s): {path}")
    return True


def generate_synthetic_dataset(
    gold_directory,
    output_gold_directory,
    dataset,
    start_date,
    end_date,
    lookback_months,
    seed,
    overwrite,
    numeric_noise,
    ratio_lower_quantile,
    ratio_upper_quantile,
):
    # When regenerating from a specific start_date, cap the profile at the month
    # before it so already-written synthetic partitions are not included in the
    # reference distribution used to generate the new ones.
    profile_end_date = None
    if start_date is not None:
        profile_end_date = _month_start(start_date) - pd.DateOffset(months=1)

    profile_paths, latest_date = _profile_paths(
        gold_directory,
        dataset,
        lookback_months,
        profile_end_date=profile_end_date,
    )
    profile = _read_partitions(profile_paths)
    # Ratio bounds are only relevant for the feature store; the label store does
    # not contain ratio columns.
    feature_ratio_bounds = {}
    if dataset == FEATURE_DATASET:
        feature_ratio_bounds = _ratio_bounds(
            profile,
            ratio_lower_quantile,
            ratio_upper_quantile,
        )

    # Gold partitions are monthly, so "until yesterday" means up to yesterday's
    # month-start partition.
    first_target_month = (
        latest_date + pd.DateOffset(months=1)
        if start_date is None
        else _month_start(start_date)
    )
    months = _target_months(first_target_month, end_date)
    if not months:
        print(f"{dataset}: no missing monthly partitions to generate.")
        return []

    written = []
    for snapshot_date in months:
        # Per-month seed so each monthly Airflow run produces distinct data.
        month_seed = seed + snapshot_date.year * 12 + snapshot_date.month
        rng = np.random.default_rng(month_seed)
        row_count = _sample_row_count(profile_paths, rng)
        if dataset == LABEL_DATASET:
            frame = _generate_label_frame(
                profile,
                snapshot_date,
                row_count,
                rng,
                numeric_noise,
            )
        else:
            frame = _generate_feature_frame(
                profile,
                snapshot_date,
                row_count,
                rng,
                numeric_noise,
            )
            frame = _recalculate_feature_fields(
                frame,
                ratio_bounds=feature_ratio_bounds,
            )
        path = _partition_path(output_gold_directory, dataset, snapshot_date)
        if _write_parquet_partition(frame, path, overwrite):
            written.append(path)
    return written


def main(
    gold_directory=DEFAULT_GOLD_DIRECTORY,
    output_gold_directory=None,
    start_date=None,
    end_date=None,
    datasets=None,
    lookback_months=12,
    seed=88,
    overwrite=False,
    numeric_noise=0.02,
    ratio_lower_quantile=DEFAULT_RATIO_LOWER_QUANTILE,
    ratio_upper_quantile=DEFAULT_RATIO_UPPER_QUANTILE,
):
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    if datasets is None:
        datasets = [FEATURE_DATASET, LABEL_DATASET]
    if output_gold_directory is None:
        output_gold_directory = gold_directory

    all_written = []
    for index, dataset in enumerate(datasets):
        # Offset seed per dataset so feature_store and label_store draw
        # independent row counts and do not produce identical volume patterns.
        written = generate_synthetic_dataset(
            gold_directory=gold_directory,
            output_gold_directory=output_gold_directory,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            lookback_months=lookback_months,
            seed=seed + index,
            overwrite=overwrite,
            numeric_noise=numeric_noise,
            ratio_lower_quantile=ratio_lower_quantile,
            ratio_upper_quantile=ratio_upper_quantile,
        )
        all_written.extend(written)
    return all_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic monthly gold-layer feature/label partitions "
            "from existing gold distribution."
        )
    )
    parser.add_argument("--gold-directory", default=DEFAULT_GOLD_DIRECTORY)
    parser.add_argument(
        "--output-gold-directory",
        default=None,
        help="Where synthetic partitions are written; defaults to --gold-directory.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "First monthly partition to generate. When supplied with --overwrite, "
            "existing partitions from this month onward can be regenerated."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last date to cover; defaults to yesterday. Monthly partitions use month start.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[FEATURE_DATASET, LABEL_DATASET],
        default=[FEATURE_DATASET, LABEL_DATASET],
    )
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--numeric-noise",
        type=float,
        default=0.02,
        help="Std-dev fraction used as noise for numeric feature columns.",
    )
    parser.add_argument(
        "--ratio-lower-quantile",
        type=float,
        default=DEFAULT_RATIO_LOWER_QUANTILE,
        help="Lower profile quantile used to cap synthetic ratio features.",
    )
    parser.add_argument(
        "--ratio-upper-quantile",
        type=float,
        default=DEFAULT_RATIO_UPPER_QUANTILE,
        help="Upper profile quantile used to cap synthetic ratio features.",
    )
    args = parser.parse_args()

    main(
        gold_directory=args.gold_directory,
        output_gold_directory=args.output_gold_directory,
        start_date=args.start_date,
        end_date=args.end_date,
        datasets=args.datasets,
        lookback_months=args.lookback_months,
        seed=args.seed,
        overwrite=args.overwrite,
        numeric_noise=args.numeric_noise,
        ratio_lower_quantile=args.ratio_lower_quantile,
        ratio_upper_quantile=args.ratio_upper_quantile,
    )
