import os
import glob
import re
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


sys.path.insert(0, "/opt/airflow")

FIRST_TRAINING_DATE = "2024-09-01"


def dag_option(context, key, default=None):
    """Read manual trigger config first, then DAG params, normalizing blanks."""
    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf and key in dag_run.conf:
        value = dag_run.conf.get(key)
    else:
        value = (context.get("params") or {}).get(key, default)

    if value == "":
        return default
    return value


def dag_option_bool(context, key, default=False):
    value = dag_option(context, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def dag_option_int(context, key, default):
    value = dag_option(context, key, default)
    if value is None:
        return default
    return int(value)


def dag_option_float(context, key, default):
    value = dag_option(context, key, default)
    if value is None:
        return default
    return float(value)


def latest_gold_snapshot_date(gold_store_directory, dataset):
    dates = gold_snapshot_dates(gold_store_directory, dataset)
    if not dates:
        raise FileNotFoundError(
            f"No gold {dataset} partitions found: {gold_store_directory}"
        )
    return dates[-1]


def gold_snapshot_dates(
    gold_store_directory,
    dataset,
    min_snapshotdate=None,
    max_snapshotdate=None,
):
    paths = glob.glob(
        os.path.join(gold_store_directory, f"gold_{dataset}_*.parquet")
    )
    dates = []
    for path in paths:
        match = re.search(
            rf"gold_{dataset}_(\d{{4}}_\d{{2}}_\d{{2}})\.parquet$",
            os.path.basename(path),
        )
        if match:
            dates.append(datetime.strptime(match.group(1), "%Y_%m_%d").strftime("%Y-%m-%d"))

    dates = sorted(dates)
    if min_snapshotdate:
        dates = [date for date in dates if date >= min_snapshotdate]
    if max_snapshotdate:
        dates = [date for date in dates if date <= max_snapshotdate]
    return dates


def generate_synthetic_gold_data(**context):
    import scripts.utils.generate_synthetic_gold_data as synthetic_gold

    # DAG params allow the same simulation DAG to generate different horizons
    # without editing code.
    start_date = dag_option(context, "synthetic_start_date")
    end_date = dag_option(context, "synthetic_end_date")
    lookback_months = dag_option_int(context, "synthetic_lookback_months", 12)
    seed = dag_option_int(context, "synthetic_seed", 88)
    overwrite = dag_option_bool(context, "synthetic_overwrite", False)
    numeric_noise = dag_option_float(context, "synthetic_numeric_noise", 0.02)
    ratio_lower_quantile = dag_option_float(
        context,
        "synthetic_ratio_lower_quantile",
        0.01,
    )
    ratio_upper_quantile = dag_option_float(
        context,
        "synthetic_ratio_upper_quantile",
        0.99,
    )

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        written = synthetic_gold.main(
            gold_directory="datamart/gold",
            output_gold_directory=None,
            start_date=start_date,
            end_date=end_date,
            datasets=["feature_store", "label_store"],
            lookback_months=lookback_months,
            seed=seed,
            overwrite=overwrite,
            numeric_noise=numeric_noise,
            ratio_lower_quantile=ratio_lower_quantile,
            ratio_upper_quantile=ratio_upper_quantile,
        )
        print("Synthetic gold generation wrote:", written)
        return {
            "written_count": len(written),
            "written_paths": written,
        }
    finally:
        os.chdir(original_directory)


def check_simulation_gold_partitions(**context):
    import scripts.ML_model.model_inference as model_inference

    model_name = dag_option(context, "model_name")
    max_snapshotdate = dag_option(context, "max_snapshotdate")
    if dag_option_bool(context, "infer_through_latest_gold", True):
        max_snapshotdate = None

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        try:
            selected_model_name = model_inference.select_model_name(model_name, model_type="xgboost")
        except (FileNotFoundError, ValueError) as error:
            # Continue to the inference task so it can bootstrap or reconcile a
            # champion model using the same logic as production inference.
            print(
                "No existing champion model available during simulation check. "
                "Continuing to inference so bootstrap training can run if possible. "
                f"Original issue: {error}"
            )
            return ["run_simulation_xgboost_inference", "run_simulation_log_reg_inference"]

        model_version = os.path.splitext(selected_model_name)[0]
        snapshot_dates = model_inference.unpredicted_gold_snapshot_dates(
            model_version=model_version,
            min_snapshotdate=FIRST_TRAINING_DATE,
            max_snapshotdate=max_snapshotdate,
        )

        if snapshot_dates:
            print("Synthetic gold partitions requiring inference:", snapshot_dates)
            return ["run_simulation_xgboost_inference", "run_simulation_log_reg_inference"]

        print("No synthetic gold partitions require inference.")
        return ["skip_simulation_xgboost_inference", "skip_simulation_log_reg_inference"]
    finally:
        os.chdir(original_directory)


def run_simulation_xgboost_inference(**context):
    import scripts.ML_model.model_inference as model_inference

    model_name = dag_option(context, "model_name")
    max_snapshotdate = dag_option(context, "max_snapshotdate")
    if dag_option_bool(context, "infer_through_latest_gold", True):
        max_snapshotdate = None

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        result = model_inference.run_new_gold_predictions(
            modelname=model_name,
            min_snapshotdate=FIRST_TRAINING_DATE,
            max_snapshotdate=max_snapshotdate,
            model_type="xgboost",
        )
        print("Simulation xgboost inference result:", result)
        return result
    finally:
        os.chdir(original_directory)


def run_simulation_log_reg_inference(**context):
    import scripts.ML_model.model_inference as model_inference

    model_name = dag_option(context, "model_name")
    max_snapshotdate = dag_option(context, "max_snapshotdate")
    if dag_option_bool(context, "infer_through_latest_gold", True):
        max_snapshotdate = None

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        result = model_inference.run_new_gold_predictions(
            modelname=model_name,
            min_snapshotdate=FIRST_TRAINING_DATE,
            max_snapshotdate=max_snapshotdate,
            model_type="log_reg",
        )
        print("Simulation log_reg inference result:", result)
        return result
    finally:
        os.chdir(original_directory)


def _run_simulation_monitoring(context, model_type):
    import scripts.ML_model.model_monitoring as model_monitoring

    if not dag_option_bool(context, "run_monitoring", True):
        raise AirflowSkipException("Simulation monitoring disabled by DAG config.")

    model_name = dag_option(context, "model_name")
    monitoring_date = dag_option(context, "monitoring_snapshotdate")
    max_snapshotdate = dag_option(context, "max_snapshotdate")
    synthetic_end_date = dag_option(context, "synthetic_end_date")
    monitor_all_snapshotdates = dag_option_bool(
        context,
        "monitor_all_snapshotdates",
        True,
    )

    if monitoring_date is None and not monitor_all_snapshotdates:
        if dag_option_bool(context, "infer_through_latest_gold", True):
            monitoring_date = synthetic_end_date
        else:
            monitoring_date = max_snapshotdate

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        if monitoring_date is not None:
            snapshot_dates = [monitoring_date]
        elif monitor_all_snapshotdates:
            max_monitoring_date = None
            if not dag_option_bool(context, "infer_through_latest_gold", True):
                max_monitoring_date = max_snapshotdate
            elif synthetic_end_date is not None:
                max_monitoring_date = synthetic_end_date

            snapshot_dates = gold_snapshot_dates(
                "datamart/gold/feature_store",
                dataset="feature_store",
                min_snapshotdate=FIRST_TRAINING_DATE,
                max_snapshotdate=max_monitoring_date,
            )
        else:
            snapshot_dates = [
                latest_gold_snapshot_date(
                    "datamart/gold/feature_store",
                    dataset="feature_store",
                )
            ]

        if not snapshot_dates:
            raise AirflowSkipException("No gold feature snapshots available for monitoring.")

        results = []
        for snapshot_date in snapshot_dates:
            result = model_monitoring.main(
                snapshotdate=snapshot_date,
                modelname=model_name,
                model_type=model_type,
            )
            results.append(result)

        print(f"Simulation {model_type} monitoring results:", results)
        return {
            "monitoring_count": len(results),
            "snapshot_dates": snapshot_dates,
            "results": results,
        }
    finally:
        os.chdir(original_directory)


def run_simulation_xgboost_monitoring(**context):
    return _run_simulation_monitoring(context, "xgboost")


def run_simulation_log_reg_monitoring(**context):
    return _run_simulation_monitoring(context, "log_reg")


def _evaluate_simulation_predictions(context, model_type):
    import scripts.ML_model.model_performance as model_performance

    if not dag_option_bool(context, "evaluate_performance", True):
        raise AirflowSkipException("Performance evaluation disabled by DAG config.")

    model_name = dag_option(context, "model_name")
    evaluation_date = dag_option(context, "performance_evaluation_date")

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        if evaluation_date is None:
            # Default to the latest synthetic/real label partition so a manual
            # run can evaluate as much as has matured.
            evaluation_date = latest_gold_snapshot_date(
                "datamart/gold/label_store",
                dataset="label_store",
            )
        result = model_performance.main(
            evaluationdate=evaluation_date,
            modelname=model_name,
            model_type=model_type,
        )
        print(f"Simulation {model_type} performance evaluation result:", result)
        return result
    except ValueError as error:
        # A simulation may create future predictions before enough labels have
        # matured. Skip cleanly instead of failing the whole simulation DAG.
        raise AirflowSkipException(str(error))
    finally:
        os.chdir(original_directory)


def evaluate_simulation_xgboost_predictions(**context):
    return _evaluate_simulation_predictions(context, "xgboost")


def evaluate_simulation_log_reg_predictions(**context):
    return _evaluate_simulation_predictions(context, "log_reg")


default_args = {
    "owner": "edward",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "simulation_inference_dag",
    default_args=default_args,
    description="Manual simulation DAG that generates synthetic gold data and runs inference.",
    schedule_interval=None,
    start_date=datetime(2024, 9, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "model_name": "",
        "max_snapshotdate": "",
        "infer_through_latest_gold": True,
        "synthetic_end_date": "",
        "synthetic_start_date": "",
        "synthetic_lookback_months": 12,
        "synthetic_seed": 88,
        "synthetic_overwrite": False,
        "synthetic_numeric_noise": 0.02,
        "synthetic_ratio_lower_quantile": 0.01,
        "synthetic_ratio_upper_quantile": 0.99,
        "run_monitoring": True,
        "monitoring_snapshotdate": "",
        "monitor_all_snapshotdates": True,
        "evaluate_performance": True,
        "performance_evaluation_date": "",
    },
    tags=["simulation", "model_inference", "gold_layer"],
) as dag:
    generate_synthetic_gold = PythonOperator(
        task_id="generate_synthetic_gold_data",
        python_callable=generate_synthetic_gold_data,
    )

    check_simulation_gold = BranchPythonOperator(
        task_id="check_simulation_gold_partitions",
        python_callable=check_simulation_gold_partitions,
    )

    run_xgboost_inference = PythonOperator(
        task_id="run_simulation_xgboost_inference",
        python_callable=run_simulation_xgboost_inference,
    )

    run_log_reg_inference = PythonOperator(
        task_id="run_simulation_log_reg_inference",
        python_callable=run_simulation_log_reg_inference,
    )

    skip_xgboost_inference = DummyOperator(
        task_id="skip_simulation_xgboost_inference",
    )

    skip_log_reg_inference = DummyOperator(
        task_id="skip_simulation_log_reg_inference",
    )

    simulation_inference_completed = DummyOperator(
        task_id="simulation_inference_completed",
        trigger_rule="none_failed_min_one_success",
    )

    simulation_monitor_start = DummyOperator(
        task_id="simulation_monitor_start",
    )

    run_xgboost_monitoring = PythonOperator(
        task_id="run_simulation_xgboost_monitoring",
        python_callable=run_simulation_xgboost_monitoring,
    )

    run_log_reg_monitoring = PythonOperator(
        task_id="run_simulation_log_reg_monitoring",
        python_callable=run_simulation_log_reg_monitoring,
    )

    simulation_monitor_completed = DummyOperator(
        task_id="simulation_monitor_completed",
        trigger_rule="none_failed_min_one_success",
    )

    simulation_evaluation_start = DummyOperator(
        task_id="simulation_evaluation_start",
    )

    evaluate_xgboost_predictions = PythonOperator(
        task_id="evaluate_simulation_xgboost_predictions",
        python_callable=evaluate_simulation_xgboost_predictions,
    )

    evaluate_log_reg_predictions = PythonOperator(
        task_id="evaluate_simulation_log_reg_predictions",
        python_callable=evaluate_simulation_log_reg_predictions,
    )

    simulation_evaluation_completed = DummyOperator(
        task_id="simulation_evaluation_completed",
        trigger_rule="none_failed_min_one_success",
    )

    generate_synthetic_gold >> check_simulation_gold
    check_simulation_gold >> run_xgboost_inference >> simulation_inference_completed
    check_simulation_gold >> run_log_reg_inference >> simulation_inference_completed
    check_simulation_gold >> skip_xgboost_inference >> simulation_inference_completed
    check_simulation_gold >> skip_log_reg_inference >> simulation_inference_completed
    simulation_inference_completed >> simulation_monitor_start
    simulation_monitor_start >> run_xgboost_monitoring >> simulation_monitor_completed
    simulation_monitor_start >> run_log_reg_monitoring >> simulation_monitor_completed
    simulation_monitor_completed >> simulation_evaluation_start
    simulation_evaluation_start >> evaluate_xgboost_predictions >> simulation_evaluation_completed
    simulation_evaluation_start >> evaluate_log_reg_predictions >> simulation_evaluation_completed
