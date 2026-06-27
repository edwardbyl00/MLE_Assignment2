# http://localhost:8080 
# docker compose up airflow-webserver airflow-scheduler


import os
import sys
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from datetime import datetime, timedelta

# make utils importable inside the container
sys.path.insert(0, '/opt/airflow')

import scripts.utils.data_processing_bronze_table as bronze
import scripts.utils.data_processing_silver_table as silver
import scripts.utils.data_processing_gold_table as gold

# Directory path setup. These are container paths because Airflow runs inside
# Docker with repo folders mounted under /opt/airflow.
bronze_lms_directory        = "/opt/airflow/datamart/bronze/lms/"
bronze_feature_directory    = "/opt/airflow/datamart/bronze/features/"
silver_loan_daily_directory = "/opt/airflow/datamart/silver/loan_daily/"
silver_feature_directory    = "/opt/airflow/datamart/silver/features/"
gold_label_store_directory  = "/opt/airflow/datamart/gold/label_store/"
gold_feature_store_directory = "/opt/airflow/datamart/gold/feature_store/"

BACKFILL_START_DATE = "2023-01-01"
BACKFILL_END_DATE = "2024-12-01"
FIRST_MATURED_LABEL_DATE = "2023-07-01"
FIRST_TRAINING_DATE = "2024-09-01"
TRAINING_INTERVAL_MONTHS = 3


def previous_month_start(date_str):
    date_value = datetime.strptime(date_str, "%Y-%m-%d")
    if date_value.month == 1:
        previous_month = datetime(date_value.year - 1, 12, 1)
    else:
        previous_month = datetime(date_value.year, date_value.month - 1, 1)
    return previous_month.strftime("%Y-%m-%d")


def month_difference(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return (
        (end_date.year - start_date.year) * 12
        + end_date.month
        - start_date.month
    )


all_dir = [
    bronze_lms_directory, bronze_feature_directory,
    silver_loan_daily_directory, silver_feature_directory,
    gold_label_store_directory, gold_feature_store_directory,
]


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


def get_spark():
    import pyspark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("data_pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def make_dirs():
    for d in all_dir:
        os.makedirs(d, exist_ok=True)
    print("Directories ready.")


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return dates


def validate_csv_partition(filepath, spark):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected CSV partition not found: {filepath}")

    row_count = spark.read.csv(filepath, header=True, inferSchema=True).count()
    if row_count <= 0:
        raise ValueError(f"CSV partition is empty: {filepath}")

    print(f"validated CSV partition: {filepath}, row_count={row_count}")


def validate_parquet_partition(filepath, spark, allow_empty=False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected parquet partition not found: {filepath}")

    row_count = spark.read.parquet(filepath).count()
    if row_count <= 0 and not allow_empty:
        raise ValueError(f"Parquet partition is empty: {filepath}")

    print(f"validated parquet partition: {filepath}, row_count={row_count}")


def task_validate_label_medallion(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    date_clean = date_str.replace("-", "_")
    spark = get_spark()
    try:
        validate_csv_partition(
            f"{bronze_lms_directory}bronze_loan_daily_{date_clean}.csv",
            spark,
        )
        validate_parquet_partition(
            f"{silver_loan_daily_directory}silver_loan_daily_{date_clean}.parquet",
            spark,
        )
        validate_parquet_partition(
            f"{gold_label_store_directory}gold_label_store_{date_clean}.parquet",
            spark,
            allow_empty=date_str < FIRST_MATURED_LABEL_DATE,
        )
    finally:
        spark.stop()


def task_validate_feature_medallion(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    date_clean = date_str.replace("-", "_")
    feature_names = ["clickstream", "attributes", "financials"]
    spark = get_spark()
    try:
        for feature_name in feature_names:
            validate_csv_partition(
                f"{bronze_feature_directory}bronze_{feature_name}_{date_clean}.csv",
                spark,
            )
            validate_parquet_partition(
                f"{silver_feature_directory}silver_{feature_name}_{date_clean}.parquet",
                spark,
            )

        validate_parquet_partition(
            f"{gold_feature_store_directory}gold_feature_store_{date_clean}.parquet",
            spark,
        )
    finally:
        spark.stop()


def validate_gold_medallion_range(start_date, end_date):
    dates = generate_first_of_month_dates(start_date, end_date)
    spark = get_spark()
    try:
        for snapshot_date in dates:
            date_clean = snapshot_date.replace("-", "_")
            validate_parquet_partition(
                f"{gold_label_store_directory}gold_label_store_{date_clean}.parquet",
                spark,
                allow_empty=snapshot_date < FIRST_MATURED_LABEL_DATE,
            )
            validate_parquet_partition(
                f"{gold_feature_store_directory}gold_feature_store_{date_clean}.parquet",
                spark,
            )
    finally:
        spark.stop()

    print(
        "validated gold medallion range:",
        f"{len(dates)} monthly gold label and feature partitions",
        f"from {start_date} to {end_date}",
    )


# task callables

def task_bronze_lms(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        bronze.process_bronze_table(date_str, bronze_lms_directory, spark)
    finally:
        spark.stop()


def task_bronze_features(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        bronze.process_bronze_features(date_str, bronze_feature_directory, spark)
    finally:
        spark.stop()


def task_silver_lms(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        silver.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    finally:
        spark.stop()


def task_silver_features(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        silver.process_silver_features(date_str, bronze_feature_directory, silver_feature_directory, spark)
    finally:
        spark.stop()


def task_gold_labels(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        gold.process_labels_gold_table(
            date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6
        )
    finally:
        spark.stop()


def task_gold_features(**context):
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    spark = get_spark()
    try:
        gold.process_gold_feature_store(date_str, silver_feature_directory, gold_feature_store_directory, spark)
    finally:
        spark.stop()


def task_xgboost_train(**context):
    import scripts.ML_model.model_train as model_train

    date_str = context["execution_date"].strftime("%Y-%m-%d")

    # Training uses 12 train/test months plus 2 OOT months. The gold label store
    # starts in July 2023, so September 2024 is the first eligible training run.
    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping training for {date_str}: insufficient matured label history. "
            f"Earliest eligible training date is {FIRST_TRAINING_DATE}."
        )

    training_data_end_date = previous_month_start(date_str)
    validate_gold_medallion_range(BACKFILL_START_DATE, training_data_end_date)

    spark = get_spark()
    try:
        model_train.train_model(date_str, spark)
    finally:
        spark.stop()


def task_logreg_train(**context):
    import scripts.ML_model.model_train as model_train

    date_str = context["execution_date"].strftime("%Y-%m-%d")

    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping log_reg training for {date_str}: insufficient matured label history. "
            f"Earliest eligible training date is {FIRST_TRAINING_DATE}."
        )

    training_data_end_date = previous_month_start(date_str)
    validate_gold_medallion_range(BACKFILL_START_DATE, training_data_end_date)

    spark = get_spark()
    try:
        model_train.train_model(date_str, spark, model_type="log_reg")
    finally:
        spark.stop()


def task_branch_training_interval(**context):
    import pandas as pd

    date_str = context["execution_date"].strftime("%Y-%m-%d")
    if date_str < FIRST_TRAINING_DATE:
        print(
            f"Skipping scheduled training for {date_str}. Earliest training date "
            f"is {FIRST_TRAINING_DATE}."
        )
        return "skip_scheduled_xgboost_training"

    log_path = "/opt/airflow/model_bank/model_log.csv"
    if not os.path.exists(log_path):
        # First eligible run: no registry exists yet, so create both model families.
        print("No model_log.csv found. Running first scheduled training for both models.")
        return ["scheduled_xgboost_training", "scheduled_log_reg_training"]

    log_df = pd.read_csv(log_path)
    if "train_date" not in log_df.columns or log_df.empty:
        print("No prior train_date found in model_log.csv. Running training for both models.")
        return ["scheduled_xgboost_training", "scheduled_log_reg_training"]

    train_dates = pd.to_datetime(log_df["train_date"], errors="coerce").dropna()
    if train_dates.empty:
        print("No valid prior train_date found. Running training for both models.")
        return ["scheduled_xgboost_training", "scheduled_log_reg_training"]

    last_train_date = train_dates.max().strftime("%Y-%m-%d")
    months_since_last_train = month_difference(last_train_date, date_str)
    if months_since_last_train >= TRAINING_INTERVAL_MONTHS:
        # Scheduled refresh path. Monitoring can still request retraining later
        # even when this regular three-month cadence does not fire.
        print(
            f"Last training was {last_train_date}; {months_since_last_train} "
            f"month(s) elapsed. Running scheduled training."
        )
        return ["scheduled_xgboost_training", "scheduled_log_reg_training"]

    print(
        f"Last training was {last_train_date}; only {months_since_last_train} "
        f"month(s) elapsed. Next scheduled training is after "
        f"{TRAINING_INTERVAL_MONTHS} months."
    )
    return ["skip_scheduled_xgboost_training", "skip_scheduled_log_reg_training"]


def task_infer_xgb(model_name=None, **context):
    import scripts.ML_model.model_inference as model_inference

    date_str = context["execution_date"].strftime("%Y-%m-%d")
    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping inference for {date_str}. Earliest inference date is "
            f"{FIRST_TRAINING_DATE}."
        )

    # Forward optional manual overrides. model_inference defaults to the champion.
    max_snapshotdate = date_str
    model_name = dag_option(context, "model_name", model_name)
    max_snapshotdate = dag_option(context, "max_snapshotdate", max_snapshotdate)
    if dag_option_bool(context, "infer_through_latest_gold", False):
        max_snapshotdate = None

    # The inference module uses paths relative to the Airflow home directory.
    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        result = model_inference.run_new_gold_predictions(
            modelname=model_name,
            model_type="xgboost",
            min_snapshotdate=FIRST_TRAINING_DATE,
            max_snapshotdate=max_snapshotdate,
        )
        print("Inference volume-trigger result:", result)
        return result
    finally:
        os.chdir(original_directory)


def task_infer_logreg(model_name=None, **context):
    import scripts.ML_model.model_inference as model_inference

    date_str = context["execution_date"].strftime("%Y-%m-%d")
    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping log_reg inference for {date_str}. Earliest inference date is "
            f"{FIRST_TRAINING_DATE}."
        )

    max_snapshotdate = date_str
    model_name = dag_option(context, "model_name", model_name)
    max_snapshotdate = dag_option(context, "max_snapshotdate", max_snapshotdate)
    if dag_option_bool(context, "infer_through_latest_gold", False):
        max_snapshotdate = None

    original_directory = os.getcwd()
    try:
        os.chdir("/opt/airflow")
        result = model_inference.run_new_gold_predictions(
            modelname=model_name,
            model_type="log_reg",
            min_snapshotdate=FIRST_TRAINING_DATE,
            max_snapshotdate=max_snapshotdate,
        )
        print("Inference volume-trigger result:", result)
        return result
    finally:
        os.chdir(original_directory)


def task_monitor_xgb(model_name=None, **context):
    import scripts.ML_model.model_monitoring as model_monitoring

    date_str = context["execution_date"].strftime("%Y-%m-%d")
    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping model monitoring for {date_str}. Earliest monitoring "
            f"date is {FIRST_TRAINING_DATE}."
        )

    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        model_name = dag_run.conf.get("model_name", model_name)

    try:
        return model_monitoring.main(date_str, model_name, model_type="xgboost")
    except FileNotFoundError as error:
        if "Missing model monitoring reference feature-store partitions" in str(error):
            raise AirflowSkipException(str(error))
        raise


def task_monitor_logreg(model_name=None, **context):
    import scripts.ML_model.model_monitoring as model_monitoring

    date_str = context["execution_date"].strftime("%Y-%m-%d")
    if date_str < FIRST_TRAINING_DATE:
        raise AirflowSkipException(
            f"Skipping log_reg model monitoring for {date_str}. Earliest monitoring "
            f"date is {FIRST_TRAINING_DATE}."
        )

    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf:
        model_name = dag_run.conf.get("model_name", model_name)

    try:
        return model_monitoring.main(date_str, model_name, model_type="log_reg")
    except FileNotFoundError as error:
        if "Missing model monitoring reference feature-store partitions" in str(error):
            raise AirflowSkipException(str(error))
        raise


def task_branch_retraining(**context):
    import pandas as pd
    import scripts.ML_model.model_inference as model_inference

    task_instance = context["ti"]
    date_str = context["execution_date"].strftime("%Y-%m-%d")
    xgb_summary = task_instance.xcom_pull(task_ids="model_xgboost_monitor")
    logreg_summary = task_instance.xcom_pull(task_ids="model_log_reg_monitor")

    def recover_monitor_summary(summary, model_type):
        if summary:
            return summary

        dag_run = context.get("dag_run")
        model_name = None
        if dag_run and dag_run.conf:
            model_name = dag_run.conf.get("model_name")
        try:
            selected_model_name = model_inference.select_model_name(
                model_name,
                "/opt/airflow/model_bank",
                model_type=model_type,
            )
        except Exception:
            return None

        model_version = os.path.splitext(selected_model_name)[0]
        history_path = (
            "/opt/airflow/datamart/gold/model_monitoring/"
            f"{model_version}/{model_version}_monitoring_history.csv"
        )
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            matching_rows = history_df[history_df["snapshot_date"].eq(date_str)]
            if not matching_rows.empty:
                return matching_rows.iloc[-1].to_dict()
        return None

    xgb_summary = recover_monitor_summary(xgb_summary, "xgboost")
    logreg_summary = recover_monitor_summary(logreg_summary, "log_reg")

    xgb_retrain = bool(xgb_summary and xgb_summary.get("retrain_required"))
    logreg_retrain = bool(logreg_summary and logreg_summary.get("retrain_required"))

    selected = []
    if xgb_retrain:
        print("XGBoost monitoring requested retraining:", xgb_summary)
        selected.append("model_xgboost_automl")
    else:
        selected.append("skip_xgboost_retraining")

    if logreg_retrain:
        print("LogReg monitoring requested retraining:", logreg_summary)
        selected.append("model_log_reg_automl")
    else:
        selected.append("skip_log_reg_retraining")

    return selected

# DAG definition

default_args = {
    'owner': 'edward',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='Historical monthly medallion and ML lifecycle pipeline',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    # Training and monitoring depend on historical gold feature partitions from
    # earlier DAG runs, so backfill months must complete in chronological order.
    max_active_runs=1,
    tags=["data_pipeline"],
) as dag:

    setup_directories = PythonOperator(
        task_id="setup_directories",
        python_callable=make_dirs,
    )

    # label store

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label_store = PythonOperator(
        task_id='bronze_label_store',
        python_callable=task_bronze_lms,
    )

    silver_label_store = PythonOperator(
        task_id="silver_label_store",
        python_callable=task_silver_lms,
    )

    gold_label_store = PythonOperator(
        task_id="gold_label_store",
        python_callable=task_gold_labels,
    )

    label_store_completed = PythonOperator(
        task_id="label_store_completed",
        python_callable=task_validate_label_medallion,
    )

    setup_directories >> dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed


    # feature store
    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")

    bronze_feature_store = PythonOperator(
        task_id="bronze_feature_store",
        python_callable=task_bronze_features,
    )

    silver_feature_store = PythonOperator(
        task_id="silver_feature_store",
        python_callable=task_silver_features,
    )

    gold_feature_store = PythonOperator(
        task_id="gold_feature_store",
        python_callable=task_gold_features,
    )

    feature_store_completed = PythonOperator(
        task_id="feature_store_completed",
        python_callable=task_validate_feature_medallion,
    )

    setup_directories >> dep_check_source_data_bronze_1 >> bronze_feature_store >> silver_feature_store >> gold_feature_store
    gold_feature_store >> feature_store_completed


    # scheduled model training
    scheduled_training_start = BranchPythonOperator(
        task_id="scheduled_training_start",
        python_callable=task_branch_training_interval,
    )
    scheduled_xgboost_training = PythonOperator(
        task_id="scheduled_xgboost_training",
        python_callable=task_xgboost_train,
    )
    scheduled_log_reg_training = PythonOperator(
        task_id="scheduled_log_reg_training",
        python_callable=task_logreg_train,
    )
    skip_scheduled_xgboost_training = DummyOperator(
        task_id="skip_scheduled_xgboost_training",
    )
    skip_scheduled_log_reg_training = DummyOperator(
        task_id="skip_scheduled_log_reg_training",
    )
    scheduled_training_completed = DummyOperator(
        task_id="scheduled_training_completed",
        trigger_rule="none_failed_min_one_success",
    )

    scheduled_training_start >> scheduled_xgboost_training >> scheduled_training_completed
    scheduled_training_start >> skip_scheduled_xgboost_training >> scheduled_training_completed
    scheduled_training_start >> scheduled_log_reg_training >> scheduled_training_completed
    scheduled_training_start >> skip_scheduled_log_reg_training >> scheduled_training_completed


    # model inference
    model_inference_start = DummyOperator(task_id="model_inference_start")
    model_xgboost_inference = PythonOperator(
        task_id="model_xgboost_inference",
        python_callable=task_infer_xgb,
    )
    model_log_reg_inference = PythonOperator(
        task_id="model_log_reg_inference",
        python_callable=task_infer_logreg,
    )
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    model_inference_start >> model_xgboost_inference >> model_inference_completed
    model_inference_start >> model_log_reg_inference >> model_inference_completed


    # model monitoring
    model_monitor_start = DummyOperator(task_id="model_monitor_start")
    model_xgboost_monitor = PythonOperator(
        task_id="model_xgboost_monitor",
        python_callable=task_monitor_xgb,
    )
    model_log_reg_monitor = PythonOperator(
        task_id="model_log_reg_monitor",
        python_callable=task_monitor_logreg,
    )
    model_monitor_completed = DummyOperator(
        task_id="model_monitor_completed",
        trigger_rule="none_failed_min_one_success",
    )

    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_xgboost_monitor >> model_monitor_completed
    model_monitor_start >> model_log_reg_monitor >> model_monitor_completed


    # model retraining
    model_automl_start = BranchPythonOperator(
        task_id="model_automl_start",
        python_callable=task_branch_retraining,
    )
    model_xgboost_automl = PythonOperator(
        task_id="model_xgboost_automl",
        python_callable=task_xgboost_train,
    )
    model_log_reg_automl = PythonOperator(
        task_id="model_log_reg_automl",
        python_callable=task_logreg_train,
    )
    skip_xgboost_retraining = DummyOperator(task_id="skip_xgboost_retraining")
    skip_log_reg_retraining = DummyOperator(task_id="skip_log_reg_retraining")
    model_automl_completed = DummyOperator(
        task_id="model_automl_completed",
        trigger_rule="none_failed_min_one_success",
    )

    label_store_completed >> scheduled_training_start
    feature_store_completed >> scheduled_training_start
    scheduled_training_completed >> model_inference_start
    model_monitor_completed >> model_automl_start
    model_automl_start >> model_xgboost_automl >> model_automl_completed
    model_automl_start >> skip_xgboost_retraining >> model_automl_completed
    model_automl_start >> model_log_reg_automl >> model_automl_completed
    model_automl_start >> skip_log_reg_retraining >> model_automl_completed
