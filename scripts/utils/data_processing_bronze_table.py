import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    data_dir = os.environ.get("AIRFLOW_DATA_DIR", "data")
    csv_file_path = os.path.join(data_dir, "lms_loan_daily.csv")

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_features(snapshot_date_str, bronze_feature_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    data_dir = os.environ.get("AIRFLOW_DATA_DIR", "data")
    raw_feature_data = {
        "clickstream": os.path.join(data_dir, "feature_clickstream.csv"),
        "attributes": os.path.join(data_dir, "features_attributes.csv"),
        "financials": os.path.join(data_dir, "features_financials.csv"),
    }

    for feature_name, file_path in raw_feature_data.items():
        df = spark.read.csv(file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
        print(f"{feature_name} - {snapshot_date_str} row count:", df.count())
        
        partition_name = f"bronze_{feature_name}_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_feature_directory + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    return df
