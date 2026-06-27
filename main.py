import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import scripts.utils.data_processing_bronze_table
import scripts.utils.data_processing_silver_table
import scripts.utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Set up Directory Paths
bronze_lms_directory = "datamart/bronze/lms/"
bronze_feature_directory   = "datamart/bronze/features/"
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_feature_directory   = "datamart/silver/features/"
gold_label_store_directory  = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"

# Create new directory is not valid
for directory in [
    bronze_lms_directory,
    bronze_feature_directory,
    silver_loan_daily_directory,
    silver_feature_directory,
    gold_label_store_directory,
    gold_feature_store_directory,
]:
    os.makedirs(directory, exist_ok=True)

# run bronze backfill
print("\n BRONZE: LMS")
for date_str in dates_str_lst:
    scripts.utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

# create bronze feature datalake
print("\n BRONZE: FEATURES")
for date_str in dates_str_lst:
    scripts.utils.data_processing_bronze_table.process_bronze_features(
        date_str,
        bronze_feature_directory,
        spark
    )

# run silver backfill
print("\n SILVER: LOAN DAILY")
for date_str in dates_str_lst:
    scripts.utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory,
                                                            spark)
# create silver feature datalake
print("\n SILVER: FEATURES")
for date_str in dates_str_lst:
    scripts.utils.data_processing_silver_table.process_silver_features(
        date_str,
        bronze_feature_directory,
        silver_feature_directory,
        spark
    )

    
# run gold backfill
print("\n GOLD: LABEL STORE")
for date_str in dates_str_lst:
    scripts.utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

print("\nGOLD: FEATURE STORE")
for date_str in dates_str_lst:
    scripts.utils.data_processing_gold_table.process_gold_feature_store(
        date_str, silver_feature_directory, gold_feature_store_directory, spark
    )

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.parquet(*files_list)
print("row_count:",df.count())


print("\nPipeline complete.")
spark.stop()