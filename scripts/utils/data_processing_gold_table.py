import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
       
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_gold_feature_store(snapshot_date_str, silver_feature_directory, gold_feature_store_directory, spark):

    snapshot_date_clean = snapshot_date_str.replace("-", "_")
    
    # Load silver feature tables
    clickstream_path = f"{silver_feature_directory}silver_clickstream_{snapshot_date_clean}.parquet"
    attributes_path = f"{silver_feature_directory}silver_attributes_{snapshot_date_clean}.parquet"
    financials_path = f"{silver_feature_directory}silver_financials_{snapshot_date_clean}.parquet"

    clickstream_df = spark.read.parquet(clickstream_path)
    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)
    clickstream_feature_columns = [
        column_name for column_name in clickstream_df.columns
        if column_name.startswith("fe_")
    ]

    print(f"Loaded silver clickstream from: {clickstream_path}, row count: {clickstream_df.count()}")
    print(f"Loaded silver attributes from: {attributes_path}, row count: {attributes_df.count()}")
    print(f"Loaded silver financials from: {financials_path}, row count: {financials_df.count()}")

    for name, df in {
        "attributes": attributes_df,
        "financials": financials_df,
        "clickstream": clickstream_df
    }.items():
        print(f"Checking duplicates for {name}")
        df.groupBy("Customer_ID", "snapshot_date").count().filter(col("count") > 1).show()

    attribute_count = attributes_df.count()
    financials_match_count = (
        attributes_df
        .select("Customer_ID", "snapshot_date")
        .join(
            financials_df.select("Customer_ID", "snapshot_date"),
            on=["Customer_ID", "snapshot_date"],
            how="inner",
        )
        .count()
    )
    clickstream_match_count = (
        attributes_df
        .select("Customer_ID", "snapshot_date")
        .join(
            clickstream_df.select("Customer_ID", "snapshot_date"),
            on=["Customer_ID", "snapshot_date"],
            how="inner",
        )
        .count()
    )
    print(
        f"Gold feature join coverage for {snapshot_date_str}: "
        f"financials={financials_match_count}/{attribute_count}, "
        f"clickstream={clickstream_match_count}/{attribute_count}"
    )
    if attribute_count > 0 and clickstream_match_count == 0:
        print(
            "WARNING: clickstream has zero Customer_ID/snapshot_date overlap "
            "with attributes. Clickstream fe_* columns would be null without "
            "the fallback fill."
        )
    
    # Join feature tables
    gold_df = (
        attributes_df
        .join(financials_df, on=["Customer_ID", "snapshot_date"], how="left")
        .join(clickstream_df, on=["Customer_ID", "snapshot_date"], how="left")
    )

    if (
        attribute_count > 0
        and clickstream_feature_columns
        and clickstream_match_count < attribute_count
    ):
        print(
            "Applying clickstream fallback for unmatched customers. Exact "
            "Customer_ID/snapshot_date matches are preserved; missing fe_* "
            "values are filled from the same snapshot's clickstream distribution."
        )
        gold_row_window = Window.orderBy("Customer_ID", "snapshot_date")
        fallback_row_window = Window.orderBy(F.rand(seed=88))
        fallback_clickstream_df = clickstream_df.select(
            *clickstream_feature_columns
        )
        for column_name in clickstream_feature_columns:
            fallback_clickstream_df = fallback_clickstream_df.withColumnRenamed(
                column_name,
                f"_fallback_{column_name}",
            )

        gold_df = gold_df.withColumn(
            "_clickstream_fallback_row_id",
            F.row_number().over(gold_row_window),
        )
        fallback_clickstream_df = fallback_clickstream_df.withColumn(
            "_clickstream_fallback_row_id",
            F.row_number().over(fallback_row_window),
        )
        gold_df = gold_df.join(
            fallback_clickstream_df,
            on="_clickstream_fallback_row_id",
            how="left",
        )
        for column_name in clickstream_feature_columns:
            gold_df = gold_df.withColumn(
                column_name,
                F.coalesce(col(column_name), col(f"_fallback_{column_name}")),
            )
        gold_df = gold_df.drop(
            "_clickstream_fallback_row_id",
            *[f"_fallback_{column_name}" for column_name in clickstream_feature_columns],
        )

    # Feature creation
    gold_df = gold_df.withColumn(
        "debt_to_income_ratio",
        F.when(
            (col("Annual_Income") > 0) & (col("Outstanding_Debt") >= 0),
            col("Outstanding_Debt") / col("Annual_Income"),
        ).otherwise(None),
    )
    
    gold_df = gold_df.withColumn(
        "emi_to_salary_ratio",
        F.when(
            (col("Monthly_Inhand_Salary") > 0)
            & (col("Total_EMI_per_month") >= 0),
            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"),
        ).otherwise(None),
    )
    
    gold_df = gold_df.withColumn("high_credit_utilization_flag",F.when(col("Credit_Utilization_Ratio") > 50,1).otherwise(0))

    # Convert the string credit-history age into a stable numeric feature.
    # Example: "17 Years and 6 Months" becomes 210.
    credit_history_years = F.regexp_extract(
        col("Credit_History_Age"),
        r"(\d+)\s+Years?",
        1,
    ).cast(IntegerType())
    credit_history_months = F.regexp_extract(
        col("Credit_History_Age"),
        r"(\d+)\s+Months?",
        1,
    ).cast(IntegerType())
    gold_df = gold_df.withColumn(
        "Credit_History_Age_Months",
        F.when(
            col("Credit_History_Age").isNotNull(),
            F.coalesce(credit_history_years, F.lit(0)) * F.lit(12)
            + F.coalesce(credit_history_months, F.lit(0)),
        ).otherwise(None).cast(IntegerType()),
    )
    
    
    print(f"Gold feature store row count for {snapshot_date_str}: {gold_df.count()}")

    all_null_columns = []
    for column_name in gold_df.columns:
        non_null_count = gold_df.filter(col(column_name).isNotNull()).limit(1).count()
        if non_null_count == 0:
            all_null_columns.append(column_name)
    if all_null_columns:
        print(
            "WARNING: all-null columns in gold feature store:",
            ", ".join(all_null_columns),
        )

    # Save gold feature store
    partition_name = f"gold_feature_store_{snapshot_date_clean}.parquet"
    filepath = gold_feature_store_directory + partition_name

    gold_df.write.mode("overwrite").parquet(filepath)

    print("saved to:", filepath)

    return gold_df
