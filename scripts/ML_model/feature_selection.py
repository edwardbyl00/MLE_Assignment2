"""Feature-selection rules shared by training, inference, and monitoring."""

# These columns are identifiers, labels, metadata, or previous prediction outputs.
# Keeping them out prevents target leakage and avoids training on non-feature fields.
excluded_col = {
    "Customer_ID",
    "snapshot_date",
    "feature_snapshot_date",
    "loan_id",
    "label",
    "label_def",
    "model_name",
    "model_predictions",
    "Name",
    "SSN",
    # Replaced by numeric Credit_History_Age_Months in the gold feature store.
    "Credit_History_Age",
}

def select_model_feature_columns(columns):
    """Return only columns that are eligible as model input features."""
    return [column for column in columns if column not in excluded_col]
