import glob
import os
import re

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

# The dashboard can run either inside the Airflow container or locally from the
# repo root, so every data path is anchored dynamically.
base_directory = "/opt/airflow" if os.path.isdir("/opt/airflow") else "."
DATAMART_DIR = os.path.join(base_directory, "datamart")
MODEL_BANK_DIR = os.path.join(base_directory, "model_bank")
MONITORING_DIR = os.path.join(base_directory, "datamart/gold/model_monitoring")
PERFORMANCE_DIR = os.path.join(base_directory, "datamart/gold/model_performance")
PREDICTIONS_DIR = os.path.join(base_directory, "datamart/gold/model_predictions")

DEFAULT_PSI_WARN = 0.10
DEFAULT_PSI_RETRAIN = 0.25
DEFAULT_CSI_WARN = 0.10
DEFAULT_CSI_RETRAIN = 0.25

STATUS_COLOUR = {
    "healthy": "#28a745",
    "warning": "#fd7e14",
    "retrain_required": "#dc3545",
}

EDA_CATALOG = {
    # Maps each visible EDA option to its folder, file prefix, and storage type.
    "Bronze": {
        "attributes":  ("bronze/features",  "bronze_attributes",  "csv"),
        "clickstream": ("bronze/features",  "bronze_clickstream", "csv"),
        "financials":  ("bronze/features",  "bronze_financials",  "csv"),
        "loan_daily":  ("bronze/lms",        "bronze_loan_daily",  "csv"),
    },
    "Silver": {
        "attributes":    ("silver/features",   "silver_attributes",    "parquet"),
        "clickstream":   ("silver/features",   "silver_clickstream",   "parquet"),
        "financials":    ("silver/features",   "silver_financials",    "parquet"),
        "loan_daily":    ("silver/loan_daily", "silver_loan_daily",    "parquet"),
    },
    "Gold": {
        "feature_store": ("gold/feature_store", "gold_feature_store", "parquet"),
        "label_store":   ("gold/label_store",   "gold_label_store",   "parquet"),
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_model_log():
    """Load the registry table that identifies champion and challenger models."""
    path = os.path.join(MODEL_BANK_DIR, "model_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def load_monitoring_history(model_version):
    """Load PSI/CSI monitoring history for one model version."""
    base = os.path.join(MONITORING_DIR, model_version, f"{model_version}_monitoring_history")
    for ext in (".parquet", ".csv"):
        path = base + ext
        if os.path.exists(path):
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_csi_detail(model_version, date_clean):
    base = os.path.join(
        MONITORING_DIR, model_version,
        f"{model_version}_csi_detail_{date_clean}",
    )
    for ext in (".parquet", ".csv"):
        path = base + ext
        if os.path.exists(path):
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_performance_history(model_version):
    """Load matured-label prediction performance history."""
    base = os.path.join(PERFORMANCE_DIR, model_version, f"{model_version}_performance_history")
    for ext in (".parquet", ".csv"):
        path = base + ext
        if os.path.exists(path):
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_performance_detail(model_version, date_clean):
    base = os.path.join(
        PERFORMANCE_DIR, model_version,
        f"{model_version}_performance_detail_{date_clean}",
    )
    for ext in (".parquet", ".csv"):
        path = base + ext
        if os.path.exists(path):
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_prediction_snapshot(model_version, date_clean):
    path = os.path.join(PREDICTIONS_DIR, model_version, f"{model_version}_predictions_{date_clean}.parquet")
    if not os.path.exists(path):
        # fallback: search by snapshot date suffix
        matches = glob.glob(os.path.join(PREDICTIONS_DIR, model_version, f"*_predictions_{date_clean}.parquet"))
        path = matches[0] if matches else path
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=120)
def prediction_summary(model_version):
    """Summarize all prediction partitions for the prediction dashboard."""
    directory = os.path.join(PREDICTIONS_DIR, model_version)
    if not os.path.isdir(directory):
        return pd.DataFrame()
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        m = re.search(r"_predictions_(\d{4}_\d{2}_\d{2})", os.path.basename(path))
        if not m:
            continue
        date_clean = m.group(1)
        try:
            df = pd.read_parquet(path)
            label_col = "label" if "label" in df.columns else None
            rows.append({
                "snapshot_date": date_clean.replace("_", "-"),
                "date_clean": date_clean,
                "row_count": len(df),
                "avg_score": round(float(df["model_predictions"].mean()), 4),
                "min_score": round(float(df["model_predictions"].min()), 4),
                "max_score": round(float(df["model_predictions"].max()), 4),
                "predicted_default_rate": round(float(df["label"].mean()), 4) if label_col else None,
                "prediction_threshold": round(float(df["prediction_threshold"].iloc[0]), 4) if "prediction_threshold" in df.columns else None,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def render_performance_panel(model_version, key_prefix):
    """Render prediction performance for a model version beside prediction outputs."""
    perf_history = load_performance_history(model_version)

    if perf_history.empty:
        st.info(
            "No performance history found for this model version. "
            "Run model_performance.py after labels have matured."
        )
        return

    perf_history = perf_history.copy()
    perf_history["evaluation_date"] = pd.to_datetime(perf_history["evaluation_date"])
    all_perf = perf_history[
        perf_history["performance_group"] == "all_matured_predictions"
    ].sort_values("evaluation_date").reset_index(drop=True)
    monthly_perf = perf_history[
        perf_history["performance_group"].astype(str).str.startswith("prediction_")
    ].copy()
    if not monthly_perf.empty:
        monthly_perf["prediction_snapshot_date"] = pd.to_datetime(
            monthly_perf["prediction_start_date"],
            errors="coerce",
        )

    latest_perf = all_perf.iloc[-1] if not all_perf.empty else perf_history.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    auc_val = latest_perf.get("auc", float("nan"))
    gini_val = latest_perf.get("gini", float("nan"))
    brier_val = latest_perf.get("brier", float("nan"))
    actual_rate = latest_perf.get("actual_default_rate", float("nan"))
    c1.metric("AUC", f"{float(auc_val):.4f}" if pd.notna(auc_val) else "n/a")
    c2.metric("Gini", f"{float(gini_val):.4f}" if pd.notna(gini_val) else "n/a")
    c3.metric("Brier Score", f"{float(brier_val):.4f}" if pd.notna(brier_val) else "n/a")
    c4.metric(
        "Actual Default Rate",
        f"{float(actual_rate):.2%}" if pd.notna(actual_rate) else "n/a",
    )

    if not monthly_perf.empty:
        fig_monthly = go.Figure()
        if "auc" in monthly_perf.columns:
            fig_monthly.add_trace(go.Scatter(
                x=monthly_perf["prediction_snapshot_date"],
                y=monthly_perf["auc"],
                mode="lines+markers",
                name="AUC",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            ))
        if "actual_default_rate" in monthly_perf.columns:
            fig_monthly.add_trace(go.Scatter(
                x=monthly_perf["prediction_snapshot_date"],
                y=monthly_perf["actual_default_rate"],
                mode="lines+markers",
                name="Actual default rate",
                line=dict(color="#dc3545", width=2, dash="dash"),
                marker=dict(size=6),
                yaxis="y2",
            ))
            fig_monthly.update_layout(
                yaxis2=dict(
                    title="Actual default rate",
                    overlaying="y",
                    side="right",
                    tickformat=".1%",
                    range=[0, max(monthly_perf["actual_default_rate"].max() * 1.5, 0.1)],
                ),
            )
        fig_monthly.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="#6c757d",
            annotation_text="Random AUC (0.5)",
            annotation_position="right",
        )
        fig_monthly.update_layout(
            title="Monthly matured prediction performance",
            xaxis_title="Prediction Snapshot Date",
            yaxis=dict(title="AUC", range=[0, 1]),
            height=360,
            margin=dict(r=120),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    class_metrics = [
        c for c in ["accuracy", "precision", "recall", "f1"]
        if c in monthly_perf.columns
    ]
    if class_metrics and not monthly_perf.empty:
        fig_cls = go.Figure()
        colours = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for metric, colour in zip(class_metrics, colours):
            fig_cls.add_trace(go.Scatter(
                x=monthly_perf["prediction_snapshot_date"],
                y=monthly_perf[metric],
                mode="lines+markers",
                name=metric.capitalize(),
                line=dict(color=colour, width=2),
                marker=dict(size=6),
            ))
        fig_cls.update_layout(
            title="Monthly classification metrics",
            xaxis_title="Prediction Snapshot Date",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=340,
            margin=dict(r=80),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    st.subheader("Score Band Calibration")
    eval_dates = sorted(
        perf_history["evaluation_date"].dt.strftime("%Y_%m_%d").unique().tolist(),
        reverse=True,
    )
    selected_eval_date = st.selectbox(
        "Evaluation date",
        eval_dates,
        key=f"{key_prefix}_eval_date",
    )
    detail_df = load_performance_detail(model_version, selected_eval_date)

    if detail_df.empty:
        st.info("No score band detail file found for this evaluation date.")
    else:
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Bar(
            x=detail_df["score_band"].astype(str),
            y=detail_df["actual_default_rate"],
            name="Actual default rate",
            marker_color="#dc3545",
            opacity=0.8,
        ))
        fig_cal.add_trace(go.Scatter(
            x=detail_df["score_band"].astype(str),
            y=detail_df["avg_score"],
            name="Avg model score",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))
        fig_cal.update_layout(
            title="Score band calibration",
            xaxis_title="Score Band",
            yaxis=dict(title="Actual Default Rate", range=[0, 1]),
            yaxis2=dict(
                title="Avg Model Score",
                overlaying="y",
                side="right",
                range=[0, 1],
            ),
            height=380,
            legend=dict(orientation="h", y=1.08),
            margin=dict(r=80),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    st.subheader("Performance History")
    history_cols = [
        c for c in [
            "evaluation_date", "performance_group",
            "row_count", "actual_default_rate", "predicted_default_rate",
            "avg_model_prediction", "prediction_threshold",
            "auc", "gini", "brier", "log_loss",
            "accuracy", "precision", "recall", "f1",
            "prediction_start_date", "prediction_end_date",
            "label_start_date", "label_end_date",
        ]
        if c in perf_history.columns
    ]
    st.dataframe(
        perf_history[history_cols].sort_values(
            ["evaluation_date", "performance_group"],
            ascending=[False, True],
        ),
        use_container_width=True,
        hide_index=True,
    )


def available_model_versions():
    versions = set()
    for directory in (MONITORING_DIR, PREDICTIONS_DIR, PERFORMANCE_DIR):
        if os.path.isdir(directory):
            versions.update(
                d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            )

    model_log = load_model_log()
    if not model_log.empty and "model_version" in model_log.columns:
        versions.update(
            str(v).replace(".pkl", "")
            for v in model_log["model_version"].dropna().tolist()
        )

    return sorted(versions)


def _eda_paths(layer, dataset):
    subfolder, prefix, fmt = EDA_CATALOG[layer][dataset]
    ext = ".csv" if fmt == "csv" else ".parquet"
    pattern = os.path.join(DATAMART_DIR, subfolder, f"{prefix}_????_??_??{ext}")
    return sorted(glob.glob(pattern))


def _date_from_path(path):
    m = re.search(r"(\d{4}_\d{2}_\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


@st.cache_data(ttl=300)
def eda_row_counts(layer, dataset):
    _, _, fmt = EDA_CATALOG[layer][dataset]
    rows = []
    for path in _eda_paths(layer, dataset):
        date_clean = _date_from_path(path)
        try:
            if fmt == "parquet":
                count = pq.ParquetFile(path).metadata.num_rows
            else:
                count = sum(1 for _ in open(path)) - 1
            rows.append({"date": date_clean.replace("_", "-"), "row_count": count})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_eda_partition(layer, dataset, date_clean):
    subfolder, prefix, fmt = EDA_CATALOG[layer][dataset]
    ext = ".csv" if fmt == "csv" else ".parquet"
    path = os.path.join(DATAMART_DIR, subfolder, f"{prefix}_{date_clean}{ext}")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path) if fmt == "csv" else pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Sidebar — top-level page navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Model Monitoring")
view = st.sidebar.radio("", ["Monitoring", "Predictions", "EDA"], label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Page: Monitoring
# ---------------------------------------------------------------------------

if view == "Monitoring":

    versions = available_model_versions()
    if not versions:
        st.error("No monitoring data found. Run model_monitoring.py first.")
        st.stop()

    model_log = load_model_log()

    default_idx = 0
    if not model_log.empty and "champion" in model_log.columns:
        champions = model_log[model_log["champion"] == 1]["model_version"].tolist()
        if champions and champions[0] in versions:
            default_idx = versions.index(champions[0])

    selected_version = st.sidebar.selectbox("Model version", versions, index=default_idx)

    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # Load monitoring data
    history = load_monitoring_history(selected_version)

    if not history.empty:
        history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
        history = history.sort_values("snapshot_date").reset_index(drop=True)
        history["psi_previous"] = history["psi_score"].shift(1)
        history["psi_change"] = history["psi_score"] - history["psi_previous"]
        history["psi_change_pct"] = (
            history["psi_change"] / history["psi_previous"].replace(0, pd.NA)
        )
        history["psi_trend"] = "first snapshot"
        history.loc[history["psi_change"] > 0, "psi_trend"] = "increased"
        history.loc[history["psi_change"] < 0, "psi_trend"] = "decreased"
        history.loc[history["psi_change"].eq(0), "psi_trend"] = "flat"
        latest = history.iloc[-1]
        psi_warn = float(latest.get("psi_warning_threshold", DEFAULT_PSI_WARN))
        psi_retrain = float(latest.get("psi_retrain_threshold", DEFAULT_PSI_RETRAIN))
        csi_warn = float(latest.get("csi_warning_threshold", DEFAULT_CSI_WARN))
        csi_retrain = float(latest.get("csi_retrain_threshold", DEFAULT_CSI_RETRAIN))
    else:
        latest = None
        psi_warn, psi_retrain = DEFAULT_PSI_WARN, DEFAULT_PSI_RETRAIN
        csi_warn, csi_retrain = DEFAULT_CSI_WARN, DEFAULT_CSI_RETRAIN

    tab_overview, tab_feature, tab_registry, tab_perf = st.tabs(
        ["Drift Overview", "Feature Drift", "Model Registry", "Model Performance"]
    )

    # -----------------------------------------------------------------------
    # Tab 1: drift overview
    # -----------------------------------------------------------------------

    with tab_overview:
        st.header("Drift Overview")

        if history.empty or latest is None:
            st.info("No monitoring history available for this model version.")
        else:
            status = str(latest.get("status", "unknown"))
            colour = STATUS_COLOUR.get(status, "#6c757d")
            st.markdown(
                f"**Latest snapshot:** {latest['snapshot_date'].date()} &nbsp;|&nbsp; "
                f"<span style='color:{colour}; font-weight:bold; font-size:1.1em;'>"
                f"{status.upper().replace('_', ' ')}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

            col1, col2, col3, col4 = st.columns(4)
            psi_delta = latest.get("psi_change")
            psi_delta_text = (
                f"{float(psi_delta):+.4f} vs previous"
                if pd.notna(psi_delta)
                else None
            )
            col1.metric(
                "PSI Score",
                f"{float(latest['psi_score']):.4f}",
                delta=psi_delta_text,
            )
            col2.metric("PSI Status", str(latest["psi_status"]))
            col3.metric("Max CSI", f"{float(latest['max_csi']):.4f}")
            col4.metric("Max CSI Feature", str(latest["max_csi_feature"]))

            st.divider()

            fig_psi = go.Figure()
            fig_psi.add_trace(go.Scatter(
                x=history["snapshot_date"],
                y=history["psi_score"],
                mode="lines+markers",
                name="PSI Score",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=7),
            ))
            fig_psi.add_hline(
                y=psi_warn, line_dash="dash", line_color="#fd7e14",
                annotation_text=f"Warning ({psi_warn})", annotation_position="right",
            )
            fig_psi.add_hline(
                y=psi_retrain, line_dash="dash", line_color="#dc3545",
                annotation_text=f"Retrain ({psi_retrain})", annotation_position="right",
            )
            fig_psi.update_layout(
                title="PSI Score Over Time",
                xaxis_title="Snapshot Date",
                yaxis_title="PSI",
                height=360,
                margin=dict(r=120),
            )
            st.plotly_chart(fig_psi, use_container_width=True)

            fig_csi = go.Figure()
            fig_csi.add_trace(go.Scatter(
                x=history["snapshot_date"],
                y=history["max_csi"],
                mode="lines+markers",
                name="Max CSI",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=7),
            ))
            fig_csi.add_hline(
                y=csi_warn, line_dash="dash", line_color="#fd7e14",
                annotation_text=f"Warning ({csi_warn})", annotation_position="right",
            )
            fig_csi.add_hline(
                y=csi_retrain, line_dash="dash", line_color="#dc3545",
                annotation_text=f"Retrain threshold ({csi_retrain})", annotation_position="right",
            )
            fig_csi.update_layout(
                title="Max CSI Over Time (informational)",
                xaxis_title="Snapshot Date",
                yaxis_title="Max CSI",
                height=360,
                margin=dict(r=120),
            )
            st.plotly_chart(fig_csi, use_container_width=True)

            st.subheader("Monitoring History")
            history_display = history.copy()
            if "psi_change_pct" in history_display.columns:
                history_display["psi_change_pct"] = history_display[
                    "psi_change_pct"
                ].apply(lambda value: f"{value:.2%}" if pd.notna(value) else "")
            display_cols = [
                "snapshot_date", "psi_score", "psi_previous",
                "psi_change", "psi_change_pct", "psi_trend", "psi_status",
                "max_csi", "max_csi_feature", "csi_status",
                "status", "retrain_required", "reference_count", "current_count",
            ]
            display_cols = [c for c in display_cols if c in history_display.columns]
            st.dataframe(
                history_display[display_cols].sort_values(
                    "snapshot_date",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------------------------
    # Tab 2: feature drift
    # -----------------------------------------------------------------------

    with tab_feature:
        st.header("Feature Drift Detail")

        if history.empty:
            st.info("No monitoring history available for this model version.")
        else:
            date_clean_col = (
                "snapshot_date_clean" if "snapshot_date_clean" in history.columns
                else "snapshot_date"
            )
            snapshot_options = sorted(history[date_clean_col].tolist(), reverse=True)
            selected_snapshot = st.selectbox("Snapshot date", snapshot_options)

            detail = load_csi_detail(selected_version, selected_snapshot)

            if detail.empty:
                st.info("No CSI detail file found for this snapshot.")
            else:
                psi_row = detail[detail["feature_name"] == "model_score"]
                csi_rows = detail[detail["feature_name"] != "model_score"].copy()

                if not psi_row.empty:
                    psi_val = float(psi_row.iloc[0]["csi"])
                    psi_stat = str(latest["psi_status"]) if latest is not None else ""
                    colour = STATUS_COLOUR.get(psi_stat, "#6c757d")
                    st.markdown(
                        f"**PSI (model score):** "
                        f"<span style='color:{colour}; font-weight:bold;'>"
                        f"{psi_val:.4f} — {psi_stat}</span>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

                if not csi_rows.empty:
                    csi_rows = csi_rows.sort_values("csi", ascending=True)

                    bar_colours = [
                        "#dc3545" if v >= csi_retrain
                        else "#fd7e14" if v >= csi_warn
                        else "#28a745"
                        for v in csi_rows["csi"]
                    ]

                    fig_bar = go.Figure(go.Bar(
                        x=csi_rows["csi"],
                        y=csi_rows["feature_name"],
                        orientation="h",
                        marker_color=bar_colours,
                        text=csi_rows["csi"].round(4),
                        textposition="outside",
                    ))
                    fig_bar.add_vline(
                        x=csi_warn, line_dash="dash", line_color="#fd7e14",
                        annotation_text=f"Warning ({csi_warn})",
                    )
                    fig_bar.add_vline(
                        x=csi_retrain, line_dash="dash", line_color="#dc3545",
                        annotation_text=f"Retrain ({csi_retrain})",
                    )
                    fig_bar.update_layout(
                        title="Feature CSI",
                        xaxis_title="CSI",
                        yaxis_title="Feature",
                        height=max(350, len(csi_rows) * 28 + 100),
                        margin=dict(r=80),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Detail Table")
                st.dataframe(detail, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Tab 3: model registry
    # -----------------------------------------------------------------------

    with tab_registry:
        st.header("Model Registry")

        if model_log.empty:
            st.info("No model_log.csv found.")
        else:
            display_cols = [
                c for c in [
                    "model_version", "train_date",
                    "auc_train", "auc_test", "auc_oot",
                    "gini_oot", "brier_oot", "log_loss_oot",
                    "calibration_method", "prediction_threshold",
                    "champion", "challenger",
                ]
                if c in model_log.columns
            ]

            def _highlight(row):
                if row.get("champion", 0) == 1:
                    return ["background-color: #d4edda; color: #155724"] * len(row)
                if row.get("challenger", 0) == 1:
                    return ["background-color: #fff3cd; color: #856404"] * len(row)
                return [""] * len(row)

            st.caption("Green = champion   |   Yellow = challenger   |   Promotion to champion requires human intervention")
            st.dataframe(
                model_log[display_cols].style.apply(_highlight, axis=1),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------------------------
    # Tab 4: model performance
    # -----------------------------------------------------------------------

    with tab_perf:
        st.header("Model Performance")

        perf_history = load_performance_history(selected_version)

        if perf_history.empty:
            st.info(
                "No performance history found for this model version. "
                "Run model_performance.py to evaluate predictions against matured labels."
            )
        else:
            perf_history["evaluation_date"] = pd.to_datetime(perf_history["evaluation_date"])

            all_perf = perf_history[
                perf_history["performance_group"] == "all_matured_predictions"
            ].sort_values("evaluation_date").reset_index(drop=True)

            latest_perf = all_perf.iloc[-1] if not all_perf.empty else perf_history.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            auc_val = latest_perf.get("auc", float("nan"))
            gini_val = latest_perf.get("gini", float("nan"))
            brier_val = latest_perf.get("brier", float("nan"))
            logloss_val = latest_perf.get("log_loss", float("nan"))
            col1.metric("AUC (latest)", f"{float(auc_val):.4f}" if pd.notna(auc_val) else "n/a")
            col2.metric("Gini (latest)", f"{float(gini_val):.4f}" if pd.notna(gini_val) else "n/a")
            col3.metric("Brier Score", f"{float(brier_val):.4f}" if pd.notna(brier_val) else "n/a")
            col4.metric("Log Loss", f"{float(logloss_val):.4f}" if pd.notna(logloss_val) else "n/a")

            st.divider()

            if len(all_perf) > 0 and "auc" in all_perf.columns:
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(
                    x=all_perf["evaluation_date"],
                    y=all_perf["auc"],
                    mode="lines+markers",
                    name="AUC",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=7),
                ))
                fig_auc.add_hline(
                    y=0.5, line_dash="dot", line_color="#6c757d",
                    annotation_text="Random (0.5)", annotation_position="right",
                )
                fig_auc.update_layout(
                    title="AUC Over Evaluation Dates",
                    xaxis_title="Evaluation Date",
                    yaxis_title="AUC",
                    yaxis=dict(range=[0, 1]),
                    height=340,
                    margin=dict(r=120),
                )
                st.plotly_chart(fig_auc, use_container_width=True)

            class_metrics = [c for c in ["accuracy", "precision", "recall", "f1"] if c in all_perf.columns]
            if class_metrics and len(all_perf) > 0:
                fig_cls = go.Figure()
                colours = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                for metric, colour in zip(class_metrics, colours):
                    fig_cls.add_trace(go.Scatter(
                        x=all_perf["evaluation_date"],
                        y=all_perf[metric],
                        mode="lines+markers",
                        name=metric.capitalize(),
                        line=dict(color=colour, width=2),
                        marker=dict(size=6),
                    ))
                fig_cls.update_layout(
                    title="Classification Metrics Over Time",
                    xaxis_title="Evaluation Date",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    height=340,
                    margin=dict(r=80),
                )
                st.plotly_chart(fig_cls, use_container_width=True)

            st.divider()

            st.subheader("Score Band Calibration")
            eval_dates = sorted(all_perf["evaluation_date"].dt.strftime("%Y_%m_%d").tolist(), reverse=True)
            if not eval_dates:
                eval_dates = sorted(
                    perf_history["evaluation_date"].dt.strftime("%Y_%m_%d").unique().tolist(),
                    reverse=True,
                )
            selected_eval_date = st.selectbox("Evaluation date", eval_dates, key="perf_eval_date")
            detail_df = load_performance_detail(selected_version, selected_eval_date)

            if detail_df.empty:
                st.info("No score band detail file found for this evaluation date.")
            else:
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Bar(
                    x=detail_df["score_band"].astype(str),
                    y=detail_df["actual_default_rate"],
                    name="Actual default rate",
                    marker_color="#dc3545",
                    opacity=0.8,
                ))
                fig_cal.add_trace(go.Scatter(
                    x=detail_df["score_band"].astype(str),
                    y=detail_df["avg_score"],
                    name="Avg model score",
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=8),
                    yaxis="y2",
                ))
                fig_cal.update_layout(
                    title="Score Band Calibration — actual default rate vs avg model score",
                    xaxis_title="Score Band",
                    yaxis=dict(title="Actual Default Rate", range=[0, 1]),
                    yaxis2=dict(
                        title="Avg Model Score",
                        overlaying="y",
                        side="right",
                        range=[0, 1],
                    ),
                    height=380,
                    legend=dict(orientation="h", y=1.08),
                    margin=dict(r=80),
                )
                st.plotly_chart(fig_cal, use_container_width=True)

                st.subheader("Score Band Detail")
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("Performance History")
            history_cols = [
                c for c in [
                    "evaluation_date", "performance_group",
                    "row_count", "actual_default_rate", "predicted_default_rate",
                    "auc", "gini", "brier", "log_loss",
                    "accuracy", "precision", "recall", "f1",
                    "prediction_start_date", "prediction_end_date",
                ]
                if c in perf_history.columns
            ]
            st.dataframe(
                perf_history[history_cols].sort_values(
                    ["evaluation_date", "performance_group"], ascending=[False, True]
                ),
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------------------------
# Page: Predictions
# ---------------------------------------------------------------------------

elif view == "Predictions":

    versions = available_model_versions()
    model_log = load_model_log()

    default_idx = 0
    if not model_log.empty and "champion" in model_log.columns:
        champions = model_log[model_log["champion"] == 1]["model_version"].tolist()
        if champions and champions[0] in versions:
            default_idx = versions.index(champions[0])

    pred_version = st.sidebar.selectbox("Model version", versions, index=default_idx, key="pred_version")

    summary = prediction_summary(pred_version)
    snapshot_dates = sorted(summary["date_clean"].tolist(), reverse=True) if not summary.empty else []
    pred_snapshot = st.sidebar.selectbox(
        "Snapshot",
        snapshot_dates if snapshot_dates else ["(none)"],
        key="pred_snapshot",
    )

    if st.sidebar.button("Refresh data", key="pred_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.header("Prediction Results")

    if summary.empty:
        st.info("No prediction files found for this model version. Run model_inference.py first.")
    else:
        summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Snapshots", len(summary))
        col2.metric("Latest snapshot", str(summary["snapshot_date"].max().date()))
        col3.metric("Latest avg score", f"{summary.sort_values('snapshot_date').iloc[-1]['avg_score']:.4f}")
        if summary["predicted_default_rate"].notna().any():
            col4.metric("Latest predicted default rate", f"{summary.sort_values('snapshot_date').iloc[-1]['predicted_default_rate']:.2%}")
        else:
            col4.metric("Latest row count", f"{int(summary.sort_values('snapshot_date').iloc[-1]['row_count']):,}")

        st.divider()

        # Volume over time
        fig_vol = go.Figure(go.Bar(
            x=summary["snapshot_date"],
            y=summary["row_count"],
            marker_color="#1f77b4",
            opacity=0.85,
        ))
        fig_vol.update_layout(
            title="Prediction volume over time",
            xaxis_title="Snapshot Date",
            yaxis_title="Row count",
            height=300,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Score trend + predicted default rate trend
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=summary["snapshot_date"],
            y=summary["avg_score"],
            mode="lines+markers",
            name="Avg model score",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ))
        if summary["predicted_default_rate"].notna().any():
            fig_score.add_trace(go.Scatter(
                x=summary["snapshot_date"],
                y=summary["predicted_default_rate"],
                mode="lines+markers",
                name="Predicted default rate",
                line=dict(color="#dc3545", width=2, dash="dash"),
                marker=dict(size=6),
                yaxis="y2",
            ))
            fig_score.update_layout(
                yaxis2=dict(
                    title="Predicted default rate",
                    overlaying="y",
                    side="right",
                    tickformat=".1%",
                    range=[0, max(summary["predicted_default_rate"].max() * 1.5, 0.1)],
                ),
            )
        if summary["prediction_threshold"].notna().any():
            threshold = float(summary["prediction_threshold"].iloc[0])
            fig_score.add_hline(
                y=threshold, line_dash="dot", line_color="#6c757d",
                annotation_text=f"Threshold ({threshold})", annotation_position="right",
            )
        fig_score.update_layout(
            title="Average model score and predicted default rate over time",
            xaxis_title="Snapshot Date",
            yaxis=dict(title="Avg model score", range=[0, 1]),
            height=360,
            margin=dict(r=120),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_score, use_container_width=True)

        st.divider()

        # Score distribution for selected snapshot
        st.subheader(f"Score distribution — {pred_snapshot}")

        if pred_snapshot != "(none)":
            snap_df = load_prediction_snapshot(pred_version, pred_snapshot)

            if snap_df.empty:
                st.info("Could not load prediction file for this snapshot.")
            else:
                threshold_val = float(snap_df["prediction_threshold"].iloc[0]) if "prediction_threshold" in snap_df.columns else None

                fig_hist = go.Figure(go.Histogram(
                    x=snap_df["model_predictions"],
                    nbinsx=50,
                    marker_color="#1f77b4",
                    opacity=0.8,
                    name="Score distribution",
                ))
                if threshold_val is not None:
                    fig_hist.add_vline(
                        x=threshold_val, line_dash="dash", line_color="#dc3545",
                        annotation_text=f"Threshold ({threshold_val:.3f})",
                        annotation_position="top right",
                    )
                fig_hist.update_layout(
                    title=f"Model score distribution — {pred_snapshot.replace('_', '-')}",
                    xaxis_title="Model score (probability)",
                    yaxis_title="Count",
                    height=340,
                    bargap=0.02,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Rows", f"{len(snap_df):,}")
                s2.metric("Avg score", f"{snap_df['model_predictions'].mean():.4f}")
                s3.metric("Median score", f"{snap_df['model_predictions'].median():.4f}")
                if "label" in snap_df.columns:
                    s4.metric("Predicted default rate", f"{snap_df['label'].mean():.2%}")
                elif threshold_val is not None:
                    pdr = (snap_df["model_predictions"] >= threshold_val).mean()
                    s4.metric("Predicted default rate", f"{pdr:.2%}")

                with st.expander("Data sample (first 200 rows)", expanded=False):
                    st.dataframe(snap_df.head(200), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Snapshot summary")
        summary_display = summary.copy()
        summary_display["snapshot_date"] = summary_display["snapshot_date"].dt.date
        if "predicted_default_rate" in summary_display.columns:
            summary_display["predicted_default_rate"] = summary_display["predicted_default_rate"].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )
        st.dataframe(
            summary_display.drop(columns=["date_clean"]).sort_values("snapshot_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.header("Prediction Performance")
        render_performance_panel(pred_version, key_prefix="prediction_page")


# ---------------------------------------------------------------------------
# Page: EDA
# ---------------------------------------------------------------------------

elif view == "EDA":

    _eda_layer = st.sidebar.selectbox("Layer", list(EDA_CATALOG.keys()), key="eda_layer")
    _eda_dataset = st.sidebar.selectbox(
        "Dataset", list(EDA_CATALOG[_eda_layer].keys()), key="eda_dataset"
    )
    _eda_available_dates = [_date_from_path(p) for p in _eda_paths(_eda_layer, _eda_dataset)]
    _eda_date = st.sidebar.selectbox(
        "Snapshot",
        sorted(_eda_available_dates, reverse=True) if _eda_available_dates else ["(none)"],
        key="eda_date",
    )

    if st.sidebar.button("Refresh data", key="eda_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.header("Exploratory Data Analysis")
    st.caption(f"{_eda_layer} / {_eda_dataset} / {_eda_date}")

    if _eda_date == "(none)":
        st.info("No partitions found for this dataset.")
    else:
        row_counts = eda_row_counts(_eda_layer, _eda_dataset)
        if not row_counts.empty:
            fig_rc = go.Figure(go.Scatter(
                x=pd.to_datetime(row_counts["date"]),
                y=row_counts["row_count"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            ))
            fig_rc.update_layout(
                title=f"Row count over time — {_eda_layer} / {_eda_dataset}",
                xaxis_title="Snapshot Date",
                yaxis_title="Row count",
                height=280,
            )
            st.plotly_chart(fig_rc, use_container_width=True)

        st.divider()

        df = load_eda_partition(_eda_layer, _eda_dataset, _eda_date)

        if df.empty:
            st.info("Partition is empty or could not be loaded.")
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            null_counts = df.isnull().sum()
            cols_with_nulls = int((null_counts > 0).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", len(df.columns))
            m3.metric("Numeric columns", len(numeric_cols))
            m4.metric("Columns with nulls", cols_with_nulls)

            st.divider()

            with st.expander("Schema", expanded=False):
                schema_rows = []
                for col in df.columns:
                    n_null = int(null_counts[col])
                    schema_rows.append({
                        "column": col,
                        "dtype": str(df[col].dtype),
                        "null_count": n_null,
                        "null_pct": round(n_null / len(df) * 100, 2) if len(df) else 0,
                        "n_unique": df[col].nunique(),
                    })
                st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

            null_series = null_counts[null_counts > 0].sort_values(ascending=False)
            if not null_series.empty:
                with st.expander("Missing values", expanded=True):
                    null_pct = (null_series / len(df) * 100).round(2)
                    fig_null = go.Figure(go.Bar(
                        x=null_series.index.tolist(),
                        y=null_pct.values,
                        marker_color="#dc3545",
                        text=null_pct.values,
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                    ))
                    fig_null.update_layout(
                        title="Null % per column",
                        xaxis_title="Column",
                        yaxis_title="Null %",
                        height=320,
                        yaxis=dict(range=[0, min(null_pct.max() * 1.25, 100)]),
                    )
                    st.plotly_chart(fig_null, use_container_width=True)

            if numeric_cols:
                with st.expander("Numeric summary", expanded=True):
                    st.dataframe(
                        df[numeric_cols].describe().T.round(4),
                        use_container_width=True,
                    )

            with st.expander("Column distribution", expanded=True):
                selected_col = st.selectbox("Column", df.columns.tolist(), key="eda_col")
                col_data = df[selected_col].dropna()

                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    fig_dist = go.Figure(go.Histogram(
                        x=col_data,
                        nbinsx=40,
                        marker_color="#1f77b4",
                        opacity=0.8,
                    ))
                    fig_dist.update_layout(
                        title=f"Distribution — {selected_col}",
                        xaxis_title=selected_col,
                        yaxis_title="Count",
                        height=340,
                        bargap=0.05,
                    )
                else:
                    vc = col_data.astype(str).value_counts().head(30)
                    fig_dist = go.Figure(go.Bar(
                        x=vc.index.tolist(),
                        y=vc.values,
                        marker_color="#1f77b4",
                    ))
                    fig_dist.update_layout(
                        title=f"Top values — {selected_col}",
                        xaxis_title=selected_col,
                        yaxis_title="Count",
                        height=340,
                    )
                st.plotly_chart(fig_dist, use_container_width=True)

            with st.expander("Data sample (first 200 rows)", expanded=False):
                st.dataframe(df.head(200), use_container_width=True, hide_index=True)
