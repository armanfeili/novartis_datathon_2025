# -*- coding: utf-8 -*-
"""
Helper file to locally compute Datathon 2025 Metrics.
This file is intended to be used by participants to test the metrics
using custom train/validation splits and also to generate submission files.

Metrics supported:
- Metric 1 (Phase 1-a): 0 actuals
- Metric 2 (Phase 1-b): 6 actuals
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Metric 1 (Phase 1-a)
# ------------------------------------------------------------------

def _compute_pe_phase1a(group: pd.DataFrame) -> float:
    """Compute PE for one (country, brand, bucket) group following the corrected Metric 1 formula."""
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan

    def sum_abs_diff(month_start: int, month_end: int) -> float:
        """Sum of absolute differences sum(|actual - pred|)."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
    
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        """Absolute difference of |sum(actuals) - sum(pred)|."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        sum_actual = subset["volume_actual"].sum()
        sum_pred = subset["volume_predict"].sum()
        return abs(sum_actual - sum_pred)

    term1 = 0.2 * sum_abs_diff(0, 23) / (24 * avg_vol)
    term2 = 0.5 * abs_sum_diff(0, 5) / (6 * avg_vol)
    term3 = 0.2 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term4 = 0.1 * abs_sum_diff(12, 23) / (12 * avg_vol)

    return term1 + term2 + term3 + term4


def _metric1(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    """Compute Metric 1 PE value.

    :param df_actual: Actual volume data
    :param df_pred: Predicted volume data
    :param df_aux: Auxiliary data with buckets and avg_vol
    :return: Weighted PE total (Phase 1)
    """
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")

    merged["start_month"] = merged.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged = merged[merged["start_month"] == 0].copy()

    pe_results = (
        merged.groupby(["country", "brand_name", "bucket"])
        .apply(_compute_pe_phase1a)
        .reset_index(name="PE")
    )

    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]

    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()


def compute_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame) -> float:
    """Compute Metric 1 (Phase 1).

    :param df_actual: Actual volume data
    :param df_pred: Predicted volume data
    :param df_aux: Auxiliary data with buckets and avg_vol
    :return: Computed Metric 1 value
    """
    return round(_metric1(df_actual, df_pred, df_aux), 4)


# ------------------------------------------------------------------
# Metric 2 (Phase 1-b)
# ------------------------------------------------------------------

def _compute_pe_phase1b(group: pd.DataFrame) -> float:
    """Compute PE for a specific country-brand-bucket group.

    :param group: DataFrame group with abs_diff and avg_vol columns
    :return: PE value for the group
    """
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan

    def sum_abs_diff(month_start: int, month_end: int) -> float:
        """Sum of absolute differences sum(|actual - pred|)."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
    
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        """Absolute difference of |sum(actuals) - sum(pred)|."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        sum_actual = subset["volume_actual"].sum()
        sum_pred = subset["volume_predict"].sum()
        return abs(sum_actual - sum_pred)

    term1 = 0.2 * sum_abs_diff(6, 23) / (18 * avg_vol)
    term2 = 0.5 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term3 = 0.3 * abs_sum_diff(12, 23) / (12 * avg_vol)
    
    return term1 + term2 + term3


def _metric2(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    """Compute Metric 2 PE value.

    :param df_actual: Actual volume data
    :param df_pred: Predicted volume data
    :param df_aux: Auxiliary data with buckets and avg_vol
    :return: Weighted PE total (Phase 2)
    """
    merged_data = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")

    merged_data["start_month"] = merged_data.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged_data = merged_data[merged_data["start_month"] == 6].copy()

    pe_results = (
        merged_data.groupby(["country", "brand_name", "bucket"])
        .apply(_compute_pe_phase1b)
        .reset_index(name="PE")
    )

    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]
    
    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()


def compute_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame) -> float:
    """Compute Metric 2 (Phase 2).

    :param df_actual: Actual volume data
    :param df_pred: Predicted volume data
    :param df_aux: Auxiliary data with buckets and avg_vol
    :return: Computed Metric 2 value
    """
    return round(_metric2(df_actual, df_pred, df_aux), 4)


# ------------------------------------------------------------------
# Workflow example
# ------------------------------------------------------------------
if __name__ == "__main__":

    # Paths (adapt as needed)
    DATA_PATH = Path("data")

    # ---- Load data ----
    # The auxiliar.metric_computation.csv contains the 'bucket', 'avg_vol', 'country' and 'brand_name' 
    # columns, and should be calculated before running this script based on the train_data.csv file
    # (take a look at the documentation for details on how to calculate 'bucket' and 'avg_vol').
    df_aux = pd.read_csv(DATA_PATH / "auxiliar_metric_computation.csv")
    train_data = pd.read_csv(DATA_PATH / "train_data.csv")
    submission_data = pd.read_csv(DATA_PATH / "submission_data.csv")
    submission = pd.read_csv(DATA_PATH / "submission_template.csv")

    # ---- Custom train/validation split ----
    train, validation = None # your_train_validation_split_function(train_data)

    # ---- Model training ----
    # Train your model here
    
    # ---- Predictions on validation set ----
    prediction = validation.copy()
    prediction["volume"] = None #model.predict(validation)

    # ---- Compute metrics on validation set ----
    m1 = compute_metric1(validation, prediction, df_aux)
    m2 = compute_metric2(validation, prediction, df_aux)

    print(f"Metric 1 - Phase 1-a (local validation): {m1}")
    print(f"Metric 2 - Phase 1-b (local validation): {m2}")
    
    
    # ---- Generate submission file ----
    # Fill in predicted 'volume' values of the submission 
    submission["volume"] = None #model.predict(submission_data)

    # ...

    # Save submission
    SAVE_PATH = Path("path/to/save/folder")
    ATTEMPT = "attempt_x"
    submission.to_csv(SAVE_PATH / f"submission_{ATTEMPT}.csv", sep=",", index=False)

    
