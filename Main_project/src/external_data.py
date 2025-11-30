"""
External/context data loaders and join helpers.

Lightweight, fail-safe utilities to bring optional context tables
into the feature pipeline. All functions return empty frames with the
expected schema when files are missing so the core pipeline keeps running.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import EXTERNAL_DATA_DIR


def _load_csv_if_exists(path: Path, columns: list) -> pd.DataFrame:
    """Load CSV with expected columns; return empty frame if missing."""
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    for col in missing:
        df[col] = pd.NA
    return df[columns]


def load_holiday_calendar(country: Optional[str] = None) -> pd.DataFrame:
    """Load holiday calendar; filter by country if provided."""
    cols = ["country", "date", "holiday_name", "is_public_holiday"]
    df = _load_csv_if_exists(EXTERNAL_DATA_DIR / "holidays.csv", cols)
    if country:
        df = df[df["country"] == country]
    return df


def load_epidemic_events() -> pd.DataFrame:
    """Load epidemic/health events with severity scores."""
    cols = ["country", "date", "event_type", "severity_score"]
    return _load_csv_if_exists(EXTERNAL_DATA_DIR / "epidemics.csv", cols)


def load_macro_indicators() -> pd.DataFrame:
    """Load macroeconomic indicators (country, date, indicator_name, value)."""
    cols = ["country", "date", "indicator_name", "value"]
    return _load_csv_if_exists(EXTERNAL_DATA_DIR / "macro.csv", cols)


def load_promo_or_policy_events() -> pd.DataFrame:
    """Load promotion/policy events."""
    cols = ["country", "date", "event_type", "intensity"]
    return _load_csv_if_exists(EXTERNAL_DATA_DIR / "promo_policy.csv", cols)


def join_external_context(panel_df: pd.DataFrame,
                          external_tables: Dict[str, pd.DataFrame],
                          max_event_lag: int = 24) -> pd.DataFrame:
    """
    Join external context tables to the panel using (country, month) if available.
    Adds simple, low-leakage aggregate features:
      - vis_holiday_flag
      - vis_epidemic_severity
      - vis_macro_value (mean across indicators)
      - vis_promo_flag
    """
    df = panel_df.copy()

    # Prefer calendar month if present; fallback to months_postgx
    time_key = "month" if "month" in df.columns else "months_postgx"
    df["_ext_time_key"] = df[time_key].astype(str)

    # Holidays
    holidays = external_tables.get("holidays")
    if holidays is not None and not holidays.empty:
        hol = holidays.copy()
        # Derive month index if date present
        if "date" in hol.columns:
            hol["month"] = pd.to_datetime(hol["date"]).dt.to_period("M").astype(str)
        hol["vis_holiday_flag"] = 1
        df = df.merge(
            hol[["country", "month", "vis_holiday_flag"]],
            left_on=["country", "_ext_time_key"],
            right_on=["country", "month"],
            how="left"
        )

    # Epidemics
    epidemics = external_tables.get("epidemics")
    if epidemics is not None and not epidemics.empty:
        epi = epidemics.copy()
        if "date" in epi.columns:
            epi["month"] = pd.to_datetime(epi["date"]).dt.to_period("M").astype(str)
        # Aggregate severity per country-month
        epi_agg = epi.groupby(["country", "month"])["severity_score"].mean().reset_index()
        epi_agg.rename(columns={"severity_score": "vis_epidemic_severity"}, inplace=True)
        df = df.merge(
            epi_agg,
            left_on=["country", "_ext_time_key"],
            right_on=["country", "month"],
            how="left"
        )

    # Macro indicators
    macro = external_tables.get("macro")
    if macro is not None and not macro.empty:
        mac = macro.copy()
        if "date" in mac.columns:
            mac["month"] = pd.to_datetime(mac["date"]).dt.to_period("M").astype(str)
        mac_agg = mac.groupby(["country", "month"])["value"].mean().reset_index()
        mac_agg.rename(columns={"value": "vis_macro_value"}, inplace=True)
        df = df.merge(
            mac_agg,
            left_on=["country", "_ext_time_key"],
            right_on=["country", "month"],
            how="left"
        )

    # Promotions/policies
    promos = external_tables.get("promotions")
    if promos is not None and not promos.empty:
        pr = promos.copy()
        if "date" in pr.columns:
            pr["month"] = pd.to_datetime(pr["date"]).dt.to_period("M").astype(str)
        pr["vis_promo_flag"] = 1
        df = df.merge(
            pr[["country", "month", "vis_promo_flag"]],
            left_on=["country", "_ext_time_key"],
            right_on=["country", "month"],
            how="left"
        )

    # Basic lag of epidemic severity (captures recency up to max_event_lag)
    if "vis_epidemic_severity" in df.columns:
        df = df.sort_values(["country", time_key])
        df["vis_epidemic_severity_lag1"] = df.groupby("country")["vis_epidemic_severity"].shift(1)
        df["vis_epidemic_severity_lag1"] = df["vis_epidemic_severity_lag1"].fillna(0)

    # Fill NaNs introduced by merges with 0
    vis_cols = [c for c in df.columns if c.startswith("vis_")]
    if vis_cols:
        df[vis_cols] = df[vis_cols].fillna(0)

    # Drop helper columns added for merging
    df.drop(columns=["_ext_time_key"], inplace=True, errors="ignore")
    for drop_col in ["month_x", "month_y"]:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    return df
