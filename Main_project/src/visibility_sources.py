"""
Visibility source interfaces for optional external signals (IoT, CSV, etc.).

These are lightweight stubs to keep the pipeline flexible without
requiring any external systems. They return empty frames when data
is not present so core training continues to work.
"""

from pathlib import Path
from typing import Dict, Protocol, Optional

import pandas as pd


class VisibilitySource(Protocol):
    """Protocol for visibility sources."""

    def load(self, config: Optional[dict] = None) -> pd.DataFrame:
        ...

    def to_feature_frame(self, panel_df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        ...


class CsvVisibilitySource:
    """
    Simple CSV-backed visibility source.

    Expected columns: country, date, sensor_type, value
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self, config: Optional[dict] = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["country", "date", "sensor_type", "value"])
        return pd.read_csv(self.path)

    def to_feature_frame(self, panel_df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        df = panel_df.copy()
        raw = self.load(config)
        if raw.empty:
            return df

        raw["month"] = pd.to_datetime(raw["date"]).dt.to_period("M").astype(str)
        agg = raw.groupby(["country", "month", "sensor_type"])["value"].mean().reset_index()
        pivot = agg.pivot_table(index=["country", "month"], columns="sensor_type", values="value", fill_value=0)
        pivot.columns = [f"vis_sensor_{c}" for c in pivot.columns]
        pivot = pivot.reset_index()

        df["_vis_time_key"] = df["month"].astype(str) if "month" in df.columns else df["months_postgx"].astype(str)
        df = df.merge(pivot, left_on=["country", "_vis_time_key"], right_on=["country", "month"], how="left")

        vis_cols = [c for c in df.columns if c.startswith("vis_sensor_")]
        if vis_cols:
            df[vis_cols] = df[vis_cols].fillna(0)

        df.drop(columns=["_vis_time_key", "month_y"], inplace=True, errors="ignore")
        return df
