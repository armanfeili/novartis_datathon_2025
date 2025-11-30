"""
Scenario analysis utilities for demand-shock simulations.

Implements a minimal, safe version inspired by the research TODOs so
pipelines can experiment with what-if adjustments without touching
training code.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ShockSpec:
    """Definition of a demand shock."""

    type: str  # "multiplicative" or "additive"
    target: Dict[str, Optional[str]]  # keys: country, brand, ther_area
    start_month: int
    end_month: int
    factor: float = 1.0  # used for multiplicative
    delta: float = 0.0  # used for additive


def _matches_target(row: pd.Series, target: Dict[str, Optional[str]]) -> bool:
    """Check if a row matches the target spec."""
    if target.get("country") and row.get("country") != target["country"]:
        return False
    if target.get("brand") and row.get("brand_name") != target["brand"]:
        return False
    if target.get("ther_area") and row.get("ther_area") != target["ther_area"]:
        return False
    return True


def apply_demand_shock(panel_df: pd.DataFrame, shock: ShockSpec) -> pd.DataFrame:
    """
    Apply a demand shock to a copy of the panel.

    Args:
        panel_df: DataFrame with columns [country, brand_name, months_postgx, volume]
        shock: ShockSpec defining the adjustment

    Returns:
        Adjusted DataFrame with a new column `volume_shocked`.
    """
    df = panel_df.copy()
    mask = (
        df["months_postgx"].between(shock.start_month, shock.end_month)
        & df.apply(lambda r: _matches_target(r, shock.target), axis=1)
    )

    if shock.type == "multiplicative":
        df.loc[mask, "volume_shocked"] = df.loc[mask, "volume"] * shock.factor
    elif shock.type == "additive":
        df.loc[mask, "volume_shocked"] = df.loc[mask, "volume"] + shock.delta
    else:
        raise ValueError(f"Unknown shock type: {shock.type}")

    # Fill untouched rows
    if "volume_shocked" not in df.columns:
        df["volume_shocked"] = df["volume"]
    df["volume_shocked"] = df["volume_shocked"].fillna(df["volume"])
    df["volume_shocked"] = np.maximum(df["volume_shocked"], 0)

    return df
