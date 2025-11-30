"""
Scenario analysis utilities for demand shock simulation and what-if analysis.

This module provides tools for simulating various scenarios that affect
pharmaceutical demand forecasting, including:
- Demand shocks (positive/negative)
- Supply disruptions
- Competitive events
- Policy changes

Based on research from:
- Ghannem et al. (2023): Supply-chain visibility and demand forecasting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Scenario Types and Definitions
# =============================================================================

class ShockType(Enum):
    """Types of demand shocks."""
    STEP = "step"  # Immediate permanent change
    IMPULSE = "impulse"  # Temporary spike then return
    RAMP = "ramp"  # Gradual change over time
    DECAY = "decay"  # Exponential decay effect
    SEASONAL = "seasonal"  # Seasonal adjustment


@dataclass
class DemandShock:
    """
    Definition of a demand shock scenario.
    
    Attributes:
        name: Identifier for the shock.
        shock_type: Type of shock (step, impulse, ramp, decay, seasonal).
        magnitude: Size of the shock (multiplicative factor, e.g., 0.8 = 20% drop).
        start_month: Month when shock begins (months_postgx).
        duration_months: How long the shock lasts (None = permanent).
        decay_rate: For decay type, exponential decay rate.
        affected_countries: List of countries affected (None = all).
        affected_brands: List of brands affected (None = all).
        affected_ther_areas: List of therapeutic areas affected (None = all).
    """
    name: str
    shock_type: ShockType
    magnitude: float
    start_month: int
    duration_months: Optional[int] = None
    decay_rate: float = 0.1
    affected_countries: Optional[List[str]] = None
    affected_brands: Optional[List[str]] = None
    affected_ther_areas: Optional[List[str]] = None


@dataclass
class ScenarioResult:
    """
    Result of applying a scenario to predictions.
    
    Attributes:
        scenario_name: Name of the scenario.
        base_predictions: Original predictions.
        adjusted_predictions: Predictions after applying scenario.
        impact_summary: Summary statistics of the impact.
        affected_series: List of (country, brand) tuples affected.
    """
    scenario_name: str
    base_predictions: pd.DataFrame
    adjusted_predictions: pd.DataFrame
    impact_summary: Dict[str, float]
    affected_series: List[Tuple[str, str]]


# =============================================================================
# Predefined Scenarios
# =============================================================================

# Generic erosion shock - accelerated erosion
ACCELERATED_EROSION = DemandShock(
    name="accelerated_erosion",
    shock_type=ShockType.RAMP,
    magnitude=0.9,  # 10% additional erosion per period
    start_month=0,
    duration_months=24,
)

# Supply disruption - temporary drop
SUPPLY_DISRUPTION = DemandShock(
    name="supply_disruption",
    shock_type=ShockType.IMPULSE,
    magnitude=0.7,  # 30% drop
    start_month=3,
    duration_months=3,
)

# Competitive launch - step decrease
COMPETITIVE_LAUNCH = DemandShock(
    name="competitive_launch",
    shock_type=ShockType.STEP,
    magnitude=0.85,  # 15% permanent decrease
    start_month=6,
    duration_months=None,  # Permanent
)

# Policy change - gradual impact
FORMULARY_EXCLUSION = DemandShock(
    name="formulary_exclusion",
    shock_type=ShockType.RAMP,
    magnitude=0.6,  # 40% decrease over duration
    start_month=0,
    duration_months=12,
)

# Epidemic demand surge - temporary increase
EPIDEMIC_SURGE = DemandShock(
    name="epidemic_surge",
    shock_type=ShockType.DECAY,
    magnitude=1.5,  # 50% increase, decaying
    start_month=0,
    duration_months=6,
    decay_rate=0.3,
)


def get_predefined_scenarios() -> Dict[str, DemandShock]:
    """Return dictionary of predefined scenario definitions."""
    return {
        'accelerated_erosion': ACCELERATED_EROSION,
        'supply_disruption': SUPPLY_DISRUPTION,
        'competitive_launch': COMPETITIVE_LAUNCH,
        'formulary_exclusion': FORMULARY_EXCLUSION,
        'epidemic_surge': EPIDEMIC_SURGE,
    }


# =============================================================================
# Shock Application Functions
# =============================================================================

def compute_shock_factor(
    shock: DemandShock,
    months_postgx: np.ndarray,
) -> np.ndarray:
    """
    Compute the shock factor for each time point.
    
    Args:
        shock: DemandShock definition.
        months_postgx: Array of months since generic entry.
        
    Returns:
        Array of multiplicative factors (1.0 = no change).
    """
    factors = np.ones_like(months_postgx, dtype=float)
    
    # Determine shock window
    start = shock.start_month
    if shock.duration_months is not None:
        end = start + shock.duration_months
    else:
        end = np.inf
    
    # Find affected months
    in_window = (months_postgx >= start) & (months_postgx < end)
    
    if not np.any(in_window):
        return factors
    
    # Apply shock based on type
    if shock.shock_type == ShockType.STEP:
        # Immediate permanent change
        factors[in_window] = shock.magnitude
    
    elif shock.shock_type == ShockType.IMPULSE:
        # Temporary spike then return
        factors[in_window] = shock.magnitude
    
    elif shock.shock_type == ShockType.RAMP:
        # Gradual change over time
        window_indices = np.where(in_window)[0]
        duration = len(window_indices)
        if duration > 0:
            # Linear interpolation from 1.0 to magnitude
            ramp = np.linspace(1.0, shock.magnitude, duration)
            factors[window_indices] = ramp
    
    elif shock.shock_type == ShockType.DECAY:
        # Exponential decay from magnitude back to 1.0
        window_months = months_postgx[in_window] - start
        decay = shock.magnitude * np.exp(-shock.decay_rate * window_months)
        # Ensure we decay toward 1.0, not 0
        factors[in_window] = 1.0 + (decay - 1.0)
    
    elif shock.shock_type == ShockType.SEASONAL:
        # Seasonal adjustment (simple sine wave)
        factors[in_window] = 1.0 + (shock.magnitude - 1.0) * np.sin(
            2 * np.pi * (months_postgx[in_window] % 12) / 12
        )
    
    return factors


def apply_demand_shock(
    predictions: pd.DataFrame,
    shock: DemandShock,
    volume_col: str = 'volume',
    country_col: str = 'country',
    brand_col: str = 'brand_name',
    time_col: str = 'months_postgx',
    ther_area_col: str = 'ther_area',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply a demand shock to predictions.
    
    Args:
        predictions: DataFrame with predictions.
        shock: DemandShock definition.
        volume_col: Name of volume column to adjust.
        country_col: Name of country column.
        brand_col: Name of brand column.
        time_col: Name of time column (months_postgx).
        ther_area_col: Name of therapeutic area column.
        inplace: Modify in place vs return copy.
        
    Returns:
        DataFrame with adjusted predictions.
    """
    if not inplace:
        predictions = predictions.copy()
    
    # Validate required columns
    required_cols = {volume_col, time_col}
    if not required_cols.issubset(set(predictions.columns)):
        missing = required_cols - set(predictions.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Build filter mask for affected series
    mask = pd.Series(True, index=predictions.index)
    
    if shock.affected_countries is not None and country_col in predictions.columns:
        countries_upper = [c.upper() for c in shock.affected_countries]
        mask &= predictions[country_col].str.upper().isin(countries_upper)
    
    if shock.affected_brands is not None and brand_col in predictions.columns:
        mask &= predictions[brand_col].isin(shock.affected_brands)
    
    if shock.affected_ther_areas is not None and ther_area_col in predictions.columns:
        mask &= predictions[ther_area_col].isin(shock.affected_ther_areas)
    
    if not mask.any():
        logger.warning(f"No series matched shock filter for '{shock.name}'")
        return predictions
    
    # Compute shock factors for matched rows
    months = predictions.loc[mask, time_col].values
    factors = compute_shock_factor(shock, months)
    
    # Apply factors
    predictions.loc[mask, volume_col] = predictions.loc[mask, volume_col] * factors
    
    # Ensure non-negative volumes
    predictions[volume_col] = predictions[volume_col].clip(lower=0)
    
    logger.info(f"Applied shock '{shock.name}' to {mask.sum()} rows")
    
    return predictions


def apply_multiple_shocks(
    predictions: pd.DataFrame,
    shocks: List[DemandShock],
    volume_col: str = 'volume',
    **kwargs,
) -> pd.DataFrame:
    """
    Apply multiple demand shocks sequentially.
    
    Args:
        predictions: DataFrame with predictions.
        shocks: List of DemandShock definitions.
        volume_col: Name of volume column.
        **kwargs: Additional arguments passed to apply_demand_shock.
        
    Returns:
        DataFrame with all shocks applied.
    """
    result = predictions.copy()
    
    for shock in shocks:
        result = apply_demand_shock(
            result,
            shock,
            volume_col=volume_col,
            inplace=True,
            **kwargs
        )
    
    return result


# =============================================================================
# Scenario Comparison and Analysis
# =============================================================================

def compare_scenarios(
    base_predictions: pd.DataFrame,
    scenarios: Dict[str, List[DemandShock]],
    volume_col: str = 'volume',
    country_col: str = 'country',
    brand_col: str = 'brand_name',
    time_col: str = 'months_postgx',
) -> Dict[str, ScenarioResult]:
    """
    Compare multiple scenarios against base predictions.
    
    Args:
        base_predictions: Original predictions DataFrame.
        scenarios: Dict mapping scenario name to list of shocks.
        volume_col: Name of volume column.
        country_col: Name of country column.
        brand_col: Name of brand column.
        time_col: Name of time column.
        
    Returns:
        Dict mapping scenario name to ScenarioResult.
    """
    results = {}
    
    for scenario_name, shocks in scenarios.items():
        # Apply shocks
        adjusted = apply_multiple_shocks(
            base_predictions,
            shocks,
            volume_col=volume_col,
            country_col=country_col,
            brand_col=brand_col,
            time_col=time_col,
        )
        
        # Compute impact summary
        base_total = base_predictions[volume_col].sum()
        adjusted_total = adjusted[volume_col].sum()
        
        impact_summary = {
            'total_volume_change': adjusted_total - base_total,
            'total_volume_change_pct': (adjusted_total - base_total) / base_total * 100 if base_total > 0 else 0,
            'mean_volume_change': (adjusted[volume_col] - base_predictions[volume_col]).mean(),
            'max_impact': (adjusted[volume_col] - base_predictions[volume_col]).min(),  # Negative = drop
            'min_impact': (adjusted[volume_col] - base_predictions[volume_col]).max(),  # Positive = increase
            'affected_periods': (adjusted[volume_col] != base_predictions[volume_col]).sum(),
        }
        
        # Find affected series
        affected_mask = adjusted[volume_col] != base_predictions[volume_col]
        if country_col in adjusted.columns and brand_col in adjusted.columns:
            affected_series = list(set(zip(
                adjusted.loc[affected_mask, country_col],
                adjusted.loc[affected_mask, brand_col]
            )))
        else:
            affected_series = []
        
        results[scenario_name] = ScenarioResult(
            scenario_name=scenario_name,
            base_predictions=base_predictions,
            adjusted_predictions=adjusted,
            impact_summary=impact_summary,
            affected_series=affected_series,
        )
    
    return results


def summarize_scenario_impacts(
    scenario_results: Dict[str, ScenarioResult],
) -> pd.DataFrame:
    """
    Create summary table of scenario impacts.
    
    Args:
        scenario_results: Dict of ScenarioResult objects.
        
    Returns:
        DataFrame with one row per scenario showing key impact metrics.
    """
    rows = []
    
    for name, result in scenario_results.items():
        row = {
            'scenario': name,
            'total_volume_change': result.impact_summary['total_volume_change'],
            'total_volume_change_pct': result.impact_summary['total_volume_change_pct'],
            'mean_volume_change': result.impact_summary['mean_volume_change'],
            'max_negative_impact': result.impact_summary['max_impact'],
            'max_positive_impact': result.impact_summary['min_impact'],
            'affected_periods': result.impact_summary['affected_periods'],
            'affected_series_count': len(result.affected_series),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def sensitivity_analysis(
    predictions: pd.DataFrame,
    base_shock: DemandShock,
    magnitude_range: Tuple[float, float] = (0.5, 1.5),
    n_steps: int = 10,
    volume_col: str = 'volume',
    **kwargs,
) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying shock magnitude.
    
    Args:
        predictions: DataFrame with predictions.
        base_shock: Base shock definition.
        magnitude_range: (min, max) magnitude to test.
        n_steps: Number of magnitude values to test.
        volume_col: Name of volume column.
        **kwargs: Additional arguments passed to apply_demand_shock.
        
    Returns:
        DataFrame with magnitude and total volume for each test.
    """
    magnitudes = np.linspace(magnitude_range[0], magnitude_range[1], n_steps)
    results = []
    
    for mag in magnitudes:
        # Create shock with test magnitude
        test_shock = DemandShock(
            name=f"{base_shock.name}_mag{mag:.2f}",
            shock_type=base_shock.shock_type,
            magnitude=mag,
            start_month=base_shock.start_month,
            duration_months=base_shock.duration_months,
            decay_rate=base_shock.decay_rate,
            affected_countries=base_shock.affected_countries,
            affected_brands=base_shock.affected_brands,
            affected_ther_areas=base_shock.affected_ther_areas,
        )
        
        # Apply and measure
        adjusted = apply_demand_shock(predictions, test_shock, volume_col=volume_col, **kwargs)
        
        results.append({
            'magnitude': mag,
            'total_volume': adjusted[volume_col].sum(),
            'mean_volume': adjusted[volume_col].mean(),
            'volume_change_pct': (adjusted[volume_col].sum() - predictions[volume_col].sum()) / 
                                 predictions[volume_col].sum() * 100 if predictions[volume_col].sum() > 0 else 0,
        })
    
    return pd.DataFrame(results)


def monte_carlo_scenario(
    predictions: pd.DataFrame,
    shock: DemandShock,
    magnitude_std: float = 0.1,
    n_simulations: int = 100,
    volume_col: str = 'volume',
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with random magnitude variations.
    
    Args:
        predictions: DataFrame with predictions.
        shock: Base shock definition.
        magnitude_std: Standard deviation for magnitude sampling.
        n_simulations: Number of simulations to run.
        volume_col: Name of volume column.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments passed to apply_demand_shock.
        
    Returns:
        Dict with simulation results including mean, std, percentiles.
    """
    rng = np.random.RandomState(seed)
    total_volumes = []
    
    for _ in range(n_simulations):
        # Sample magnitude from normal distribution centered on base magnitude
        sampled_magnitude = rng.normal(shock.magnitude, magnitude_std)
        sampled_magnitude = np.clip(sampled_magnitude, 0.1, 3.0)  # Reasonable bounds
        
        # Create shock with sampled magnitude
        test_shock = DemandShock(
            name=shock.name,
            shock_type=shock.shock_type,
            magnitude=sampled_magnitude,
            start_month=shock.start_month,
            duration_months=shock.duration_months,
            decay_rate=shock.decay_rate,
            affected_countries=shock.affected_countries,
            affected_brands=shock.affected_brands,
            affected_ther_areas=shock.affected_ther_areas,
        )
        
        # Apply and record total volume
        adjusted = apply_demand_shock(predictions, test_shock, volume_col=volume_col, **kwargs)
        total_volumes.append(adjusted[volume_col].sum())
    
    total_volumes = np.array(total_volumes)
    base_volume = predictions[volume_col].sum()
    
    return {
        'n_simulations': n_simulations,
        'base_volume': base_volume,
        'mean_volume': total_volumes.mean(),
        'std_volume': total_volumes.std(),
        'min_volume': total_volumes.min(),
        'max_volume': total_volumes.max(),
        'percentile_5': np.percentile(total_volumes, 5),
        'percentile_25': np.percentile(total_volumes, 25),
        'percentile_50': np.percentile(total_volumes, 50),
        'percentile_75': np.percentile(total_volumes, 75),
        'percentile_95': np.percentile(total_volumes, 95),
        'mean_change_pct': (total_volumes.mean() - base_volume) / base_volume * 100 if base_volume > 0 else 0,
    }


# =============================================================================
# Scenario Configuration and Loading
# =============================================================================

def load_scenarios_from_config(config: Dict[str, Any]) -> List[DemandShock]:
    """
    Load scenario definitions from configuration.
    
    Args:
        config: Configuration dict with scenario definitions.
        
    Returns:
        List of DemandShock objects.
        
    Config format:
        scenarios:
          - name: "my_shock"
            shock_type: "step"  # step, impulse, ramp, decay, seasonal
            magnitude: 0.9
            start_month: 0
            duration_months: 12
            affected_countries: ["US", "DE"]
    """
    scenarios = []
    
    for scenario_def in config.get('scenarios', []):
        shock_type_str = scenario_def.get('shock_type', 'step').upper()
        
        try:
            shock_type = ShockType[shock_type_str]
        except KeyError:
            logger.warning(f"Unknown shock type: {shock_type_str}, defaulting to STEP")
            shock_type = ShockType.STEP
        
        shock = DemandShock(
            name=scenario_def.get('name', 'unnamed'),
            shock_type=shock_type,
            magnitude=scenario_def.get('magnitude', 1.0),
            start_month=scenario_def.get('start_month', 0),
            duration_months=scenario_def.get('duration_months'),
            decay_rate=scenario_def.get('decay_rate', 0.1),
            affected_countries=scenario_def.get('affected_countries'),
            affected_brands=scenario_def.get('affected_brands'),
            affected_ther_areas=scenario_def.get('affected_ther_areas'),
        )
        scenarios.append(shock)
    
    return scenarios


def create_scenario_config(shock: DemandShock) -> Dict[str, Any]:
    """
    Convert DemandShock to configuration dict.
    
    Args:
        shock: DemandShock object.
        
    Returns:
        Configuration dict suitable for YAML serialization.
    """
    return {
        'name': shock.name,
        'shock_type': shock.shock_type.value,
        'magnitude': shock.magnitude,
        'start_month': shock.start_month,
        'duration_months': shock.duration_months,
        'decay_rate': shock.decay_rate,
        'affected_countries': shock.affected_countries,
        'affected_brands': shock.affected_brands,
        'affected_ther_areas': shock.affected_ther_areas,
    }
