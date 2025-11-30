"""
External data loading and integration for Novartis Datathon 2025.

This module provides utilities for loading and integrating external data sources
such as holidays, epidemic events, macroeconomic indicators, and promotional events.

Based on research from:
- Ghannem et al. (2023): Supply-chain visibility and demand forecasting
- Li et al. (2024): CNN-LSTM for predicting drug sales volume

Data sources follow canonical schemas:
- Holidays: date, country, holiday_name, holiday_type
- Epidemics: date, country, event_name, severity (1-5)
- Macro indicators: date, country, indicator_name, value
- Promo/Policy events: date, country, event_type, event_description
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical Schemas
# =============================================================================

@dataclass
class HolidayRecord:
    """Canonical schema for holiday data."""
    date: date
    country: str
    holiday_name: str
    holiday_type: str  # "national", "regional", "religious", "observed"
    
    
@dataclass
class EpidemicEvent:
    """Canonical schema for epidemic/pandemic events."""
    start_date: date
    end_date: Optional[date]
    country: str
    event_name: str
    severity: int  # 1-5 scale (1=minor, 5=major pandemic)
    affected_ther_areas: List[str] = field(default_factory=list)


@dataclass
class MacroIndicator:
    """Canonical schema for macroeconomic indicators."""
    date: date
    country: str
    indicator_name: str  # "gdp_growth", "unemployment", "inflation", "healthcare_spend"
    value: float
    unit: str = ""  # "%", "USD", etc.


@dataclass
class PromoOrPolicyEvent:
    """Canonical schema for promotional or policy events."""
    start_date: date
    end_date: Optional[date]
    country: str
    event_type: str  # "price_cut", "patent_expiry", "formulary_change", "marketing_campaign"
    event_description: str
    affected_brands: List[str] = field(default_factory=list)
    impact_magnitude: Optional[float] = None  # Expected impact on volume (-1 to 1)


# =============================================================================
# External Data Loaders
# =============================================================================

def load_holiday_calendar(
    path: Optional[Union[str, Path]] = None,
    countries: Optional[List[str]] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
) -> pd.DataFrame:
    """
    Load holiday calendar data.
    
    Args:
        path: Path to holiday CSV file. If None, returns empty DataFrame.
        countries: Filter to specific countries.
        start_date: Filter holidays on or after this date.
        end_date: Filter holidays on or before this date.
        
    Returns:
        DataFrame with columns: date, country, holiday_name, holiday_type
        
    Notes:
        - Returns empty DataFrame with correct schema if file not found.
        - Dates are converted to pandas datetime.
        - Country codes are uppercased for consistency.
    """
    # Define expected schema
    schema_cols = ['date', 'country', 'holiday_name', 'holiday_type']
    
    if path is None:
        logger.debug("No holiday calendar path provided, returning empty DataFrame")
        return pd.DataFrame(columns=schema_cols)
    
    path = Path(path)
    if not path.exists():
        logger.warning(f"Holiday calendar file not found: {path}")
        return pd.DataFrame(columns=schema_cols)
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = {'date', 'country'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Holiday file missing columns: {missing}")
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['date'] = pd.to_datetime(df['date'])
        df['country'] = df['country'].str.upper()
        
        # Add missing optional columns
        if 'holiday_name' not in df.columns:
            df['holiday_name'] = 'Unknown'
        if 'holiday_type' not in df.columns:
            df['holiday_type'] = 'national'
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['date'] <= end_dt]
        
        logger.info(f"Loaded {len(df)} holiday records from {path}")
        return df[schema_cols]
        
    except Exception as e:
        logger.error(f"Error loading holiday calendar: {e}")
        return pd.DataFrame(columns=schema_cols)


def load_epidemic_events(
    path: Optional[Union[str, Path]] = None,
    countries: Optional[List[str]] = None,
    min_severity: int = 1,
) -> pd.DataFrame:
    """
    Load epidemic/pandemic event data.
    
    Args:
        path: Path to epidemic events CSV file.
        countries: Filter to specific countries.
        min_severity: Minimum severity level (1-5) to include.
        
    Returns:
        DataFrame with columns: start_date, end_date, country, event_name, 
                               severity, affected_ther_areas
                               
    Notes:
        - severity is 1-5 scale (1=minor outbreak, 5=major pandemic)
        - affected_ther_areas is a pipe-separated string in CSV
    """
    schema_cols = ['start_date', 'end_date', 'country', 'event_name', 
                   'severity', 'affected_ther_areas']
    
    if path is None:
        logger.debug("No epidemic events path provided, returning empty DataFrame")
        return pd.DataFrame(columns=schema_cols)
    
    path = Path(path)
    if not path.exists():
        logger.warning(f"Epidemic events file not found: {path}")
        return pd.DataFrame(columns=schema_cols)
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = {'start_date', 'country', 'event_name'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Epidemic file missing columns: {missing}")
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['start_date'] = pd.to_datetime(df['start_date'])
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'])
        else:
            df['end_date'] = pd.NaT
        
        df['country'] = df['country'].str.upper()
        
        # Add missing optional columns
        if 'severity' not in df.columns:
            df['severity'] = 3  # Default to medium severity
        if 'affected_ther_areas' not in df.columns:
            df['affected_ther_areas'] = ''
        
        df['severity'] = df['severity'].astype(int).clip(1, 5)
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        df = df[df['severity'] >= min_severity]
        
        logger.info(f"Loaded {len(df)} epidemic events from {path}")
        return df[schema_cols]
        
    except Exception as e:
        logger.error(f"Error loading epidemic events: {e}")
        return pd.DataFrame(columns=schema_cols)


def load_macro_indicators(
    path: Optional[Union[str, Path]] = None,
    countries: Optional[List[str]] = None,
    indicators: Optional[List[str]] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
) -> pd.DataFrame:
    """
    Load macroeconomic indicator data.
    
    Args:
        path: Path to macro indicators CSV file.
        countries: Filter to specific countries.
        indicators: Filter to specific indicator names.
        start_date: Filter to data on or after this date.
        end_date: Filter to data on or before this date.
        
    Returns:
        DataFrame with columns: date, country, indicator_name, value, unit
        
    Common indicators:
        - gdp_growth: Annual GDP growth rate (%)
        - unemployment: Unemployment rate (%)
        - inflation: Consumer price inflation (%)
        - healthcare_spend: Healthcare spending per capita (USD)
    """
    schema_cols = ['date', 'country', 'indicator_name', 'value', 'unit']
    
    if path is None:
        logger.debug("No macro indicators path provided, returning empty DataFrame")
        return pd.DataFrame(columns=schema_cols)
    
    path = Path(path)
    if not path.exists():
        logger.warning(f"Macro indicators file not found: {path}")
        return pd.DataFrame(columns=schema_cols)
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = {'date', 'country', 'indicator_name', 'value'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Macro indicators file missing columns: {missing}")
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['date'] = pd.to_datetime(df['date'])
        df['country'] = df['country'].str.upper()
        df['indicator_name'] = df['indicator_name'].str.lower().str.replace(' ', '_')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if 'unit' not in df.columns:
            df['unit'] = ''
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if indicators:
            indicators_lower = [i.lower().replace(' ', '_') for i in indicators]
            df = df[df['indicator_name'].isin(indicators_lower)]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['date'] <= end_dt]
        
        # Drop rows with invalid values
        df = df.dropna(subset=['value'])
        
        logger.info(f"Loaded {len(df)} macro indicator records from {path}")
        return df[schema_cols]
        
    except Exception as e:
        logger.error(f"Error loading macro indicators: {e}")
        return pd.DataFrame(columns=schema_cols)


def load_promo_or_policy_events(
    path: Optional[Union[str, Path]] = None,
    countries: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load promotional or policy event data.
    
    Args:
        path: Path to promo/policy events CSV file.
        countries: Filter to specific countries.
        event_types: Filter to specific event types.
        
    Returns:
        DataFrame with columns: start_date, end_date, country, event_type,
                               event_description, affected_brands, impact_magnitude
                               
    Event types:
        - price_cut: Price reduction event
        - patent_expiry: Patent expiration
        - formulary_change: Formulary/reimbursement change
        - marketing_campaign: Marketing or promotional campaign
        - regulatory_approval: New indication or formulation approval
        - competitor_launch: Competitor product launch
    """
    schema_cols = ['start_date', 'end_date', 'country', 'event_type',
                   'event_description', 'affected_brands', 'impact_magnitude']
    
    if path is None:
        logger.debug("No promo/policy events path provided, returning empty DataFrame")
        return pd.DataFrame(columns=schema_cols)
    
    path = Path(path)
    if not path.exists():
        logger.warning(f"Promo/policy events file not found: {path}")
        return pd.DataFrame(columns=schema_cols)
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = {'start_date', 'country', 'event_type'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(f"Promo/policy file missing columns: {missing}")
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['start_date'] = pd.to_datetime(df['start_date'])
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'])
        else:
            df['end_date'] = pd.NaT
        
        df['country'] = df['country'].str.upper()
        df['event_type'] = df['event_type'].str.lower().str.replace(' ', '_')
        
        # Add missing optional columns
        if 'event_description' not in df.columns:
            df['event_description'] = ''
        if 'affected_brands' not in df.columns:
            df['affected_brands'] = ''
        if 'impact_magnitude' not in df.columns:
            df['impact_magnitude'] = np.nan
        else:
            df['impact_magnitude'] = pd.to_numeric(df['impact_magnitude'], errors='coerce')
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if event_types:
            event_types_lower = [e.lower().replace(' ', '_') for e in event_types]
            df = df[df['event_type'].isin(event_types_lower)]
        
        logger.info(f"Loaded {len(df)} promo/policy events from {path}")
        return df[schema_cols]
        
    except Exception as e:
        logger.error(f"Error loading promo/policy events: {e}")
        return pd.DataFrame(columns=schema_cols)


# =============================================================================
# External Context Integration
# =============================================================================

def join_external_context(
    panel: pd.DataFrame,
    holidays: Optional[pd.DataFrame] = None,
    epidemics: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    promo_policy: Optional[pd.DataFrame] = None,
    date_col: str = 'month',
    country_col: str = 'country',
    brand_col: str = 'brand_name',
    aggregate_holidays: str = 'count',  # 'count', 'binary', 'list'
    fill_missing: bool = True,
) -> pd.DataFrame:
    """
    Join external context data to the main panel.
    
    Args:
        panel: Main panel DataFrame with time series data.
        holidays: Holiday calendar DataFrame.
        epidemics: Epidemic events DataFrame.
        macro: Macro indicators DataFrame.
        promo_policy: Promo/policy events DataFrame.
        date_col: Name of date column in panel.
        country_col: Name of country column in panel.
        brand_col: Name of brand column in panel.
        aggregate_holidays: How to aggregate holidays per month.
        fill_missing: Fill missing values with defaults.
        
    Returns:
        Panel DataFrame with additional context columns:
        - ext_n_holidays: Number of holidays in the month
        - ext_is_holiday_month: Binary indicator for holiday month
        - ext_epidemic_severity: Max epidemic severity affecting month
        - ext_epidemic_active: Binary indicator for active epidemic
        - ext_gdp_growth: GDP growth rate
        - ext_unemployment: Unemployment rate
        - ext_inflation: Inflation rate
        - ext_healthcare_spend: Healthcare spending
        - ext_promo_active: Binary indicator for active promotion
        - ext_policy_change: Binary indicator for policy change
    """
    result = panel.copy()
    
    # Ensure date column is datetime
    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])
    
    # Process holidays
    if holidays is not None and len(holidays) > 0:
        result = _join_holidays(result, holidays, date_col, country_col, aggregate_holidays)
    else:
        # Add empty columns
        result['ext_n_holidays'] = 0
        result['ext_is_holiday_month'] = 0
    
    # Process epidemics
    if epidemics is not None and len(epidemics) > 0:
        result = _join_epidemics(result, epidemics, date_col, country_col)
    else:
        result['ext_epidemic_severity'] = 0
        result['ext_epidemic_active'] = 0
    
    # Process macro indicators
    if macro is not None and len(macro) > 0:
        result = _join_macro_indicators(result, macro, date_col, country_col)
    else:
        for col in ['ext_gdp_growth', 'ext_unemployment', 'ext_inflation', 'ext_healthcare_spend']:
            result[col] = np.nan
    
    # Process promo/policy events
    if promo_policy is not None and len(promo_policy) > 0:
        result = _join_promo_policy(result, promo_policy, date_col, country_col, brand_col)
    else:
        result['ext_promo_active'] = 0
        result['ext_policy_change'] = 0
    
    # Fill missing values
    if fill_missing:
        result = _fill_external_missing(result)
    
    return result


def _join_holidays(
    panel: pd.DataFrame,
    holidays: pd.DataFrame,
    date_col: str,
    country_col: str,
    aggregate: str,
) -> pd.DataFrame:
    """Join holiday data to panel."""
    # Extract year-month from panel dates
    if date_col in panel.columns:
        panel['_year_month'] = panel[date_col].dt.to_period('M')
    else:
        # If no date column, return with empty holiday columns
        panel['ext_n_holidays'] = 0
        panel['ext_is_holiday_month'] = 0
        return panel
    
    # Extract year-month from holidays
    holidays = holidays.copy()
    holidays['_year_month'] = holidays['date'].dt.to_period('M')
    
    # Aggregate holidays per country-month
    holiday_agg = holidays.groupby([holidays['country'], '_year_month']).agg(
        n_holidays=('holiday_name', 'count')
    ).reset_index()
    holiday_agg.columns = ['country', '_year_month', 'ext_n_holidays']
    
    # Merge
    panel = panel.merge(
        holiday_agg,
        left_on=[country_col, '_year_month'],
        right_on=['country', '_year_month'],
        how='left',
        suffixes=('', '_hol')
    )
    
    # Fill NaN with 0 (no holidays)
    panel['ext_n_holidays'] = panel['ext_n_holidays'].fillna(0).astype(int)
    panel['ext_is_holiday_month'] = (panel['ext_n_holidays'] > 0).astype(int)
    
    # Clean up
    panel = panel.drop(columns=['_year_month'], errors='ignore')
    if 'country_hol' in panel.columns:
        panel = panel.drop(columns=['country_hol'], errors='ignore')
    
    return panel


def _join_epidemics(
    panel: pd.DataFrame,
    epidemics: pd.DataFrame,
    date_col: str,
    country_col: str,
) -> pd.DataFrame:
    """Join epidemic event data to panel."""
    panel = panel.copy()
    panel['ext_epidemic_severity'] = 0
    panel['ext_epidemic_active'] = 0
    
    if date_col not in panel.columns:
        return panel
    
    # For each row, check if any epidemic is active
    for idx, event in epidemics.iterrows():
        start = event['start_date']
        end = event['end_date'] if pd.notna(event['end_date']) else pd.Timestamp.now()
        severity = event['severity']
        country = event['country']
        
        # Find matching rows
        mask = (
            (panel[country_col] == country) &
            (panel[date_col] >= start) &
            (panel[date_col] <= end)
        )
        
        # Update severity (take max)
        panel.loc[mask, 'ext_epidemic_severity'] = np.maximum(
            panel.loc[mask, 'ext_epidemic_severity'],
            severity
        )
        panel.loc[mask, 'ext_epidemic_active'] = 1
    
    return panel


def _join_macro_indicators(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    date_col: str,
    country_col: str,
) -> pd.DataFrame:
    """Join macroeconomic indicator data to panel."""
    panel = panel.copy()
    
    # Initialize columns
    indicator_cols = {
        'gdp_growth': 'ext_gdp_growth',
        'unemployment': 'ext_unemployment', 
        'inflation': 'ext_inflation',
        'healthcare_spend': 'ext_healthcare_spend'
    }
    
    for col in indicator_cols.values():
        panel[col] = np.nan
    
    if date_col not in panel.columns:
        return panel
    
    # Pivot macro indicators to wide format
    for indicator_name, panel_col in indicator_cols.items():
        indicator_df = macro[macro['indicator_name'] == indicator_name].copy()
        if len(indicator_df) == 0:
            continue
        
        # Extract year-month
        indicator_df['_year_month'] = indicator_df['date'].dt.to_period('M')
        
        # Get latest value per country-month
        indicator_df = indicator_df.sort_values('date').groupby(
            ['country', '_year_month']
        ).last().reset_index()
        
        # Panel year-month
        panel['_year_month'] = panel[date_col].dt.to_period('M')
        
        # Merge
        panel = panel.merge(
            indicator_df[['country', '_year_month', 'value']],
            left_on=[country_col, '_year_month'],
            right_on=['country', '_year_month'],
            how='left',
            suffixes=('', '_macro')
        )
        
        panel[panel_col] = panel['value'].fillna(panel[panel_col])
        panel = panel.drop(columns=['value', '_year_month'], errors='ignore')
        if 'country_macro' in panel.columns:
            panel = panel.drop(columns=['country_macro'], errors='ignore')
    
    # Clean up any remaining _year_month column
    panel = panel.drop(columns=['_year_month'], errors='ignore')
    
    return panel


def _join_promo_policy(
    panel: pd.DataFrame,
    promo_policy: pd.DataFrame,
    date_col: str,
    country_col: str,
    brand_col: str,
) -> pd.DataFrame:
    """Join promotional and policy event data to panel."""
    panel = panel.copy()
    panel['ext_promo_active'] = 0
    panel['ext_policy_change'] = 0
    
    if date_col not in panel.columns:
        return panel
    
    promo_types = {'price_cut', 'marketing_campaign', 'competitor_launch'}
    policy_types = {'patent_expiry', 'formulary_change', 'regulatory_approval'}
    
    for idx, event in promo_policy.iterrows():
        start = event['start_date']
        end = event['end_date'] if pd.notna(event['end_date']) else start + pd.DateOffset(months=1)
        event_type = event['event_type']
        country = event['country']
        affected_brands = event.get('affected_brands', '')
        
        # Build mask
        mask = (
            (panel[country_col] == country) &
            (panel[date_col] >= start) &
            (panel[date_col] <= end)
        )
        
        # Filter by affected brands if specified
        if affected_brands and pd.notna(affected_brands) and affected_brands != '':
            brand_list = [b.strip() for b in str(affected_brands).split('|')]
            if brand_col in panel.columns:
                mask = mask & panel[brand_col].isin(brand_list)
        
        # Update flags
        if event_type in promo_types:
            panel.loc[mask, 'ext_promo_active'] = 1
        elif event_type in policy_types:
            panel.loc[mask, 'ext_policy_change'] = 1
    
    return panel


def _fill_external_missing(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill missing external context values with defaults."""
    panel = panel.copy()
    
    # Fill binary/count columns with 0
    binary_cols = [
        'ext_n_holidays', 'ext_is_holiday_month',
        'ext_epidemic_severity', 'ext_epidemic_active',
        'ext_promo_active', 'ext_policy_change'
    ]
    
    for col in binary_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)
    
    # For macro indicators, forward fill within country, then use global mean
    macro_cols = ['ext_gdp_growth', 'ext_unemployment', 'ext_inflation', 'ext_healthcare_spend']
    
    if 'country' in panel.columns:
        for col in macro_cols:
            if col in panel.columns:
                # Forward fill within country (using transform with ffill)
                panel[col] = panel.groupby('country')[col].transform(lambda x: x.ffill())
                # Backward fill within country (using transform with bfill)
                panel[col] = panel.groupby('country')[col].transform(lambda x: x.bfill())
                # Fill remaining with global mean
                panel[col] = panel[col].fillna(panel[col].mean())
    
    return panel


# =============================================================================
# Utility Functions
# =============================================================================

def create_external_feature_names() -> List[str]:
    """Return list of all external context feature names."""
    return [
        'ext_n_holidays',
        'ext_is_holiday_month',
        'ext_epidemic_severity',
        'ext_epidemic_active',
        'ext_gdp_growth',
        'ext_unemployment',
        'ext_inflation',
        'ext_healthcare_spend',
        'ext_promo_active',
        'ext_policy_change',
    ]


def validate_external_data(
    holidays: Optional[pd.DataFrame] = None,
    epidemics: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    promo_policy: Optional[pd.DataFrame] = None,
) -> Dict[str, bool]:
    """
    Validate external data sources.
    
    Returns:
        Dict mapping source name to validity status.
    """
    results = {}
    
    # Validate holidays
    if holidays is not None:
        required_cols = {'date', 'country'}
        results['holidays'] = required_cols.issubset(set(holidays.columns))
    else:
        results['holidays'] = True  # None is valid (optional)
    
    # Validate epidemics
    if epidemics is not None:
        required_cols = {'start_date', 'country', 'event_name'}
        results['epidemics'] = required_cols.issubset(set(epidemics.columns))
    else:
        results['epidemics'] = True
    
    # Validate macro
    if macro is not None:
        required_cols = {'date', 'country', 'indicator_name', 'value'}
        results['macro'] = required_cols.issubset(set(macro.columns))
    else:
        results['macro'] = True
    
    # Validate promo_policy
    if promo_policy is not None:
        required_cols = {'start_date', 'country', 'event_type'}
        results['promo_policy'] = required_cols.issubset(set(promo_policy.columns))
    else:
        results['promo_policy'] = True
    
    return results


def get_external_data_summary(
    holidays: Optional[pd.DataFrame] = None,
    epidemics: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    promo_policy: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Get summary statistics for external data sources.
    
    Returns:
        Dict with summary statistics per source.
    """
    summary = {}
    
    if holidays is not None and len(holidays) > 0:
        summary['holidays'] = {
            'n_records': len(holidays),
            'n_countries': holidays['country'].nunique(),
            'date_range': (holidays['date'].min(), holidays['date'].max()),
            'holiday_types': holidays.get('holiday_type', pd.Series()).unique().tolist()
        }
    
    if epidemics is not None and len(epidemics) > 0:
        summary['epidemics'] = {
            'n_events': len(epidemics),
            'n_countries': epidemics['country'].nunique(),
            'severity_range': (epidemics['severity'].min(), epidemics['severity'].max()),
            'event_names': epidemics['event_name'].unique().tolist()[:10]  # Top 10
        }
    
    if macro is not None and len(macro) > 0:
        summary['macro'] = {
            'n_records': len(macro),
            'n_countries': macro['country'].nunique(),
            'indicators': macro['indicator_name'].unique().tolist(),
            'date_range': (macro['date'].min(), macro['date'].max())
        }
    
    if promo_policy is not None and len(promo_policy) > 0:
        summary['promo_policy'] = {
            'n_events': len(promo_policy),
            'n_countries': promo_policy['country'].nunique(),
            'event_types': promo_policy['event_type'].unique().tolist()
        }
    
    return summary
