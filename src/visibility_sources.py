"""
Visibility sources for supply-chain data integration.

This module provides a protocol and implementations for loading supply-chain
visibility data from various sources (CSV files, APIs, databases).

Based on research from:
- Ghannem et al. (2023): Supply-chain visibility and demand forecasting

Visibility metrics include:
- Inventory levels at various points in the supply chain
- Order fill rates and lead times
- Supplier reliability scores
- Distribution center capacity utilization
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Visibility Data Schemas
# =============================================================================

@dataclass
class InventoryRecord:
    """Canonical schema for inventory visibility data."""
    date: date
    country: str
    brand_name: str
    location_type: str  # "manufacturer", "distributor", "hospital", "pharmacy"
    inventory_units: float
    days_of_supply: float
    stock_out_risk: float  # 0-1 probability
    
    
@dataclass
class OrderRecord:
    """Canonical schema for order visibility data."""
    date: date
    country: str
    brand_name: str
    order_quantity: float
    filled_quantity: float
    lead_time_days: int
    order_type: str  # "replenishment", "emergency", "scheduled"


@dataclass
class SupplierRecord:
    """Canonical schema for supplier visibility data."""
    date: date
    country: str
    brand_name: str
    supplier_id: str
    reliability_score: float  # 0-1
    on_time_delivery_rate: float  # 0-1
    quality_score: float  # 0-1


@dataclass
class DistributionRecord:
    """Canonical schema for distribution visibility data."""
    date: date
    country: str
    distribution_center_id: str
    capacity_utilization: float  # 0-1
    throughput_units: float
    backlog_units: float


# =============================================================================
# Visibility Source Protocol
# =============================================================================

class VisibilitySource(Protocol):
    """
    Protocol for visibility data sources.
    
    Implementations should provide methods to load various types
    of supply-chain visibility data.
    """
    
    def load_inventory(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load inventory visibility data."""
        ...
    
    def load_orders(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load order visibility data."""
        ...
    
    def load_supplier_metrics(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load supplier reliability metrics."""
        ...
    
    def load_distribution_metrics(
        self,
        countries: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load distribution center metrics."""
        ...
    
    def get_aggregated_visibility(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        aggregation_period: str = 'M',
    ) -> pd.DataFrame:
        """Get aggregated visibility features."""
        ...


# =============================================================================
# CSV Visibility Source Implementation
# =============================================================================

class CsvVisibilitySource:
    """
    CSV file-based implementation of VisibilitySource.
    
    Expects CSV files in a specified directory with the following structure:
    - inventory.csv: Inventory visibility data
    - orders.csv: Order visibility data
    - suppliers.csv: Supplier metrics
    - distribution.csv: Distribution center metrics
    
    Args:
        base_path: Path to directory containing visibility CSV files.
        config: Optional configuration dict with file names and column mappings.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_path = Path(base_path)
        self.config = config or {}
        
        # Default file names
        self._file_names = {
            'inventory': self.config.get('inventory_file', 'inventory.csv'),
            'orders': self.config.get('orders_file', 'orders.csv'),
            'suppliers': self.config.get('suppliers_file', 'suppliers.csv'),
            'distribution': self.config.get('distribution_file', 'distribution.csv'),
        }
        
        # Column mappings (can be overridden in config)
        self._column_mappings = self.config.get('column_mappings', {})
    
    def _get_file_path(self, file_type: str) -> Path:
        """Get full path for a file type."""
        return self.base_path / self._file_names[file_type]
    
    def _load_csv(
        self,
        file_type: str,
        required_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load and validate a CSV file.
        
        Args:
            file_type: Type of file to load.
            required_cols: Required columns to validate.
            
        Returns:
            Loaded DataFrame or empty DataFrame if file not found.
        """
        path = self._get_file_path(file_type)
        
        if not path.exists():
            logger.warning(f"Visibility file not found: {path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path)
            
            # Apply column mappings
            if file_type in self._column_mappings:
                df = df.rename(columns=self._column_mappings[file_type])
            
            # Validate required columns
            if required_cols:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    logger.warning(f"Missing columns in {file_type}: {missing}")
                    return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} records from {path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_type} file: {e}")
            return pd.DataFrame()
    
    def load_inventory(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load inventory visibility data.
        
        Returns:
            DataFrame with columns: date, country, brand_name, location_type,
                                   inventory_units, days_of_supply, stock_out_risk
        """
        schema_cols = ['date', 'country', 'brand_name', 'location_type',
                      'inventory_units', 'days_of_supply', 'stock_out_risk']
        
        df = self._load_csv('inventory', ['date', 'country', 'brand_name'])
        
        if df.empty:
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['date'] = pd.to_datetime(df['date'])
        df['country'] = df['country'].str.upper()
        
        # Add missing columns with defaults
        if 'location_type' not in df.columns:
            df['location_type'] = 'distributor'
        if 'inventory_units' not in df.columns:
            df['inventory_units'] = np.nan
        if 'days_of_supply' not in df.columns:
            df['days_of_supply'] = np.nan
        if 'stock_out_risk' not in df.columns:
            df['stock_out_risk'] = np.nan
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if brands:
            df = df[df['brand_name'].isin(brands)]
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        return df[schema_cols]
    
    def load_orders(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load order visibility data.
        
        Returns:
            DataFrame with columns: date, country, brand_name, order_quantity,
                                   filled_quantity, lead_time_days, order_type
        """
        schema_cols = ['date', 'country', 'brand_name', 'order_quantity',
                      'filled_quantity', 'lead_time_days', 'order_type']
        
        df = self._load_csv('orders', ['date', 'country', 'brand_name'])
        
        if df.empty:
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        df['date'] = pd.to_datetime(df['date'])
        df['country'] = df['country'].str.upper()
        
        # Add missing columns with defaults
        if 'order_quantity' not in df.columns:
            df['order_quantity'] = np.nan
        if 'filled_quantity' not in df.columns:
            df['filled_quantity'] = df.get('order_quantity', np.nan)
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = np.nan
        if 'order_type' not in df.columns:
            df['order_type'] = 'replenishment'
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if brands:
            df = df[df['brand_name'].isin(brands)]
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        return df[schema_cols]
    
    def load_supplier_metrics(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load supplier reliability metrics.
        
        Returns:
            DataFrame with columns: date, country, brand_name, supplier_id,
                                   reliability_score, on_time_delivery_rate, quality_score
        """
        schema_cols = ['date', 'country', 'brand_name', 'supplier_id',
                      'reliability_score', 'on_time_delivery_rate', 'quality_score']
        
        df = self._load_csv('suppliers', ['country', 'brand_name'])
        
        if df.empty:
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df['date'] = pd.NaT
        
        df['country'] = df['country'].str.upper()
        
        # Add missing columns with defaults
        if 'supplier_id' not in df.columns:
            df['supplier_id'] = 'UNKNOWN'
        if 'reliability_score' not in df.columns:
            df['reliability_score'] = np.nan
        if 'on_time_delivery_rate' not in df.columns:
            df['on_time_delivery_rate'] = np.nan
        if 'quality_score' not in df.columns:
            df['quality_score'] = np.nan
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        if brands:
            df = df[df['brand_name'].isin(brands)]
        
        return df[schema_cols]
    
    def load_distribution_metrics(
        self,
        countries: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load distribution center metrics.
        
        Returns:
            DataFrame with columns: date, country, distribution_center_id,
                                   capacity_utilization, throughput_units, backlog_units
        """
        schema_cols = ['date', 'country', 'distribution_center_id',
                      'capacity_utilization', 'throughput_units', 'backlog_units']
        
        df = self._load_csv('distribution', ['country'])
        
        if df.empty:
            return pd.DataFrame(columns=schema_cols)
        
        # Normalize columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            df['date'] = pd.NaT
        
        df['country'] = df['country'].str.upper()
        
        # Add missing columns with defaults
        if 'distribution_center_id' not in df.columns:
            df['distribution_center_id'] = 'DC_UNKNOWN'
        if 'capacity_utilization' not in df.columns:
            df['capacity_utilization'] = np.nan
        if 'throughput_units' not in df.columns:
            df['throughput_units'] = np.nan
        if 'backlog_units' not in df.columns:
            df['backlog_units'] = np.nan
        
        # Apply filters
        if countries:
            countries_upper = [c.upper() for c in countries]
            df = df[df['country'].isin(countries_upper)]
        
        return df[schema_cols]
    
    def get_aggregated_visibility(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        aggregation_period: str = 'M',
    ) -> pd.DataFrame:
        """
        Get aggregated visibility features per country-brand-period.
        
        Args:
            countries: Filter to specific countries.
            brands: Filter to specific brands.
            aggregation_period: Pandas period string ('M' for month, 'W' for week).
            
        Returns:
            DataFrame with aggregated visibility features:
            - vis_avg_inventory: Average inventory units
            - vis_avg_days_of_supply: Average days of supply
            - vis_avg_stock_out_risk: Average stock-out risk
            - vis_fill_rate: Order fill rate (filled/ordered)
            - vis_avg_lead_time: Average lead time in days
            - vis_supplier_reliability: Average supplier reliability
            - vis_capacity_utilization: Average DC capacity utilization
        """
        # Load all data
        inventory = self.load_inventory(countries, brands)
        orders = self.load_orders(countries, brands)
        suppliers = self.load_supplier_metrics(countries, brands)
        distribution = self.load_distribution_metrics(countries)
        
        # Initialize result DataFrame
        result_dfs = []
        
        # Aggregate inventory
        if not inventory.empty:
            inventory['period'] = inventory['date'].dt.to_period(aggregation_period)
            inv_agg = inventory.groupby(['country', 'brand_name', 'period']).agg({
                'inventory_units': 'mean',
                'days_of_supply': 'mean',
                'stock_out_risk': 'mean'
            }).reset_index()
            inv_agg.columns = ['country', 'brand_name', 'period',
                              'vis_avg_inventory', 'vis_avg_days_of_supply', 'vis_avg_stock_out_risk']
            result_dfs.append(inv_agg)
        
        # Aggregate orders
        if not orders.empty:
            orders['period'] = orders['date'].dt.to_period(aggregation_period)
            ord_agg = orders.groupby(['country', 'brand_name', 'period']).agg({
                'order_quantity': 'sum',
                'filled_quantity': 'sum',
                'lead_time_days': 'mean'
            }).reset_index()
            ord_agg['vis_fill_rate'] = ord_agg['filled_quantity'] / ord_agg['order_quantity'].replace(0, np.nan)
            ord_agg['vis_avg_lead_time'] = ord_agg['lead_time_days']
            ord_agg = ord_agg[['country', 'brand_name', 'period', 'vis_fill_rate', 'vis_avg_lead_time']]
            result_dfs.append(ord_agg)
        
        # Aggregate suppliers
        if not suppliers.empty and 'date' in suppliers.columns and suppliers['date'].notna().any():
            suppliers['period'] = suppliers['date'].dt.to_period(aggregation_period)
            sup_agg = suppliers.groupby(['country', 'brand_name', 'period']).agg({
                'reliability_score': 'mean',
                'on_time_delivery_rate': 'mean'
            }).reset_index()
            sup_agg.columns = ['country', 'brand_name', 'period',
                              'vis_supplier_reliability', 'vis_on_time_delivery']
            result_dfs.append(sup_agg)
        
        # Aggregate distribution (country-level only)
        if not distribution.empty and 'date' in distribution.columns and distribution['date'].notna().any():
            distribution['period'] = distribution['date'].dt.to_period(aggregation_period)
            dist_agg = distribution.groupby(['country', 'period']).agg({
                'capacity_utilization': 'mean'
            }).reset_index()
            dist_agg.columns = ['country', 'period', 'vis_capacity_utilization']
            result_dfs.append(dist_agg)
        
        # Merge all aggregations
        if not result_dfs:
            return pd.DataFrame(columns=['country', 'brand_name', 'period'])
        
        result = result_dfs[0]
        for df in result_dfs[1:]:
            if 'brand_name' in df.columns:
                result = result.merge(df, on=['country', 'brand_name', 'period'], how='outer')
            else:
                result = result.merge(df, on=['country', 'period'], how='left')
        
        return result


# =============================================================================
# Mock Visibility Source for Testing
# =============================================================================

class MockVisibilitySource:
    """
    Mock visibility source for testing.
    
    Generates synthetic visibility data based on patterns.
    """
    
    def __init__(
        self,
        countries: List[str] = None,
        brands: List[str] = None,
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        seed: int = 42,
    ):
        self.countries = countries or ['US', 'DE', 'FR', 'UK']
        self.brands = brands or ['BRAND_A', 'BRAND_B', 'BRAND_C']
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rng = np.random.RandomState(seed)
    
    def load_inventory(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Generate synthetic inventory data."""
        countries = countries or self.countries
        brands = brands or self.brands
        start = pd.to_datetime(start_date) if start_date else self.start_date
        end = pd.to_datetime(end_date) if end_date else self.end_date
        
        dates = pd.date_range(start, end, freq='M')
        records = []
        
        for country in countries:
            for brand in brands:
                for dt in dates:
                    records.append({
                        'date': dt,
                        'country': country,
                        'brand_name': brand,
                        'location_type': self.rng.choice(['manufacturer', 'distributor', 'pharmacy']),
                        'inventory_units': max(0, 1000 + self.rng.normal(0, 200)),
                        'days_of_supply': max(5, 30 + self.rng.normal(0, 10)),
                        'stock_out_risk': np.clip(0.1 + self.rng.normal(0, 0.05), 0, 1),
                    })
        
        return pd.DataFrame(records)
    
    def load_orders(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Generate synthetic order data."""
        countries = countries or self.countries
        brands = brands or self.brands
        start = pd.to_datetime(start_date) if start_date else self.start_date
        end = pd.to_datetime(end_date) if end_date else self.end_date
        
        dates = pd.date_range(start, end, freq='M')
        records = []
        
        for country in countries:
            for brand in brands:
                for dt in dates:
                    order_qty = max(100, 500 + self.rng.normal(0, 100))
                    fill_rate = np.clip(0.95 + self.rng.normal(0, 0.02), 0.8, 1.0)
                    records.append({
                        'date': dt,
                        'country': country,
                        'brand_name': brand,
                        'order_quantity': order_qty,
                        'filled_quantity': order_qty * fill_rate,
                        'lead_time_days': max(1, int(7 + self.rng.normal(0, 2))),
                        'order_type': self.rng.choice(['replenishment', 'scheduled']),
                    })
        
        return pd.DataFrame(records)
    
    def load_supplier_metrics(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic supplier metrics."""
        countries = countries or self.countries
        brands = brands or self.brands
        
        records = []
        for country in countries:
            for brand in brands:
                records.append({
                    'date': self.end_date,  # Latest metrics
                    'country': country,
                    'brand_name': brand,
                    'supplier_id': f'SUP_{brand}_{country}',
                    'reliability_score': np.clip(0.9 + self.rng.normal(0, 0.05), 0.7, 1.0),
                    'on_time_delivery_rate': np.clip(0.92 + self.rng.normal(0, 0.03), 0.8, 1.0),
                    'quality_score': np.clip(0.95 + self.rng.normal(0, 0.02), 0.85, 1.0),
                })
        
        return pd.DataFrame(records)
    
    def load_distribution_metrics(
        self,
        countries: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic distribution metrics."""
        countries = countries or self.countries
        dates = pd.date_range(self.start_date, self.end_date, freq='M')
        
        records = []
        for country in countries:
            for dt in dates:
                records.append({
                    'date': dt,
                    'country': country,
                    'distribution_center_id': f'DC_{country}_01',
                    'capacity_utilization': np.clip(0.75 + self.rng.normal(0, 0.1), 0.4, 1.0),
                    'throughput_units': max(1000, 5000 + self.rng.normal(0, 1000)),
                    'backlog_units': max(0, 100 + self.rng.normal(0, 50)),
                })
        
        return pd.DataFrame(records)
    
    def get_aggregated_visibility(
        self,
        countries: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        aggregation_period: str = 'M',
    ) -> pd.DataFrame:
        """Get aggregated visibility features."""
        # Use same logic as CsvVisibilitySource
        csv_source = CsvVisibilitySource.__new__(CsvVisibilitySource)
        csv_source.base_path = Path('.')
        csv_source.config = {}
        
        # Override load methods to use mock data
        inventory = self.load_inventory(countries, brands)
        orders = self.load_orders(countries, brands)
        
        result_dfs = []
        
        # Aggregate inventory
        if not inventory.empty:
            inventory['period'] = inventory['date'].dt.to_period(aggregation_period)
            inv_agg = inventory.groupby(['country', 'brand_name', 'period']).agg({
                'inventory_units': 'mean',
                'days_of_supply': 'mean',
                'stock_out_risk': 'mean'
            }).reset_index()
            inv_agg.columns = ['country', 'brand_name', 'period',
                              'vis_avg_inventory', 'vis_avg_days_of_supply', 'vis_avg_stock_out_risk']
            result_dfs.append(inv_agg)
        
        # Aggregate orders
        if not orders.empty:
            orders['period'] = orders['date'].dt.to_period(aggregation_period)
            ord_agg = orders.groupby(['country', 'brand_name', 'period']).agg({
                'order_quantity': 'sum',
                'filled_quantity': 'sum',
                'lead_time_days': 'mean'
            }).reset_index()
            ord_agg['vis_fill_rate'] = ord_agg['filled_quantity'] / ord_agg['order_quantity'].replace(0, np.nan)
            ord_agg['vis_avg_lead_time'] = ord_agg['lead_time_days']
            ord_agg = ord_agg[['country', 'brand_name', 'period', 'vis_fill_rate', 'vis_avg_lead_time']]
            result_dfs.append(ord_agg)
        
        if not result_dfs:
            return pd.DataFrame()
        
        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = result.merge(df, on=['country', 'brand_name', 'period'], how='outer')
        
        return result


# =============================================================================
# Feature Extraction from Visibility Data
# =============================================================================

def create_visibility_feature_names() -> List[str]:
    """Return list of all visibility feature names."""
    return [
        'vis_avg_inventory',
        'vis_avg_days_of_supply',
        'vis_avg_stock_out_risk',
        'vis_fill_rate',
        'vis_avg_lead_time',
        'vis_supplier_reliability',
        'vis_on_time_delivery',
        'vis_capacity_utilization',
    ]


def join_visibility_features(
    panel: pd.DataFrame,
    visibility_source: Union[CsvVisibilitySource, MockVisibilitySource],
    date_col: str = 'month',
    country_col: str = 'country',
    brand_col: str = 'brand_name',
    fill_missing: bool = True,
) -> pd.DataFrame:
    """
    Join visibility features to the main panel.
    
    Args:
        panel: Main panel DataFrame.
        visibility_source: Visibility data source.
        date_col: Name of date column in panel.
        country_col: Name of country column in panel.
        brand_col: Name of brand column in panel.
        fill_missing: Fill missing values with defaults.
        
    Returns:
        Panel with additional visibility feature columns.
    """
    result = panel.copy()
    
    # Get aggregated visibility features
    visibility_agg = visibility_source.get_aggregated_visibility()
    
    if visibility_agg.empty:
        # Add empty columns
        for col in create_visibility_feature_names():
            result[col] = np.nan
        return result
    
    # Convert period to datetime for joining
    if date_col in result.columns:
        result['_period'] = pd.to_datetime(result[date_col]).dt.to_period('M')
    else:
        logger.warning(f"Date column '{date_col}' not found in panel")
        for col in create_visibility_feature_names():
            result[col] = np.nan
        return result
    
    # Merge
    result = result.merge(
        visibility_agg,
        left_on=[country_col, brand_col, '_period'],
        right_on=['country', 'brand_name', 'period'],
        how='left',
        suffixes=('', '_vis')
    )
    
    # Clean up
    result = result.drop(columns=['_period', 'period'], errors='ignore')
    if 'country_vis' in result.columns:
        result = result.drop(columns=['country_vis'], errors='ignore')
    if 'brand_name_vis' in result.columns:
        result = result.drop(columns=['brand_name_vis'], errors='ignore')
    
    # Fill missing values
    if fill_missing:
        for col in create_visibility_feature_names():
            if col in result.columns:
                # Fill with median
                result[col] = result[col].fillna(result[col].median())
                # Final fallback to 0
                result[col] = result[col].fillna(0)
    
    return result
