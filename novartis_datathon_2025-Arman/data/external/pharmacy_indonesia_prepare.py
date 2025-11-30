#!/usr/bin/env python3
"""
Pharmacy Indonesia Dataset Preparation Script
==============================================

This script converts the Indonesian pharmacy dataset into Novartis-like tables
so that the same feature pipeline and model code can be reused.

HOW TO TAKE ADVANTAGE OF THIS:
------------------------------
These three output files can be read by the **same data loading and feature pipeline**
as the Novartis dataset (maybe with a `dataset_id` flag):
  - df_volume_pharmacy_train.csv
  - df_generics_pharmacy_train.csv
  - df_medicine_info_pharmacy_train.csv

They allow you to:
  1. Test the end-to-end pipeline (features, model, CV) on a large real dataset.
  2. Run multi-dataset experiments by adding `COUNTRY_PHARMACY_ID` as another country in the panel.

IMPORTANT NOTES:
----------------
- `months_postgx` is NOT based on real LOE (loss of exclusivity) dates.
  For this external dataset, it is simply months since the product's **first appearance** in 2015.
  It is a proxy to re-use the same pipeline, not a real LOE.
- `n_gxs` is set to 1.0 as a dummy value for schema compatibility.
- `ther_area` uses manufacturer code as a placeholder.
- `biological` / `small_molecule` are naive assumptions for schema compatibility.

Usage:
    python pharmacy_indonesia_prepare.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
EXTERNAL_ROOT = Path("/Users/armanfeili/code/New Projects/novartis_datathon_2025/data/external")
RAW_ROOT = EXTERNAL_ROOT / "pharmacy_indonesia_raw"
PROCESSED_ROOT = EXTERNAL_ROOT / "pharmacy_indonesia_processed"
NOVARTIS_LIKE_ROOT = EXTERNAL_ROOT / "pharmacy_indonesia_novartis_like"

# Constants
COUNTRY_ID = "COUNTRY_PHARMACY_ID"


# ============================================================================
# STEP 3: LOAD RAW DATA
# ============================================================================

def load_raw_tables() -> Dict[str, pd.DataFrame]:
    """
    Load the raw CSV files from the pharmacy_indonesia_raw folder.
    
    Returns:
        Dictionary containing all raw DataFrames with proper dtypes.
    """
    print("Loading raw tables...")
    
    # Load det_sales
    det_sales = pd.read_csv(
        RAW_ROOT / "det_sales.csv",
        dtype={
            "NO_RESEP": str,
            "KD_OBAT": str,
            "QTY": float,
            "HNA": float,
            "HJ": float,
            "PPN_JUAL": float,
        }
    )
    print(f"  det_sales: {len(det_sales):,} rows")
    
    # Load ms_sales
    ms_sales = pd.read_csv(
        RAW_ROOT / "ms_sales.csv",
        dtype={
            "NO_RESEP": str,
            "KD_CUST": str,
            "KD_DOKTER": str,
            "REG_AS": str,
            "JAM_JUAL": str,
            "RACIK": str,
        },
        parse_dates=["TGL"]
    )
    print(f"  ms_sales: {len(ms_sales):,} rows")
    
    # Load transaction
    transaction = pd.read_csv(
        RAW_ROOT / "transaction.csv",
        dtype={
            "NO_RESEP": str,
            "KD_CUST": str,
            "KD_OBAT": str,
            "QTY": float,
            "HNA": float,
            "HJ": float,
        },
        parse_dates=["TGL"]
    )
    print(f"  transaction: {len(transaction):,} rows")
    
    # Load ms_product
    ms_product = pd.read_csv(
        RAW_ROOT / "ms_product.csv",
        dtype={
            "KD_OBAT": str,
            "NAMA": str,
            "SAT_JUAL": str,
            "KD_PABRIK": str,
            "HJ_RP": float,
        }
    )
    print(f"  ms_product: {len(ms_product):,} rows")
    
    return {
        "det_sales": det_sales,
        "ms_sales": ms_sales,
        "transaction": transaction,
        "ms_product": ms_product,
    }


# ============================================================================
# STEP 4: BUILD INTEGRATED TRANSACTION TABLE
# ============================================================================

def build_integrated_transactions(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build an integrated transaction table by joining transaction with ms_product.
    
    Args:
        tables: Dictionary of raw DataFrames.
        
    Returns:
        Integrated transactions DataFrame with product attributes.
    """
    print("\nBuilding integrated transactions...")
    
    transaction = tables["transaction"].copy()
    ms_product = tables["ms_product"].copy()
    
    # Left-join product info
    integrated = transaction.merge(
        ms_product[["KD_OBAT", "NAMA", "SAT_JUAL", "KD_PABRIK", "HJ_RP"]],
        on="KD_OBAT",
        how="left"
    )
    
    print(f"  Integrated transactions: {len(integrated):,} rows")
    print(f"  Products with info: {integrated['NAMA'].notna().sum():,} ({integrated['NAMA'].notna().mean()*100:.1f}%)")
    
    # Ensure TGL is datetime
    integrated["TGL"] = pd.to_datetime(integrated["TGL"])
    
    # Save for reference
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_ROOT / "transactions_integrated.parquet"
    integrated.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return integrated


# ============================================================================
# STEP 5: BUILD MONTHLY PRODUCT-LEVEL PANEL
# ============================================================================

def build_monthly_panel(transactions_integrated: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly product-level panel from integrated transactions.
    
    Args:
        transactions_integrated: Integrated transactions DataFrame.
        
    Returns:
        Monthly panel DataFrame with volume and other aggregations.
    """
    print("\nBuilding monthly product panel...")
    
    df = transactions_integrated.copy()
    
    # Add month column (first day of month)
    df["TGL"] = pd.to_datetime(df["TGL"])
    df["month"] = df["TGL"].dt.to_period("M").dt.to_timestamp()
    
    # Define canonical columns
    df["country"] = COUNTRY_ID
    df["brand_name"] = df["KD_OBAT"]  # Treat KD_OBAT as brand_name
    
    # Calculate revenue and cost per transaction line
    df["line_revenue"] = df["QTY"] * df["HJ"]
    df["line_cost"] = df["QTY"] * df["HNA"]
    
    # Group to monthly product level
    monthly_panel = df.groupby(["country", "brand_name", "month"]).agg(
        volume=("QTY", "sum"),
        revenue=("line_revenue", "sum"),
        cost=("line_cost", "sum"),
        n_tx=("NO_RESEP", "nunique"),
    ).reset_index()
    
    # Calculate derived metrics
    monthly_panel["margin_abs"] = monthly_panel["revenue"] - monthly_panel["cost"]
    monthly_panel["avg_price"] = np.where(
        monthly_panel["volume"] > 0,
        monthly_panel["revenue"] / monthly_panel["volume"],
        0
    )
    
    print(f"  Monthly panel: {len(monthly_panel):,} rows")
    print(f"  Unique products: {monthly_panel['brand_name'].nunique():,}")
    print(f"  Date range: {monthly_panel['month'].min()} to {monthly_panel['month'].max()}")
    
    # Sort by country, brand_name, month
    monthly_panel = monthly_panel.sort_values(["country", "brand_name", "month"]).reset_index(drop=True)
    
    # Create months_since_first_obs (integer index for each product's timeline)
    # NOTE: For Novartis, `months_postgx` is months relative to LOE.
    # For this external dataset, `months_postgx` is simply months since the product's
    # **first appearance** in 2015. It is a proxy to re-use the same pipeline, not a real LOE.
    monthly_panel["months_since_first_obs"] = (
        monthly_panel.groupby(["country", "brand_name"])["month"]
        .rank(method="dense")
        .astype(int) - 1
    )
    
    # For this dataset, months_postgx = months_since_first_obs (synthetic proxy)
    monthly_panel["months_postgx"] = monthly_panel["months_since_first_obs"]
    
    # Create month_str column (e.g., "Jan", "Feb") to match Novartis format
    monthly_panel["month_str"] = monthly_panel["month"].dt.strftime("%b")
    
    # Save the monthly panel
    output_path = PROCESSED_ROOT / "monthly_panel.parquet"
    monthly_panel.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return monthly_panel


# ============================================================================
# STEP 6: CREATE df_volume-like CSV
# ============================================================================

def build_df_volume_like(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create a df_volume-like CSV matching the Novartis schema.
    
    Schema: country, brand_name, month, months_postgx, volume
    
    Args:
        monthly_panel: Monthly product panel DataFrame.
        
    Returns:
        df_volume-like DataFrame.
    """
    print("\nBuilding df_volume-like table...")
    
    df_volume_like = monthly_panel[[
        "country",
        "brand_name",
        "month_str",
        "months_postgx",
        "volume",
    ]].rename(columns={
        "month_str": "month",
    }).copy()
    
    # Ensure column order matches Novartis exactly
    df_volume_like = df_volume_like[["country", "brand_name", "month", "months_postgx", "volume"]]
    
    print(f"  df_volume_like: {len(df_volume_like):,} rows")
    
    # Save
    NOVARTIS_LIKE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = NOVARTIS_LIKE_ROOT / "df_volume_pharmacy_train.csv"
    df_volume_like.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df_volume_like


# ============================================================================
# STEP 7: CREATE df_generics-like CSV (SYNTHETIC)
# ============================================================================

def build_df_generics_like(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create a df_generics-like CSV (synthetic, for schema compatibility).
    
    Schema: country, brand_name, months_postgx, n_gxs
    
    NOTE: n_gxs is set to 1.0 as a dummy value. We do not have real LOE or
    generic competitor counts for this dataset. This is used only to keep
    the schema identical to the Novartis dataset.
    
    Args:
        monthly_panel: Monthly product panel DataFrame.
        
    Returns:
        df_generics-like DataFrame.
    """
    print("\nBuilding df_generics-like table (synthetic)...")
    
    # Get one row per (country, brand_name, months_postgx)
    df_generics_like = (
        monthly_panel[["country", "brand_name", "months_postgx"]]
        .drop_duplicates()
        .copy()
    )
    
    # Add synthetic n_gxs column
    # NOTE: This is a dummy field, used only to keep the schema identical.
    # Set n_gxs = 1.0 for all rows (meaning "1 product in the market").
    df_generics_like["n_gxs"] = 1.0
    
    # Ensure column order matches Novartis exactly
    df_generics_like = df_generics_like[["country", "brand_name", "months_postgx", "n_gxs"]]
    
    # Sort for consistency
    df_generics_like = df_generics_like.sort_values(
        ["country", "brand_name", "months_postgx"]
    ).reset_index(drop=True)
    
    print(f"  df_generics_like: {len(df_generics_like):,} rows")
    
    # Save
    output_path = NOVARTIS_LIKE_ROOT / "df_generics_pharmacy_train.csv"
    df_generics_like.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df_generics_like


# ============================================================================
# STEP 8: CREATE df_medicine_info-like CSV
# ============================================================================

def build_df_medicine_info_like(ms_product: pd.DataFrame) -> pd.DataFrame:
    """
    Create a df_medicine_info-like CSV (best effort mapping).
    
    Schema: country, brand_name, ther_area, hospital_rate, main_package, biological, small_molecule
    
    Mapping notes:
    - ther_area: Uses KD_PABRIK (manufacturer) as a placeholder.
    - hospital_rate: Set to NaN (unknown).
    - main_package: Mapped from SAT_JUAL.
    - biological/small_molecule: Naive assumptions for schema compatibility.
    
    Args:
        ms_product: Product master DataFrame.
        
    Returns:
        df_medicine_info-like DataFrame.
    """
    print("\nBuilding df_medicine_info-like table...")
    
    df_med_info = ms_product.copy()
    
    # Set country
    df_med_info["country"] = COUNTRY_ID
    
    # Map brand_name to KD_OBAT
    df_med_info["brand_name"] = df_med_info["KD_OBAT"]
    
    # Placeholder therapeutic area (using manufacturer code)
    # NOTE: This is only a placeholder - we don't have real therapeutic area info.
    df_med_info["ther_area"] = df_med_info["KD_PABRIK"].fillna("Unknown_ther_area")
    
    # Hospital rate unknown
    df_med_info["hospital_rate"] = np.nan
    
    # Map SAT_JUAL to main_package
    package_map = {
        "tab": "PILL",
        "tablet": "PILL",
        "kapsul": "PILL",
        "kaplet": "PILL",
        "suppos": "SUPPOSITORY",
        "amp": "INJECTION",
        "vial": "INJECTION",
        "injeksi": "INJECTION",
        "syr": "SYRUP",
        "sirup": "SYRUP",
        "susp": "SUSPENSION",
        "cream": "CREAM",
        "salep": "OINTMENT",
        "gel": "GEL",
        "drops": "EYE DROP",
        "tetes": "EYE DROP",
        "inhaler": "INHALER",
        "patch": "PATCH",
        "sachet": "SACHET",
        "powder": "POWDER",
        "serbuk": "POWDER",
    }
    
    # Normalize SAT_JUAL and map
    df_med_info["sat_jual_lower"] = df_med_info["SAT_JUAL"].fillna("").str.lower().str.strip()
    df_med_info["main_package"] = df_med_info["sat_jual_lower"].map(package_map).fillna("Others")
    
    # Biological / small_molecule
    # NOTE: Naive assumptions for schema compatibility - we don't have explicit info.
    df_med_info["biological"] = False
    df_med_info["small_molecule"] = True
    
    # Select and order columns to match Novartis schema
    df_medicine_info_like = df_med_info[[
        "country",
        "brand_name",
        "ther_area",
        "hospital_rate",
        "main_package",
        "biological",
        "small_molecule",
    ]].drop_duplicates().reset_index(drop=True)
    
    print(f"  df_medicine_info_like: {len(df_medicine_info_like):,} rows")
    print(f"  Package distribution:")
    print(df_medicine_info_like["main_package"].value_counts().head(10).to_string())
    
    # Save
    output_path = NOVARTIS_LIKE_ROOT / "df_medicine_info_pharmacy_train.csv"
    df_medicine_info_like.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df_medicine_info_like


# ============================================================================
# STEP 9: SANITY CHECKS / VALIDATION
# ============================================================================

def validate_outputs():
    """
    Validate the generated Novartis-like CSVs.
    """
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    # Load generated files
    df_volume = pd.read_csv(NOVARTIS_LIKE_ROOT / "df_volume_pharmacy_train.csv")
    df_generics = pd.read_csv(NOVARTIS_LIKE_ROOT / "df_generics_pharmacy_train.csv")
    df_medicine_info = pd.read_csv(NOVARTIS_LIKE_ROOT / "df_medicine_info_pharmacy_train.csv")
    
    # Expected columns
    expected_volume_cols = ["country", "brand_name", "month", "months_postgx", "volume"]
    expected_generics_cols = ["country", "brand_name", "months_postgx", "n_gxs"]
    expected_medicine_info_cols = ["country", "brand_name", "ther_area", "hospital_rate", 
                                    "main_package", "biological", "small_molecule"]
    
    print("\n--- df_volume_pharmacy_train.csv ---")
    print(f"Columns: {list(df_volume.columns)}")
    print(f"Expected: {expected_volume_cols}")
    print(f"Match: {list(df_volume.columns) == expected_volume_cols}")
    print(f"Shape: {df_volume.shape}")
    print(f"Missing in key columns: {df_volume[['country', 'brand_name', 'month', 'months_postgx', 'volume']].isna().sum().to_dict()}")
    print(df_volume.head().to_string())
    print(f"\nDtypes:\n{df_volume.dtypes}")
    
    print("\n--- df_generics_pharmacy_train.csv ---")
    print(f"Columns: {list(df_generics.columns)}")
    print(f"Expected: {expected_generics_cols}")
    print(f"Match: {list(df_generics.columns) == expected_generics_cols}")
    print(f"Shape: {df_generics.shape}")
    print(df_generics.head().to_string())
    print(f"\nDtypes:\n{df_generics.dtypes}")
    
    print("\n--- df_medicine_info_pharmacy_train.csv ---")
    print(f"Columns: {list(df_medicine_info.columns)}")
    print(f"Expected: {expected_medicine_info_cols}")
    print(f"Match: {list(df_medicine_info.columns) == expected_medicine_info_cols}")
    print(f"Shape: {df_medicine_info.shape}")
    print(df_medicine_info.head().to_string())
    print(f"\nDtypes:\n{df_medicine_info.dtypes}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ df_volume_pharmacy_train.csv: {len(df_volume):,} rows")
    print(f"✓ df_generics_pharmacy_train.csv: {len(df_generics):,} rows")
    print(f"✓ df_medicine_info_pharmacy_train.csv: {len(df_medicine_info):,} rows")
    print(f"\nOutput directory: {NOVARTIS_LIKE_ROOT}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main orchestration function.
    """
    print("="*60)
    print("PHARMACY INDONESIA DATASET PREPARATION")
    print("="*60)
    
    # Ensure output directories exist
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    NOVARTIS_LIKE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Load raw data
    tables = load_raw_tables()
    
    # Step 4: Build integrated transactions
    transactions_integrated = build_integrated_transactions(tables)
    
    # Step 5: Build monthly panel
    monthly_panel = build_monthly_panel(transactions_integrated)
    
    # Step 6: Create df_volume-like
    df_volume_like = build_df_volume_like(monthly_panel)
    
    # Step 7: Create df_generics-like (synthetic)
    df_generics_like = build_df_generics_like(monthly_panel)
    
    # Step 8: Create df_medicine_info-like
    df_medicine_info_like = build_df_medicine_info_like(tables["ms_product"])
    
    # Step 9: Validate outputs
    validate_outputs()
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
