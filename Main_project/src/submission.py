# =============================================================================
# File: src/submission.py
# Description: Functions to generate and validate submission files
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def generate_submission(predictions_df: pd.DataFrame,
                        scenario: int,
                        validate: bool = True) -> pd.DataFrame:
    """
    Generate submission file from predictions.
    
    Args:
        predictions_df: DataFrame with columns [country, brand_name, months_postgx, volume]
        scenario: 1 or 2
        validate: If True, validate submission before returning
        
    Returns:
        Submission DataFrame
    """
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    missing = set(required_cols) - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions: {missing}")
    
    submission = predictions_df[required_cols].copy()
    
    # Ensure no negative volumes
    submission['volume'] = submission['volume'].clip(lower=0)
    
    if validate:
        validate_submission(submission, scenario)
    
    return submission


def validate_submission(submission_df: pd.DataFrame, scenario: int) -> bool:
    """
    Validate submission format and content.
    
    Args:
        submission_df: Submission DataFrame
        scenario: 1 or 2
        
    Returns:
        True if valid
        
    Raises:
        AssertionError if validation fails
    """
    print(f"\n🔍 Validating Scenario {scenario} submission...")
    
    # Check required columns
    required_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    assert all(col in submission_df.columns for col in required_cols), \
        f"Missing required columns. Expected: {required_cols}"
    
    # Check no missing values
    assert submission_df['volume'].notna().all(), \
        "Found NaN values in volume predictions"
    
    # Check no negative volumes
    assert (submission_df['volume'] >= 0).all(), \
        "Found negative volume predictions"
    
    # Check months_postgx range
    if scenario == 1:
        expected_months = set(range(0, 24))  # 0-23
        expected_rows_per_brand = 24
    else:
        expected_months = set(range(6, 24))  # 6-23
        expected_rows_per_brand = 18
    
    # Check each brand has correct months
    errors = []
    for (country, brand), group in submission_df.groupby(['country', 'brand_name']):
        actual_months = set(group['months_postgx'].values)
        if actual_months != expected_months:
            missing = expected_months - actual_months
            extra = actual_months - expected_months
            errors.append(f"{country}/{brand}: missing={missing}, extra={extra}")
    
    if errors:
        print(f"❌ Found {len(errors)} brands with wrong months:")
        for err in errors[:5]:
            print(f"   {err}")
        raise AssertionError(f"Found {len(errors)} brands with incorrect months_postgx")
    
    # Check total row count
    n_brands = submission_df[['country', 'brand_name']].drop_duplicates().shape[0]
    expected_total = n_brands * expected_rows_per_brand
    actual_total = len(submission_df)
    
    assert actual_total == expected_total, \
        f"Wrong total rows. Expected {expected_total} ({n_brands} brands × {expected_rows_per_brand} months), got {actual_total}"
    
    print(f"✅ Submission validation passed!")
    print(f"   Brands: {n_brands}")
    print(f"   Total rows: {actual_total}")
    print(f"   Volume range: [{submission_df['volume'].min():.2f}, {submission_df['volume'].max():.2f}]")
    
    return True


def save_submission(submission_df: pd.DataFrame,
                    scenario: int,
                    suffix: str = "",
                    include_timestamp: bool = True) -> Path:
    """
    Save submission to file.
    
    Args:
        submission_df: Validated submission DataFrame
        scenario: 1 or 2
        suffix: Optional suffix for filename
        include_timestamp: If True, add timestamp to filename
        
    Returns:
        Path to saved file
    """
    # Build filename
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scenario{scenario}_{timestamp}"
    else:
        filename = f"scenario{scenario}"
    
    if suffix:
        filename = f"{filename}_{suffix}"
    
    filename = f"{filename}.csv"
    filepath = SUBMISSIONS_DIR / filename
    
    # Save
    submission_df.to_csv(filepath, index=False)
    print(f"✅ Submission saved to: {filepath}")
    
    return filepath


def load_submission(filepath: Path) -> pd.DataFrame:
    """Load a submission file."""
    return pd.read_csv(filepath)


def compare_submissions(sub1_path: Path, sub2_path: Path) -> pd.DataFrame:
    """
    Compare two submission files.
    
    Args:
        sub1_path: Path to first submission
        sub2_path: Path to second submission
        
    Returns:
        DataFrame with differences
    """
    sub1 = load_submission(sub1_path)
    sub2 = load_submission(sub2_path)
    
    merged = sub1.merge(
        sub2, on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_1', '_2')
    )
    
    merged['diff'] = merged['volume_1'] - merged['volume_2']
    merged['diff_pct'] = merged['diff'] / merged['volume_1'] * 100
    
    print(f"\n📊 Submission Comparison:")
    print(f"   Mean absolute diff: {merged['diff'].abs().mean():.4f}")
    print(f"   Mean % diff: {merged['diff_pct'].abs().mean():.2f}%")
    print(f"   Max diff: {merged['diff'].abs().max():.4f}")
    
    return merged


if __name__ == "__main__":
    # Demo: Generate a test submission
    print("=" * 60)
    print("SUBMISSION DEMO")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets
    from bucket_calculator import compute_avg_j
    from models import BaselineModels
    
    # Load test data
    volume_test, generics_test, medicine_test = load_all_data(train=False)
    
    # Get unique test brands
    test_brands = volume_test[['country', 'brand_name']].drop_duplicates()
    print(f"Test brands: {len(test_brands)}")
    
    # Load training data to compute avg_j
    volume_train, generics_train, medicine_train = load_all_data(train=True)
    merged_train = merge_datasets(volume_train, generics_train, medicine_train)
    avg_j = compute_avg_j(merged_train)
    
    # Merge with test brands
    test_avg_j = test_brands.merge(avg_j, on=['country', 'brand_name'], how='left')
    
    # Fill missing avg_vol with median
    median_avg_vol = test_avg_j['avg_vol'].median()
    test_avg_j['avg_vol'] = test_avg_j['avg_vol'].fillna(median_avg_vol)
    
    # Generate predictions for Scenario 1 (months 0-23)
    pred_s1 = BaselineModels.exponential_decay(
        test_avg_j,
        months_to_predict=list(range(0, 24)),
        decay_rate=0.05
    )
    
    # Generate and save submission
    submission_s1 = generate_submission(pred_s1, scenario=1)
    save_submission(submission_s1, scenario=1, suffix="baseline_exp_decay")
    
    # Generate predictions for Scenario 2 (months 6-23)
    pred_s2 = BaselineModels.exponential_decay(
        test_avg_j,
        months_to_predict=list(range(6, 24)),
        decay_rate=0.05
    )
    
    submission_s2 = generate_submission(pred_s2, scenario=2)
    save_submission(submission_s2, scenario=2, suffix="baseline_exp_decay")
    
    print("\n✅ Submission demo complete!")
