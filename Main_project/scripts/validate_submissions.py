# =============================================================================
# File: scripts/validate_submissions.py
# Description: Validate submission files against the official template
#
# The submission must match the template EXACTLY:
#   - Same (country, brand_name, months_postgx) combinations
#   - 7,488 total rows
#   - 340 unique brands
#   - No NaN values in volume
# =============================================================================

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *


def load_template():
    """Load the official submission template."""
    template_path = SUBMISSIONS_DIR / "submission_template.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return pd.read_csv(template_path)


def validate_against_template(submission: pd.DataFrame, template: pd.DataFrame) -> tuple:
    """
    Validate submission matches template exactly.
    
    Returns:
        (is_valid, errors_list)
    """
    errors = []
    
    # Check columns
    expected_cols = ['country', 'brand_name', 'months_postgx', 'volume']
    if list(submission.columns) != expected_cols:
        errors.append(f"Column mismatch: got {list(submission.columns)}, expected {expected_cols}")
    
    # Check row count
    if len(submission) != len(template):
        errors.append(f"Row count mismatch: got {len(submission)}, expected {len(template)}")
    
    # Check for NaN in volume
    nan_count = submission['volume'].isna().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values in volume column")
    
    # Check for negative/zero volumes
    if (submission['volume'] <= 0).any():
        neg_count = (submission['volume'] <= 0).sum()
        errors.append(f"Found {neg_count} non-positive volumes (must be > 0)")
    
    # Check each (country, brand_name, months_postgx) combination
    template_keys = set(zip(template['country'], template['brand_name'], template['months_postgx']))
    submission_keys = set(zip(submission['country'], submission['brand_name'], submission['months_postgx']))
    
    missing_keys = template_keys - submission_keys
    extra_keys = submission_keys - template_keys
    
    if missing_keys:
        errors.append(f"Missing {len(missing_keys)} (country, brand, month) combinations from template")
        # Show first 5
        for i, key in enumerate(list(missing_keys)[:5]):
            errors.append(f"   Missing: {key}")
        if len(missing_keys) > 5:
            errors.append(f"   ... and {len(missing_keys) - 5} more")
    
    if extra_keys:
        errors.append(f"Found {len(extra_keys)} extra (country, brand, month) combinations not in template")
        for i, key in enumerate(list(extra_keys)[:5]):
            errors.append(f"   Extra: {key}")
        if len(extra_keys) > 5:
            errors.append(f"   ... and {len(extra_keys) - 5} more")
    
    # Check brand-level month ranges
    template_brand_months = template.groupby(['country', 'brand_name'])['months_postgx'].agg(['min', 'max'])
    sub_brand_months = submission.groupby(['country', 'brand_name'])['months_postgx'].agg(['min', 'max'])
    
    # Check scenario distribution
    template_starts = template.groupby(['country', 'brand_name'])['months_postgx'].min()
    s1_count = (template_starts == 0).sum()
    s2_count = (template_starts == 6).sum()
    
    sub_starts = submission.groupby(['country', 'brand_name'])['months_postgx'].min()
    sub_s1 = (sub_starts == 0).sum()
    sub_s2 = (sub_starts == 6).sum()
    
    if sub_s1 != s1_count or sub_s2 != s2_count:
        errors.append(f"Scenario distribution mismatch:")
        errors.append(f"   Template: S1={s1_count}, S2={s2_count}")
        errors.append(f"   Submission: S1={sub_s1}, S2={sub_s2}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_unified_submission(filepath: Path) -> bool:
    """Validate a unified submission file against the template."""
    
    print(f"\n{'=' * 60}")
    print(f"📄 Validating: {filepath.name}")
    print("=" * 60)
    
    # Load files
    try:
        submission = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    try:
        template = load_template()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    # Basic stats
    n_brands = submission.groupby(['country', 'brand_name']).ngroups
    print(f"\n📊 Submission Stats:")
    print(f"   Rows: {len(submission)} (expected: {len(template)})")
    print(f"   Brands: {n_brands} (expected: 340)")
    print(f"   Columns: {list(submission.columns)}")
    
    if 'volume' in submission.columns and not submission['volume'].isna().all():
        print(f"   Volume range: [{submission['volume'].min():.2f}, {submission['volume'].max():.2f}]")
        print(f"   Volume mean: {submission['volume'].mean():.2f}")
    
    # Check scenario distribution
    brand_starts = submission.groupby(['country', 'brand_name'])['months_postgx'].min()
    s1_brands = (brand_starts == 0).sum()
    s2_brands = (brand_starts == 6).sum()
    other_brands = len(brand_starts) - s1_brands - s2_brands
    
    print(f"\n📊 Scenario Distribution:")
    print(f"   Scenario 1 (months 0-23): {s1_brands} brands")
    print(f"   Scenario 2 (months 6-23): {s2_brands} brands")
    if other_brands > 0:
        print(f"   ⚠️ Other start months: {other_brands} brands")
    
    # Validate against template
    is_valid, errors = validate_against_template(submission, template)
    
    if is_valid:
        print(f"\n✅ VALIDATION PASSED!")
        print(f"   Ready for upload to competition platform.")
        return True
    else:
        print(f"\n❌ VALIDATION FAILED!")
        for error in errors:
            print(f"   {error}")
        return False


def validate_all_submissions():
    """Validate all submission files in submissions directory."""
    
    print("=" * 70)
    print("🔍 VALIDATING SUBMISSIONS AGAINST TEMPLATE")
    print("=" * 70)
    
    # Find submission files (exclude template and example)
    all_files = list(SUBMISSIONS_DIR.glob("*.csv"))
    submission_files = [f for f in all_files 
                       if 'template' not in f.name.lower() 
                       and 'example' not in f.name.lower()]
    
    if not submission_files:
        print("\n⚠️ No submission files found in submissions/")
        print("   (excluding template and example files)")
        return
    
    print(f"\nFound {len(submission_files)} submission files to validate:\n")
    
    results = []
    
    for filepath in sorted(submission_files):
        is_valid = validate_unified_submission(filepath)
        results.append({
            'file': filepath.name,
            'status': '✅ VALID' if is_valid else '❌ INVALID'
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    for r in results:
        print(f"   {r['status']} {r['file']}")
    
    n_valid = sum(1 for r in results if '✅' in r['status'])
    n_total = len(results)
    
    print(f"\n✅ Valid: {n_valid}/{n_total}")
    
    if n_valid == n_total:
        print("\n🎉 All submissions are valid and ready for upload!")
    else:
        print("\n⚠️ Some submissions have issues. Please fix before uploading.")
    
    # Recommend best file
    final_files = [f for f in submission_files if 'final' in f.name.lower()]
    if final_files:
        print(f"\n📤 Recommended file for upload: {final_files[0].name}")


def main():
    parser = argparse.ArgumentParser(description='Validate submissions against template')
    parser.add_argument('--file', type=str, help='Single file to validate')
    
    args = parser.parse_args()
    
    if args.file:
        validate_unified_submission(Path(args.file))
    else:
        validate_all_submissions()


if __name__ == "__main__":
    main()
