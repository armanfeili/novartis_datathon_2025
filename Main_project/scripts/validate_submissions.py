# =============================================================================
# File: scripts/validate_submissions.py
# Description: Validate submission files before upload
# =============================================================================

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from submission import validate_submission, load_submission


def validate_all_submissions():
    """Validate all submission files in submissions directory."""
    
    print("=" * 70)
    print("🔍 VALIDATING ALL SUBMISSIONS")
    print("=" * 70)
    
    submission_files = list(SUBMISSIONS_DIR.glob("*.csv"))
    
    if not submission_files:
        print("\n⚠️ No submission files found in submissions/")
        return
    
    print(f"\nFound {len(submission_files)} submission files:\n")
    
    results = []
    
    for filepath in sorted(submission_files):
        print(f"\n{'=' * 50}")
        print(f"📄 {filepath.name}")
        print("=" * 50)
        
        # Determine scenario from filename
        if 'scenario1' in filepath.name.lower():
            scenario = 1
        elif 'scenario2' in filepath.name.lower():
            scenario = 2
        else:
            # Try to infer from content
            df = load_submission(filepath)
            min_month = df['months_postgx'].min()
            scenario = 1 if min_month == 0 else 2
            print(f"   Inferred scenario: {scenario} (min month = {min_month})")
        
        try:
            df = load_submission(filepath)
            
            # Basic stats
            print(f"\n   📊 Basic Stats:")
            print(f"      Rows: {len(df)}")
            print(f"      Brands: {df[['country', 'brand_name']].drop_duplicates().shape[0]}")
            print(f"      Months range: [{df['months_postgx'].min()}, {df['months_postgx'].max()}]")
            print(f"      Volume range: [{df['volume'].min():.2f}, {df['volume'].max():.2f}]")
            
            # Validate
            valid = validate_submission(df, scenario)
            
            results.append({
                'file': filepath.name,
                'scenario': scenario,
                'rows': len(df),
                'brands': df[['country', 'brand_name']].drop_duplicates().shape[0],
                'status': '✅ VALID' if valid else '❌ INVALID'
            })
            
        except Exception as e:
            print(f"\n   ❌ Error: {str(e)}")
            results.append({
                'file': filepath.name,
                'scenario': scenario,
                'rows': None,
                'brands': None,
                'status': f'❌ ERROR: {str(e)[:50]}'
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    n_valid = sum(1 for r in results if '✅' in r['status'])
    n_total = len(results)
    
    print(f"\n✅ Valid: {n_valid}/{n_total}")
    
    if n_valid == n_total:
        print("\n🎉 All submissions are valid and ready for upload!")
    else:
        print("\n⚠️ Some submissions have issues. Please fix before uploading.")


def validate_single_file(filepath: str, scenario: int):
    """Validate a single submission file."""
    
    print("=" * 70)
    print(f"🔍 VALIDATING: {filepath}")
    print("=" * 70)
    
    df = load_submission(Path(filepath))
    
    # Basic stats
    print(f"\n📊 Basic Stats:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Brands: {df[['country', 'brand_name']].drop_duplicates().shape[0]}")
    print(f"   Countries: {df['country'].nunique()}")
    print(f"   Months range: [{df['months_postgx'].min()}, {df['months_postgx'].max()}]")
    print(f"   Volume range: [{df['volume'].min():.4f}, {df['volume'].max():.4f}]")
    print(f"   Volume mean: {df['volume'].mean():.4f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Validate
    try:
        valid = validate_submission(df, scenario)
        print("\n✅ Submission is VALID!")
    except AssertionError as e:
        print(f"\n❌ Validation FAILED: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Validate submissions')
    parser.add_argument('--file', type=str, help='Single file to validate')
    parser.add_argument('--scenario', type=int, choices=[1, 2], 
                        help='Scenario (required if --file specified)')
    
    args = parser.parse_args()
    
    if args.file:
        if not args.scenario:
            print("Error: --scenario required when validating single file")
            return
        validate_single_file(args.file, args.scenario)
    else:
        validate_all_submissions()


if __name__ == "__main__":
    main()
