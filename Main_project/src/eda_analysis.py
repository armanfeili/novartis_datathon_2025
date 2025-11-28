# =============================================================================
# File: src/eda_analysis.py
# Description: EDA analysis functions (computation only, no plotting)
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def analyze_data_quality(df: pd.DataFrame, name: str = "dataset") -> dict:
    """
    Comprehensive data quality analysis.
    
    Args:
        df: DataFrame to analyze
        name: Name for reporting
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'name': name,
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_stats'] = {}
    for col in numeric_cols:
        report['numeric_stats'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'zeros': (df[col] == 0).sum(),
            'negatives': (df[col] < 0).sum()
        }
    
    return report


def analyze_brand_coverage(df: pd.DataFrame) -> dict:
    """
    Analyze coverage per brand.
    
    Args:
        df: DataFrame with country, brand_name, months_postgx
        
    Returns:
        Dictionary with coverage analysis
    """
    # Unique brands
    brands = df[['country', 'brand_name']].drop_duplicates()
    
    # Months per brand
    months_per_brand = df.groupby(['country', 'brand_name'])['months_postgx'].agg(['min', 'max', 'count'])
    
    # Pre-entry vs post-entry months
    pre_entry = df[df['months_postgx'] < 0].groupby(['country', 'brand_name']).size()
    post_entry = df[df['months_postgx'] >= 0].groupby(['country', 'brand_name']).size()
    
    report = {
        'n_brands': len(brands),
        'n_countries': brands['country'].nunique(),
        'brands_per_country': brands.groupby('country').size().to_dict(),
        'months_per_brand_stats': {
            'mean': months_per_brand['count'].mean(),
            'min': months_per_brand['count'].min(),
            'max': months_per_brand['count'].max()
        },
        'pre_entry_months_stats': {
            'mean': pre_entry.mean() if len(pre_entry) > 0 else 0,
            'min': pre_entry.min() if len(pre_entry) > 0 else 0,
            'max': pre_entry.max() if len(pre_entry) > 0 else 0
        },
        'post_entry_months_stats': {
            'mean': post_entry.mean() if len(post_entry) > 0 else 0,
            'min': post_entry.min() if len(post_entry) > 0 else 0,
            'max': post_entry.max() if len(post_entry) > 0 else 0
        }
    }
    
    return report


def analyze_bucket_distribution(aux_df: pd.DataFrame) -> dict:
    """
    Analyze bucket distribution and characteristics.
    
    Args:
        aux_df: Auxiliary file with bucket assignments
        
    Returns:
        Dictionary with bucket analysis
    """
    bucket_counts = aux_df['bucket'].value_counts().sort_index()
    
    report = {
        'bucket_counts': bucket_counts.to_dict(),
        'bucket_pct': (bucket_counts / len(aux_df) * 100).to_dict(),
        'bucket1_stats': {},
        'bucket2_stats': {}
    }
    
    for bucket_id, bucket_name in [(1, 'bucket1_stats'), (2, 'bucket2_stats')]:
        bucket_data = aux_df[aux_df['bucket'] == bucket_id]
        if len(bucket_data) > 0:
            report[bucket_name] = {
                'count': len(bucket_data),
                'mean_erosion_avg': bucket_data['mean_erosion'].mean(),
                'mean_erosion_min': bucket_data['mean_erosion'].min(),
                'mean_erosion_max': bucket_data['mean_erosion'].max(),
                'avg_vol_mean': bucket_data['avg_vol'].mean(),
                'avg_vol_median': bucket_data['avg_vol'].median()
            }
    
    return report


def analyze_erosion_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average erosion curves by months_postgx.
    
    Args:
        df: DataFrame with vol_norm column
        
    Returns:
        DataFrame with erosion curves
    """
    # Filter to post-entry months
    post_entry = df[(df['months_postgx'] >= 0) & (df['months_postgx'] <= 23)].copy()
    
    # Average erosion by month
    erosion_by_month = post_entry.groupby('months_postgx')['vol_norm'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Percentiles
    for pct in [25, 75]:
        erosion_by_month[f'p{pct}'] = post_entry.groupby('months_postgx')['vol_norm'].quantile(pct/100).values
    
    return erosion_by_month


def analyze_erosion_by_bucket(df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute erosion curves separately by bucket.
    
    Args:
        df: DataFrame with volume data
        aux_df: Auxiliary file with bucket assignments
        
    Returns:
        DataFrame with erosion by bucket and month
    """
    # Merge bucket info
    merged = df.merge(aux_df[['country', 'brand_name', 'bucket', 'avg_vol']], 
                      on=['country', 'brand_name'])
    
    # Compute normalized volume
    merged['vol_norm'] = merged['volume'] / merged['avg_vol']
    
    # Filter to post-entry
    post_entry = merged[(merged['months_postgx'] >= 0) & (merged['months_postgx'] <= 23)]
    
    # Group by bucket and month
    erosion_by_bucket = post_entry.groupby(['bucket', 'months_postgx'])['vol_norm'].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    return erosion_by_bucket


def analyze_competition_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze impact of generic competition on volume.
    
    Args:
        df: DataFrame with n_gxs and volume columns
        
    Returns:
        DataFrame with competition analysis
    """
    # Filter to post-entry
    post_entry = df[df['months_postgx'] >= 0].copy()
    
    # Bin n_gxs
    post_entry['n_gxs_bin'] = pd.cut(post_entry['n_gxs'], 
                                      bins=[-1, 0, 1, 3, 5, 10, 100],
                                      labels=['0', '1', '2-3', '4-5', '6-10', '10+'])
    
    # Group by competition level
    competition_impact = post_entry.groupby('n_gxs_bin').agg({
        'volume': ['mean', 'median', 'count'],
        'vol_norm': ['mean', 'median'] if 'vol_norm' in post_entry.columns else {}
    }).reset_index()
    
    return competition_impact


def analyze_medicine_characteristics(df: pd.DataFrame) -> dict:
    """
    Analyze medicine characteristics distribution.
    
    Args:
        df: DataFrame with medicine info
        
    Returns:
        Dictionary with characteristic analysis
    """
    report = {}
    
    # Therapeutic area distribution
    if 'ther_area' in df.columns:
        report['ther_area'] = df['ther_area'].value_counts().to_dict()
    
    # Biological vs small molecule
    if 'biological' in df.columns:
        report['biological'] = df['biological'].value_counts().to_dict()
    
    if 'small_molecule' in df.columns:
        report['small_molecule'] = df['small_molecule'].value_counts().to_dict()
    
    # Hospital rate distribution
    if 'hospital_rate' in df.columns:
        report['hospital_rate'] = {
            'mean': df['hospital_rate'].mean(),
            'median': df['hospital_rate'].median(),
            'std': df['hospital_rate'].std(),
            'min': df['hospital_rate'].min(),
            'max': df['hospital_rate'].max()
        }
    
    # Main package distribution
    if 'main_package' in df.columns:
        report['main_package'] = df['main_package'].value_counts().head(10).to_dict()
    
    return report


def run_full_eda(merged: pd.DataFrame = None, aux_df: pd.DataFrame = None, 
                 save_report: bool = True) -> dict:
    """
    Run complete EDA and return all results in visualization-friendly format.
    
    This function provides comprehensive EDA results that can be used for 
    visualization in notebooks or scripts.
    
    Args:
        merged: Pre-merged DataFrame (if None, will load data)
        aux_df: Auxiliary file DataFrame (if None, will create)
        save_report: If True, save reports to files
        
    Returns:
        Dictionary with all EDA results including:
        - bucket_distribution: DataFrame with bucket, count, percentage
        - erosion_curves: DataFrame with erosion by bucket and month
        - n_gxs_impact: DataFrame with erosion vs competition
        - competition_trajectory: DataFrame with n_gxs over time
        - ther_area_analysis: DataFrame with erosion by therapeutic area
        - bio_vs_small: DataFrame with biological vs small molecule
        - hospital_rate_analysis: DataFrame with erosion by hospital rate
        - erosion_speed: DataFrame with erosion speed metrics
        - data_summary: dict with basic dataset stats
    """
    print("=" * 70)
    print("🔍 RUNNING FULL EDA ANALYSIS")
    print("=" * 70)
    
    from data_loader import load_all_data, merge_datasets
    from bucket_calculator import compute_avg_j, create_auxiliary_file, compute_normalized_volume
    
    results = {}
    
    # Load data if not provided
    if merged is None:
        print("\n📂 Loading data...")
        volume, generics, medicine = load_all_data(train=True)
        merged = merge_datasets(volume, generics, medicine)
    
    # Create auxiliary file if not provided
    if aux_df is None:
        print("\n🪣 Creating auxiliary file...")
        aux_df = create_auxiliary_file(merged, save=save_report)
    
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # ====================
    # Data Summary
    # ====================
    print("\n📊 Computing data summary...")
    results['data_summary'] = {
        'shape': merged.shape,
        'n_brands': merged[['country', 'brand_name']].drop_duplicates().shape[0],
        'n_countries': merged['country'].nunique(),
        'months_postgx_range': (merged['months_postgx'].min(), merged['months_postgx'].max())
    }
    
    # ====================
    # Bucket Distribution (DataFrame format for plotting)
    # ====================
    print("\n📊 Analyzing bucket distribution...")
    bucket_counts = aux_df['bucket'].value_counts().sort_index()
    results['bucket_distribution'] = pd.DataFrame({
        'bucket': bucket_counts.index,
        'count': bucket_counts.values,
        'percentage': (bucket_counts.values / len(aux_df) * 100).round(1)
    })
    
    # ====================
    # Add normalized volume to merged
    # ====================
    merged_with_norm = compute_normalized_volume(merged, avg_j)
    merged_with_bucket = merged_with_norm.merge(
        aux_df[['country', 'brand_name', 'bucket']], 
        on=['country', 'brand_name']
    )
    
    # ====================
    # Erosion Curves by Bucket
    # ====================
    print("\n📉 Computing erosion curves by bucket...")
    post_entry = merged_with_bucket[
        (merged_with_bucket['months_postgx'] >= 0) & 
        (merged_with_bucket['months_postgx'] <= 23)
    ]
    
    erosion_curves = post_entry.groupby(['bucket', 'months_postgx']).agg({
        'vol_norm': ['mean', 'std']
    }).reset_index()
    erosion_curves.columns = ['bucket', 'months_postgx', 'mean_vol_norm', 'std_vol_norm']
    results['erosion_curves'] = erosion_curves
    
    if save_report:
        erosion_curves.to_csv(DATA_PROCESSED / "eda_erosion_curves.csv", index=False)
    
    # ====================
    # n_gxs Impact
    # ====================
    print("\n🏁 Analyzing n_gxs impact...")
    n_gxs_impact = post_entry.groupby('n_gxs').agg({
        'vol_norm': 'mean'
    }).reset_index()
    n_gxs_impact.columns = ['n_gxs', 'mean_vol_norm']
    results['n_gxs_impact'] = n_gxs_impact
    
    # ====================
    # Competition Trajectory Over Time
    # ====================
    print("\n📈 Computing competition trajectory...")
    competition_trajectory = post_entry.groupby('months_postgx').agg({
        'n_gxs': ['mean', 'std']
    }).reset_index()
    competition_trajectory.columns = ['months_postgx', 'mean_n_gxs', 'std_n_gxs']
    results['competition_trajectory'] = competition_trajectory
    
    # ====================
    # Therapeutic Area Analysis
    # ====================
    print("\n💊 Analyzing by therapeutic area...")
    ther_analysis = pd.DataFrame()
    if 'ther_area' in post_entry.columns:
        ther_analysis = post_entry.groupby('ther_area').agg({
            'vol_norm': 'mean'
        }).reset_index()
        ther_analysis.columns = ['ther_area', 'mean_erosion']
    results['ther_area_analysis'] = ther_analysis
    
    # ====================
    # Biological vs Small Molecule
    # ====================
    print("\n🧬 Analyzing biological vs small molecule...")
    bio_vs_small = pd.DataFrame()
    if 'biological' in post_entry.columns:
        bio_vs_small = post_entry.pivot_table(
            index='months_postgx',
            columns='biological',
            values='vol_norm',
            aggfunc='mean'
        )
        bio_vs_small.columns = [f"Biological={c}" for c in bio_vs_small.columns]
    results['bio_vs_small'] = bio_vs_small
    
    # ====================
    # Hospital Rate Analysis
    # ====================
    print("\n🏥 Analyzing by hospital rate...")
    hospital_analysis = pd.DataFrame()
    if 'hospital_rate' in post_entry.columns:
        # Bin hospital rate
        post_entry_hr = post_entry.copy()
        post_entry_hr['hospital_bin'] = pd.cut(
            post_entry_hr['hospital_rate'], 
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['0-25%', '25-50%', '50-75%', '75-100%']
        )
        hospital_analysis = post_entry_hr.pivot_table(
            index='months_postgx',
            columns='hospital_bin',
            values='vol_norm',
            aggfunc='mean',
            observed=False
        )
    results['hospital_rate_analysis'] = hospital_analysis
    
    # ====================
    # Erosion Speed Analysis
    # ====================
    print("\n⚡ Computing erosion speed metrics...")
    erosion_speed_data = []
    
    for (country, brand), group in merged_with_bucket.groupby(['country', 'brand_name']):
        bucket = group['bucket'].iloc[0]
        post = group[group['months_postgx'] >= 0].sort_values('months_postgx')
        
        if len(post) == 0:
            continue
        
        # Time to 50% erosion
        time_to_50pct = None
        for _, row in post.iterrows():
            if row['vol_norm'] <= 0.5:
                time_to_50pct = row['months_postgx']
                break
        
        # Erosion in first 6 months
        first_6m = post[post['months_postgx'] <= 6]['vol_norm'].mean()
        erosion_first_6m = 1 - first_6m if not pd.isna(first_6m) else None
        
        # Final equilibrium (months 18-23)
        final_months = post[post['months_postgx'] >= 18]['vol_norm'].mean()
        
        erosion_speed_data.append({
            'country': country,
            'brand_name': brand,
            'bucket': bucket,
            'time_to_50pct': time_to_50pct,
            'erosion_first_6m': erosion_first_6m,
            'final_equilibrium': final_months
        })
    
    results['erosion_speed'] = pd.DataFrame(erosion_speed_data)
    
    # ====================
    # Save report
    # ====================
    if save_report:
        import json
        report_data = {
            'data_summary': {
                'shape': list(results['data_summary']['shape']),
                'n_brands': int(results['data_summary']['n_brands']),
                'n_countries': int(results['data_summary']['n_countries']),
                'months_postgx_range': [int(x) for x in results['data_summary']['months_postgx_range']]
            },
            'bucket_distribution': [
                {'bucket': int(r['bucket']), 'count': int(r['count']), 'percentage': float(r['percentage'])}
                for r in results['bucket_distribution'].to_dict('records')
            ]
        }
        with open(REPORTS_DIR / 'eda_summary.json', 'w') as f:
            json.dump(report_data, f, indent=2)
    
    # ====================
    # Summary Print
    # ====================
    print("\n" + "=" * 70)
    print("📋 EDA SUMMARY")
    print("=" * 70)
    print(f"   Dataset Shape: {results['data_summary']['shape']}")
    print(f"   Total brands: {results['data_summary']['n_brands']}")
    print(f"   Countries: {results['data_summary']['n_countries']}")
    bucket_df = results['bucket_distribution']
    for _, row in bucket_df.iterrows():
        print(f"   Bucket {int(row['bucket'])}: {int(row['count'])} brands ({row['percentage']}%)")
    
    return results


def run_full_eda_legacy(save_results: bool = True) -> dict:
    """
    Legacy version of run_full_eda for backward compatibility.
    
    Args:
        save_results: If True, save intermediate results
        
    Returns:
        Dictionary with all EDA results (legacy format)
    """
    print("=" * 70)
    print("🔍 RUNNING FULL EDA ANALYSIS (Legacy)")
    print("=" * 70)
    
    from data_loader import load_all_data, merge_datasets
    from bucket_calculator import compute_avg_j, create_auxiliary_file, compute_normalized_volume
    
    results = {}
    
    # Load data
    print("\n📂 Loading data...")
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    # Data quality
    print("\n📊 Analyzing data quality...")
    results['volume_quality'] = analyze_data_quality(volume, "volume_train")
    results['generics_quality'] = analyze_data_quality(generics, "generics_train")
    results['medicine_quality'] = analyze_data_quality(medicine, "medicine_info_train")
    results['merged_quality'] = analyze_data_quality(merged, "merged_train")
    
    # Brand coverage
    print("\n📈 Analyzing brand coverage...")
    results['brand_coverage'] = analyze_brand_coverage(merged)
    
    # Create auxiliary file
    print("\n🪣 Creating auxiliary file...")
    aux_df = create_auxiliary_file(merged, save=save_results)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # Bucket distribution
    print("\n📊 Analyzing bucket distribution...")
    results['bucket_distribution'] = analyze_bucket_distribution(aux_df)
    
    # Add normalized volume
    merged_with_norm = compute_normalized_volume(merged, avg_j)
    
    # Erosion curves
    print("\n📉 Computing erosion curves...")
    erosion_curves = analyze_erosion_curves(merged_with_norm)
    results['erosion_curves'] = erosion_curves
    
    if save_results:
        erosion_curves.to_csv(DATA_PROCESSED / "eda_erosion_curves.csv", index=False)
    
    # Erosion by bucket
    print("\n📊 Computing erosion by bucket...")
    erosion_by_bucket = analyze_erosion_by_bucket(merged, aux_df)
    results['erosion_by_bucket'] = erosion_by_bucket
    
    if save_results:
        erosion_by_bucket.to_csv(DATA_PROCESSED / "eda_erosion_by_bucket.csv", index=False)
    
    # Competition impact
    print("\n🏁 Analyzing competition impact...")
    competition = analyze_competition_impact(merged_with_norm)
    results['competition_impact'] = competition
    
    # Medicine characteristics
    print("\n💊 Analyzing medicine characteristics...")
    results['medicine_chars'] = analyze_medicine_characteristics(merged)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 EDA SUMMARY")
    print("=" * 70)
    print(f"   Total brands: {results['brand_coverage']['n_brands']}")
    print(f"   Countries: {results['brand_coverage']['n_countries']}")
    print(f"   Bucket 1 (high erosion): {results['bucket_distribution']['bucket_counts'].get(1, 0)}")
    print(f"   Bucket 2 (lower erosion): {results['bucket_distribution']['bucket_counts'].get(2, 0)}")
    
    return results


if __name__ == "__main__":
    results = run_full_eda(save_report=True)
    print("\n✅ EDA analysis complete!")
