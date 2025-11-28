# =============================================================================
# File: src/evaluation.py
# Description: Functions to evaluate models using official PE metric
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def compute_pe_scenario1(actual_df: pd.DataFrame,
                         pred_df: pd.DataFrame,
                         aux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Prediction Error for Scenario 1 (months 0-23).
    
    PE = 0.2 * (Σ|actual-pred| / 24*avg_vol) 
       + 0.5 * (|Σ(0-5) actual - Σ(0-5) pred| / 6*avg_vol)
       + 0.2 * (|Σ(6-11) actual - Σ(6-11) pred| / 6*avg_vol)
       + 0.1 * (|Σ(12-23) actual - Σ(12-23) pred| / 12*avg_vol)
    
    Args:
        actual_df: DataFrame with actual volumes
        pred_df: DataFrame with predicted volumes
        aux_df: Auxiliary file with avg_vol and bucket
        
    Returns:
        DataFrame with PE per brand
    """
    # Merge actual, predictions, and auxiliary data
    merged = actual_df.merge(
        pred_df, on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_actual', '_pred')
    ).merge(aux_df[['country', 'brand_name', 'avg_vol', 'bucket']], 
            on=['country', 'brand_name'])
    
    results = []
    for (country, brand), group in merged.groupby(['country', 'brand_name']):
        avg_vol = group['avg_vol'].iloc[0]
        bucket = group['bucket'].iloc[0]
        
        if avg_vol == 0 or np.isnan(avg_vol):
            continue
        
        # Term 1: Monthly absolute error (0-23)
        monthly_err = group['volume_actual'].sub(group['volume_pred']).abs().sum()
        term1 = S1_MONTHLY_WEIGHT * monthly_err / (24 * avg_vol)
        
        # Term 2: Accumulated error months 0-5
        m0_5 = group[group['months_postgx'].between(0, 5)]
        sum_err_0_5 = abs(m0_5['volume_actual'].sum() - m0_5['volume_pred'].sum())
        term2 = S1_SUM_0_5_WEIGHT * sum_err_0_5 / (6 * avg_vol)
        
        # Term 3: Accumulated error months 6-11
        m6_11 = group[group['months_postgx'].between(6, 11)]
        sum_err_6_11 = abs(m6_11['volume_actual'].sum() - m6_11['volume_pred'].sum())
        term3 = S1_SUM_6_11_WEIGHT * sum_err_6_11 / (6 * avg_vol)
        
        # Term 4: Accumulated error months 12-23
        m12_23 = group[group['months_postgx'].between(12, 23)]
        sum_err_12_23 = abs(m12_23['volume_actual'].sum() - m12_23['volume_pred'].sum())
        term4 = S1_SUM_12_23_WEIGHT * sum_err_12_23 / (12 * avg_vol)
        
        pe = term1 + term2 + term3 + term4
        
        results.append({
            'country': country,
            'brand_name': brand,
            'bucket': bucket,
            'avg_vol': avg_vol,
            'PE': pe,
            'term1_monthly': term1,
            'term2_sum_0_5': term2,
            'term3_sum_6_11': term3,
            'term4_sum_12_23': term4
        })
    
    return pd.DataFrame(results)


def compute_pe_scenario2(actual_df: pd.DataFrame,
                         pred_df: pd.DataFrame,
                         aux_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Prediction Error for Scenario 2 (months 6-23).
    
    PE = 0.2 * (Σ|actual-pred| / 18*avg_vol)
       + 0.5 * (|Σ(6-11) actual - Σ(6-11) pred| / 6*avg_vol)
       + 0.3 * (|Σ(12-23) actual - Σ(12-23) pred| / 12*avg_vol)
    
    Args:
        actual_df: DataFrame with actual volumes (months 6-23)
        pred_df: DataFrame with predicted volumes (months 6-23)
        aux_df: Auxiliary file with avg_vol and bucket
        
    Returns:
        DataFrame with PE per brand
    """
    # Merge actual, predictions, and auxiliary data
    merged = actual_df.merge(
        pred_df, on=['country', 'brand_name', 'months_postgx'],
        suffixes=('_actual', '_pred')
    ).merge(aux_df[['country', 'brand_name', 'avg_vol', 'bucket']], 
            on=['country', 'brand_name'])
    
    results = []
    for (country, brand), group in merged.groupby(['country', 'brand_name']):
        avg_vol = group['avg_vol'].iloc[0]
        bucket = group['bucket'].iloc[0]
        
        if avg_vol == 0 or np.isnan(avg_vol):
            continue
        
        # Filter to months 6-23 only
        group = group[group['months_postgx'] >= 6]
        
        # Term 1: Monthly absolute error (6-23)
        monthly_err = group['volume_actual'].sub(group['volume_pred']).abs().sum()
        term1 = S2_MONTHLY_WEIGHT * monthly_err / (18 * avg_vol)
        
        # Term 2: Accumulated error months 6-11
        m6_11 = group[group['months_postgx'].between(6, 11)]
        sum_err_6_11 = abs(m6_11['volume_actual'].sum() - m6_11['volume_pred'].sum())
        term2 = S2_SUM_6_11_WEIGHT * sum_err_6_11 / (6 * avg_vol)
        
        # Term 3: Accumulated error months 12-23
        m12_23 = group[group['months_postgx'].between(12, 23)]
        sum_err_12_23 = abs(m12_23['volume_actual'].sum() - m12_23['volume_pred'].sum())
        term3 = S2_SUM_12_23_WEIGHT * sum_err_12_23 / (12 * avg_vol)
        
        pe = term1 + term2 + term3
        
        results.append({
            'country': country,
            'brand_name': brand,
            'bucket': bucket,
            'avg_vol': avg_vol,
            'PE': pe,
            'term1_monthly': term1,
            'term2_sum_6_11': term2,
            'term3_sum_12_23': term3
        })
    
    return pd.DataFrame(results)


def compute_final_metric(pe_df: pd.DataFrame) -> float:
    """
    Compute final weighted metric.
    
    PE_final = (2/n_B1) * Σ(PE_B1) + (1/n_B2) * Σ(PE_B2)
    
    Bucket 1 (high erosion) is weighted 2x!
    
    Args:
        pe_df: DataFrame with PE per brand and bucket
        
    Returns:
        Final weighted PE score
    """
    bucket1 = pe_df[pe_df['bucket'] == 1]
    bucket2 = pe_df[pe_df['bucket'] == 2]
    
    n_b1 = len(bucket1)
    n_b2 = len(bucket2)
    
    if n_b1 == 0 and n_b2 == 0:
        return np.nan
    
    score = 0
    if n_b1 > 0:
        score += (BUCKET_1_WEIGHT / n_b1) * bucket1['PE'].sum()
    if n_b2 > 0:
        score += (BUCKET_2_WEIGHT / n_b2) * bucket2['PE'].sum()
    
    return score


def evaluate_model(actual_df: pd.DataFrame,
                   pred_df: pd.DataFrame,
                   aux_df: pd.DataFrame,
                   scenario: int = 1) -> dict:
    """
    Full evaluation pipeline.
    
    Args:
        actual_df: DataFrame with actual volumes
        pred_df: DataFrame with predicted volumes
        aux_df: Auxiliary file with avg_vol and bucket
        scenario: 1 or 2
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATING SCENARIO {scenario}")
    print("=" * 60)
    
    # Compute PE per brand
    if scenario == 1:
        pe_df = compute_pe_scenario1(actual_df, pred_df, aux_df)
    else:
        pe_df = compute_pe_scenario2(actual_df, pred_df, aux_df)
    
    # Compute final metric
    final_score = compute_final_metric(pe_df)
    
    # Bucket breakdown
    bucket1 = pe_df[pe_df['bucket'] == 1]
    bucket2 = pe_df[pe_df['bucket'] == 2]
    
    bucket1_avg_pe = bucket1['PE'].mean() if len(bucket1) > 0 else np.nan
    bucket2_avg_pe = bucket2['PE'].mean() if len(bucket2) > 0 else np.nan
    
    results = {
        'scenario': scenario,
        'final_score': final_score,
        'bucket1_avg_pe': bucket1_avg_pe,
        'bucket2_avg_pe': bucket2_avg_pe,
        'n_bucket1': len(bucket1),
        'n_bucket2': len(bucket2),
        'n_total_brands': len(pe_df),
        'pe_details': pe_df
    }
    
    # Print summary
    print(f"\n📊 EVALUATION RESULTS - Scenario {scenario}")
    print(f"   Final Score: {final_score:.4f}")
    print(f"   Bucket 1 Avg PE: {bucket1_avg_pe:.4f} (n={len(bucket1)}, weight=2x)")
    print(f"   Bucket 2 Avg PE: {bucket2_avg_pe:.4f} (n={len(bucket2)}, weight=1x)")
    
    return results


def analyze_worst_predictions(pe_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Identify worst-predicted brands for error analysis.
    
    Args:
        pe_df: DataFrame with PE per brand
        n: Number of worst brands to return
        
    Returns:
        DataFrame with worst predictions
    """
    worst = pe_df.nlargest(n, 'PE')
    print(f"\n⚠️ Top {n} Worst Predictions:")
    print(worst[['country', 'brand_name', 'bucket', 'PE']].to_string(index=False))
    return worst


def compare_models(results_list: list, model_names: list) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results_list: List of evaluation result dicts
        model_names: List of model names
        
    Returns:
        Comparison DataFrame
    """
    comparison = []
    for name, results in zip(model_names, results_list):
        comparison.append({
            'model': name,
            'final_score': results['final_score'],
            'bucket1_pe': results['bucket1_avg_pe'],
            'bucket2_pe': results['bucket2_avg_pe']
        })
    
    df = pd.DataFrame(comparison).sort_values('final_score')
    print("\n📊 Model Comparison:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    # Demo: Evaluate a baseline model
    print("=" * 60)
    print("EVALUATION DEMO")
    print("=" * 60)
    
    from data_loader import load_all_data, merge_datasets, split_train_validation
    from bucket_calculator import compute_avg_j, create_auxiliary_file
    from models import BaselineModels
    
    # Load data
    volume, generics, medicine = load_all_data(train=True)
    merged = merge_datasets(volume, generics, medicine)
    
    # Create auxiliary file
    aux_df = create_auxiliary_file(merged, save=True)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # Split for validation
    train_df, val_df = split_train_validation(merged)
    
    # Get actual post-entry volumes from validation set
    actual_df = val_df[val_df['months_postgx'].between(0, 23)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ].copy()
    
    # Generate baseline predictions
    val_brands = val_df[['country', 'brand_name']].drop_duplicates()
    val_avg_j = avg_j.merge(val_brands, on=['country', 'brand_name'])
    
    pred_df = BaselineModels.exponential_decay(
        val_avg_j, 
        months_to_predict=list(range(0, 24)),
        decay_rate=0.05
    )
    
    # Evaluate
    results = evaluate_model(actual_df, pred_df, aux_df, scenario=1)
    
    # Analyze worst predictions
    analyze_worst_predictions(results['pe_details'])
    
    print("\n✅ Evaluation demo complete!")
