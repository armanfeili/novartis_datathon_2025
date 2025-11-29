"""
📊 Model Results Analysis Script

This script saves detailed JSON data and CSV files for each figure.
All data files are saved to reports/03_model_data/ for later interpretation.

Usage:
    python notebooks/03_model_results.py
"""

# Setup - Add src to path
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports - handle running from different directories
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # notebooks/ -> Main_project/
sys.path.insert(0, str(project_root / 'src'))

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("✅ Setup complete!")
print(f"Project root: {project_root}")

# Import from src modules
from config import *

print("✅ Config imported!")
print(f"Models directory: {MODELS_DIR}")
print(f"Reports directory: {REPORTS_DIR}")

# Create output directory for model results data
MODEL_DATA_DIR = REPORTS_DIR / '03_model_data'
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FIGURES_DIR = MODEL_DATA_DIR / 'figures'
MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Model data will be saved to: {MODEL_DATA_DIR}")

# =============================================================================
# Helper Functions
# =============================================================================

def save_model_json(data: dict, filename: str, description: str = ""):
    """Save model results data to JSON with proper type conversion."""
    
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to Python native types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_serializable(x) for x in obj.tolist()]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    output = {
        'metadata': {
            'filename': filename,
            'description': description,
            'generated_at': datetime.now().isoformat(),
            'figure_file': f"figures/{filename.replace('.json', '.png')}"
        },
        'data': convert_to_serializable(data)
    }
    
    filepath = MODEL_DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Saved: {filepath.name}")
    return filepath


def save_model_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV."""
    filepath = MODEL_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"   💾 Saved: {filepath.name}")
    return filepath


# =============================================================================
# Load Model Comparison Results
# =============================================================================

print("\n" + "="*60)
print("📂 LOADING MODEL COMPARISON RESULTS")
print("="*60)

# Load comparison results (try latest first, then fall back to old naming)
comparison_s1 = None
comparison_s2 = None

try:
    # Try new naming convention first
    if (REPORTS_DIR / 'model_comparison_scenario1_latest.csv').exists():
        comparison_s1 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario1_latest.csv')
    else:
        comparison_s1 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario1.csv')
    print("✅ Loaded Scenario 1 results")
    print(comparison_s1.to_string())
except FileNotFoundError:
    print("⚠️ Scenario 1 results not found. Run scripts/train_models.py --scenario 1 first.")

try:
    # Try new naming convention first
    if (REPORTS_DIR / 'model_comparison_scenario2_latest.csv').exists():
        comparison_s2 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario2_latest.csv')
    else:
        comparison_s2 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario2.csv')
    print("✅ Loaded Scenario 2 results")
    print(comparison_s2.to_string())
except FileNotFoundError:
    print("⚠️ Scenario 2 results not found. Run scripts/train_models.py --scenario 2 first.")

# =============================================================================
# FIGURE 1: Model Comparison (Both Scenarios)
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 1: Model Comparison")
print("="*60)

# Prepare JSON data
model_comparison_json = {
    'summary': {
        'scenario_1_available': comparison_s1 is not None,
        'scenario_2_available': comparison_s2 is not None
    },
    'scenario_1': {},
    'scenario_2': {},
    'interpretation': {
        'metric': 'Prediction Error (PE) - lower is better',
        'bucket_weighting': 'Bucket 1 (high erosion) has 2x weight',
        'scenario_1_description': 'Predict months 0-23 (full trajectory)',
        'scenario_2_description': 'Predict months 6-23 (given months 0-5)'
    }
}

if comparison_s1 is not None:
    best_s1 = comparison_s1.loc[comparison_s1['final_score'].idxmin()]
    model_comparison_json['scenario_1'] = {
        'best_model': best_s1['model'],
        'best_score': float(best_s1['final_score']),
        'all_models': comparison_s1.to_dict('records'),
        'model_ranking': comparison_s1.sort_values('final_score')['model'].tolist()
    }
    save_model_csv(comparison_s1, 'fig01_model_comparison_s1.csv')

if comparison_s2 is not None:
    best_s2 = comparison_s2.loc[comparison_s2['final_score'].idxmin()]
    model_comparison_json['scenario_2'] = {
        'best_model': best_s2['model'],
        'best_score': float(best_s2['final_score']),
        'all_models': comparison_s2.to_dict('records'),
        'model_ranking': comparison_s2.sort_values('final_score')['model'].tolist()
    }
    save_model_csv(comparison_s2, 'fig01_model_comparison_s2.csv')

save_model_json(model_comparison_json, 'fig01_model_comparison.json',
                'Model comparison across both scenarios')

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scenario 1
if comparison_s1 is not None:
    colors = ['#2ecc71' if i == comparison_s1['final_score'].idxmin() else '#3498db' 
              for i in range(len(comparison_s1))]
    bars1 = axes[0].barh(comparison_s1['model'], comparison_s1['final_score'], 
                         color=colors, edgecolor='black')
    axes[0].set_xlabel('Final Score (PE, lower is better)', fontsize=11)
    axes[0].set_title('Scenario 1: Model Comparison\n(Predict months 0-23)', fontsize=12)
    axes[0].invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars1, comparison_s1['final_score']):
        axes[0].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                     f'{val:.4f}', va='center', fontsize=10)
    
    # Mark best model
    best_idx = comparison_s1['final_score'].idxmin()
    axes[0].annotate('🏆 BEST', xy=(comparison_s1.loc[best_idx, 'final_score'], best_idx),
                     xytext=(10, 0), textcoords='offset points', fontsize=10, color='green')
else:
    axes[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Scenario 1: Model Comparison')

# Scenario 2
if comparison_s2 is not None:
    colors = ['#2ecc71' if i == comparison_s2['final_score'].idxmin() else '#e74c3c' 
              for i in range(len(comparison_s2))]
    bars2 = axes[1].barh(comparison_s2['model'], comparison_s2['final_score'], 
                         color=colors, edgecolor='black')
    axes[1].set_xlabel('Final Score (PE, lower is better)', fontsize=11)
    axes[1].set_title('Scenario 2: Model Comparison\n(Predict months 6-23, given 0-5)', fontsize=12)
    axes[1].invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars2, comparison_s2['final_score']):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                     f'{val:.4f}', va='center', fontsize=10)
    
    # Mark best model
    best_idx = comparison_s2['final_score'].idxmin()
    axes[1].annotate('🏆 BEST', xy=(comparison_s2.loc[best_idx, 'final_score'], best_idx),
                     xytext=(10, 0), textcoords='offset points', fontsize=10, color='green')
else:
    axes[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Scenario 2: Model Comparison')

plt.tight_layout()
plt.savefig(MODEL_FIGURES_DIR / 'fig01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig01_model_comparison.png")

# =============================================================================
# FIGURE 2: Bucket-Level Performance
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 2: Bucket-Level Performance")
print("="*60)

# Prepare JSON data
bucket_performance_json = {
    'summary': {
        'bucket_1_definition': 'High erosion brands (2x weight in scoring)',
        'bucket_2_definition': 'Lower erosion brands (1x weight)'
    },
    'scenario_1_bucket_performance': [],
    'scenario_2_bucket_performance': []
}

if comparison_s1 is not None and 'bucket1_pe' in comparison_s1.columns:
    for _, row in comparison_s1.iterrows():
        bucket_performance_json['scenario_1_bucket_performance'].append({
            'model': row['model'],
            'bucket1_pe': float(row['bucket1_pe']) if pd.notna(row['bucket1_pe']) else None,
            'bucket2_pe': float(row['bucket2_pe']) if pd.notna(row['bucket2_pe']) else None,
            'final_score': float(row['final_score'])
        })

if comparison_s2 is not None and 'bucket1_pe' in comparison_s2.columns:
    for _, row in comparison_s2.iterrows():
        bucket_performance_json['scenario_2_bucket_performance'].append({
            'model': row['model'],
            'bucket1_pe': float(row['bucket1_pe']) if pd.notna(row['bucket1_pe']) else None,
            'bucket2_pe': float(row['bucket2_pe']) if pd.notna(row['bucket2_pe']) else None,
            'final_score': float(row['final_score'])
        })

save_model_json(bucket_performance_json, 'fig02_bucket_performance.json',
                'Bucket-level performance comparison')

# Plot bucket-level analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
width = 0.35

# Scenario 1 bucket breakdown
if comparison_s1 is not None and 'bucket1_pe' in comparison_s1.columns:
    x = np.arange(len(comparison_s1))
    
    bars1 = axes[0].bar(x - width/2, comparison_s1['bucket1_pe'].fillna(0), width, 
                        label='Bucket 1 (2× weight)', color='#ff6b6b', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, comparison_s1['bucket2_pe'].fillna(0), width,
                        label='Bucket 2', color='#4ecdc4', edgecolor='black')
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Prediction Error (PE)')
    axes[0].set_title('Scenario 1: Bucket-Level Performance', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison_s1['model'], rotation=45, ha='right')
    axes[0].legend()
else:
    axes[0].text(0.5, 0.5, 'No bucket data available', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Scenario 1: Bucket-Level Performance')

# Scenario 2 bucket breakdown  
if comparison_s2 is not None and 'bucket1_pe' in comparison_s2.columns:
    x = np.arange(len(comparison_s2))
    
    bars1 = axes[1].bar(x - width/2, comparison_s2['bucket1_pe'].fillna(0), width,
                        label='Bucket 1 (2× weight)', color='#ff6b6b', edgecolor='black')
    bars2 = axes[1].bar(x + width/2, comparison_s2['bucket2_pe'].fillna(0), width,
                        label='Bucket 2', color='#4ecdc4', edgecolor='black')
    
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Prediction Error (PE)')
    axes[1].set_title('Scenario 2: Bucket-Level Performance', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_s2['model'], rotation=45, ha='right')
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, 'No bucket data available', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Scenario 2: Bucket-Level Performance')

plt.tight_layout()
plt.savefig(MODEL_FIGURES_DIR / 'fig02_bucket_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig02_bucket_performance.png")

# =============================================================================
# FIGURE 3: Feature Importance (LightGBM)
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 3: Feature Importance")
print("="*60)

feature_importance_json = {
    'summary': {
        'model_type': 'LightGBM',
        'importance_metric': 'Feature importance (gain/split)'
    },
    'scenario_1_importance': [],
    'scenario_2_importance': []
}

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for idx, scenario in enumerate([1, 2]):
    model_path = MODELS_DIR / f'scenario{scenario}_lightgbm.joblib'
    
    if model_path.exists():
        data = joblib.load(model_path)
        model = data['model']
        feature_names = data['feature_names']
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Save to JSON
        top_15 = importance.tail(15)
        importance_list = [
            {'feature': row['feature'], 'importance': float(row['importance'])}
            for _, row in top_15.iterrows()
        ]
        
        if scenario == 1:
            feature_importance_json['scenario_1_importance'] = importance_list
        else:
            feature_importance_json['scenario_2_importance'] = importance_list
        
        # Save full importance to CSV
        save_model_csv(importance, f'fig03_feature_importance_s{scenario}.csv')
        
        # Plot
        axes[idx].barh(top_15['feature'], top_15['importance'], 
                       color='steelblue', edgecolor='black')
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'Scenario {scenario}: Top 15 Features (LightGBM)', fontsize=12)
    else:
        axes[idx].text(0.5, 0.5, f'Model not found\nRun train_models.py first',
                       ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'Scenario {scenario}: Feature Importance')

save_model_json(feature_importance_json, 'fig03_feature_importance.json',
                'Feature importance from LightGBM models')

plt.tight_layout()
plt.savefig(MODEL_FIGURES_DIR / 'fig03_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig03_feature_importance.png")

# =============================================================================
# FIGURE 4: Submission Analysis
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 4: Submission Analysis")
print("="*60)

# Load and analyze submissions
submission_files = list(SUBMISSIONS_DIR.glob('scenario*_*_final.csv'))

print(f"📁 Found {len(submission_files)} submission files:")
for f in submission_files:
    print(f"   - {f.name}")

submissions = {}
submission_analysis_json = {
    'summary': {
        'files_found': len(submission_files)
    },
    'scenario_1': {},
    'scenario_2': {}
}

for f in submission_files:
    df = pd.read_csv(f)
    scenario = 1 if 'scenario1' in f.name else 2
    submissions[scenario] = df
    
    # Calculate statistics
    stats = {
        'filename': f.name,
        'total_rows': len(df),
        'unique_brands': df[['country', 'brand_name']].drop_duplicates().shape[0],
        'volume_stats': {
            'min': float(df['volume'].min()),
            'max': float(df['volume'].max()),
            'mean': float(df['volume'].mean()),
            'median': float(df['volume'].median()),
            'std': float(df['volume'].std())
        },
        'months_range': [int(df['months_postgx'].min()), int(df['months_postgx'].max())]
    }
    
    if scenario == 1:
        submission_analysis_json['scenario_1'] = stats
    else:
        submission_analysis_json['scenario_2'] = stats
    
    print(f"\n📊 {f.name}:")
    print(f"   Rows: {len(df):,}")
    print(f"   Brands: {stats['unique_brands']}")
    print(f"   Volume range: [{stats['volume_stats']['min']:.2f}, {stats['volume_stats']['max']:.2f}]")

# Save submission summaries
for scenario, df in submissions.items():
    avg_by_month = df.groupby('months_postgx')['volume'].agg(['mean', 'std', 'count']).reset_index()
    save_model_csv(avg_by_month, f'fig04_submission_by_month_s{scenario}.csv')

save_model_json(submission_analysis_json, 'fig04_submission_analysis.json',
                'Submission files analysis')

# Visualize submission predictions
if len(submissions) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (scenario, df) in enumerate(submissions.items()):
        # Average predicted volume by month
        avg_by_month = df.groupby('months_postgx')['volume'].mean()
        
        axes[idx].plot(avg_by_month.index, avg_by_month.values, marker='o', 
                       linewidth=2, color='steelblue')
        axes[idx].fill_between(avg_by_month.index, 0, avg_by_month.values, alpha=0.3)
        axes[idx].set_xlabel('Months Post Generic Entry')
        axes[idx].set_ylabel('Average Predicted Volume')
        axes[idx].set_title(f'Scenario {scenario}: Predicted Erosion Curve', fontsize=12)
        
        # Add annotations
        if scenario == 1:
            axes[idx].axvspan(0, 5, alpha=0.2, color='red', label='High weight (50%)')
            axes[idx].axvspan(6, 11, alpha=0.1, color='orange', label='Medium weight (20%)')
        else:
            axes[idx].axvspan(6, 11, alpha=0.2, color='red', label='High weight (50%)')
            axes[idx].axvspan(12, 23, alpha=0.1, color='orange', label='Medium weight (30%)')
        axes[idx].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(MODEL_FIGURES_DIR / 'fig04_submission_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: fig04_submission_predictions.png")

# =============================================================================
# FIGURE 5: Prediction Distribution
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 5: Prediction Distribution")
print("="*60)

prediction_dist_json = {
    'summary': {
        'visualization': 'Log-transformed volume distribution'
    },
    'scenario_1_distribution': {},
    'scenario_2_distribution': {}
}

if len(submissions) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (scenario, df) in enumerate(submissions.items()):
        # Log scale for better visualization
        log_volume = np.log1p(df['volume'])
        
        # Calculate distribution stats
        dist_stats = {
            'mean_log': float(log_volume.mean()),
            'std_log': float(log_volume.std()),
            'median_log': float(log_volume.median()),
            'mean_original': float(np.expm1(log_volume.mean())),
            'median_original': float(np.expm1(log_volume.median()))
        }
        
        if scenario == 1:
            prediction_dist_json['scenario_1_distribution'] = dist_stats
        else:
            prediction_dist_json['scenario_2_distribution'] = dist_stats
        
        # Plot
        axes[idx].hist(log_volume, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[idx].axvline(x=log_volume.median(), color='red', linestyle='--', 
                          label=f'Median: {np.expm1(log_volume.median()):,.0f}')
        axes[idx].set_xlabel('Log(Volume + 1)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Scenario {scenario}: Predicted Volume Distribution', fontsize=12)
        axes[idx].legend()
    
    save_model_json(prediction_dist_json, 'fig05_prediction_distribution.json',
                    'Prediction distribution analysis')
    
    plt.tight_layout()
    plt.savefig(MODEL_FIGURES_DIR / 'fig05_prediction_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: fig05_prediction_distribution.png")

# =============================================================================
# COMPLETE SUMMARY
# =============================================================================

print("\n" + "="*70)
print("📊 MODEL RESULTS SUMMARY")
print("="*70)

# Save complete summary
complete_summary = {
    'metadata': {
        'generated_at': datetime.now().isoformat()
    },
    'scenario_1_best': {},
    'scenario_2_best': {},
    'submissions': {},
    'figures_generated': [
        'fig01_model_comparison.png',
        'fig02_bucket_performance.png',
        'fig03_feature_importance.png',
        'fig04_submission_predictions.png',
        'fig05_prediction_distribution.png'
    ]
}

if comparison_s1 is not None:
    best_s1 = comparison_s1.loc[comparison_s1['final_score'].idxmin()]
    complete_summary['scenario_1_best'] = {
        'model': best_s1['model'],
        'final_score': float(best_s1['final_score']),
        'bucket1_pe': float(best_s1['bucket1_pe']) if pd.notna(best_s1.get('bucket1_pe')) else None,
        'bucket2_pe': float(best_s1['bucket2_pe']) if pd.notna(best_s1.get('bucket2_pe')) else None
    }
    print(f"\n🏆 SCENARIO 1 BEST MODEL:")
    print(f"   Model: {best_s1['model']}")
    print(f"   Final Score: {best_s1['final_score']:.4f}")

if comparison_s2 is not None:
    best_s2 = comparison_s2.loc[comparison_s2['final_score'].idxmin()]
    complete_summary['scenario_2_best'] = {
        'model': best_s2['model'],
        'final_score': float(best_s2['final_score']),
        'bucket1_pe': float(best_s2['bucket1_pe']) if pd.notna(best_s2.get('bucket1_pe')) else None,
        'bucket2_pe': float(best_s2['bucket2_pe']) if pd.notna(best_s2.get('bucket2_pe')) else None
    }
    print(f"\n🏆 SCENARIO 2 BEST MODEL:")
    print(f"   Model: {best_s2['model']}")
    print(f"   Final Score: {best_s2['final_score']:.4f}")

for scenario, df in submissions.items():
    complete_summary['submissions'][f'scenario_{scenario}'] = {
        'predictions': len(df),
        'brands': df[['country', 'brand_name']].drop_duplicates().shape[0]
    }
    print(f"\n📁 SUBMISSIONS - Scenario {scenario}:")
    print(f"   Predictions: {len(df):,}")

with open(MODEL_DATA_DIR / 'model_results_complete_summary.json', 'w', encoding='utf-8') as f:
    json.dump(complete_summary, f, indent=2, ensure_ascii=False)
print(f"\n   💾 Saved: model_results_complete_summary.json")

print(f"\n✅ All data saved to: {MODEL_DATA_DIR}")
print(f"✅ All figures saved to: {MODEL_FIGURES_DIR}")
print("="*70)
print("\n📤 Ready for competition submission!")
