"""
📊 Model Results Analysis Script

This script saves detailed JSON data and CSV files for each figure.
All data files are saved to reports/03_model_data/ for later interpretation.

Usage:
    python notebooks/03_model_results.py                    # Use latest files
    python notebooks/03_model_results.py 20251129_072545    # Use specific timestamp

Configuration:
    Set RESULTS_TIMESTAMP below to use specific timestamped files,
    or pass timestamp as command-line argument.
    Leave as None or "latest" to auto-detect most recent files.
"""

# Setup - Add src to path
import sys
from pathlib import Path
import json
import re
from datetime import datetime

# =============================================================================
# CONFIGURATION: Specify timestamp to analyze specific run results
# =============================================================================
# Set to specific timestamp (e.g., "20251129_072545") or None/"latest" for most recent
# Can also be passed as command-line argument: python 03_model_results.py 20251129_072545
RESULTS_TIMESTAMP = None  # e.g., "20251129_072545" or None for latest
# =============================================================================

# Parse command-line argument for timestamp
if len(sys.argv) > 1 and sys.argv[1] not in ['--help', '-h']:
    RESULTS_TIMESTAMP = sys.argv[1]
    print(f"📅 Using timestamp from command line: {RESULTS_TIMESTAMP}")

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
# Timestamp-based File Resolution
# =============================================================================

def find_timestamped_file(directory: Path, base_pattern: str, timestamp: str = None) -> Path:
    """
    Find file matching pattern with optional timestamp.
    
    Args:
        directory: Directory to search
        base_pattern: Base filename pattern (e.g., "model_comparison_unified", "unified_lightgbm")
        timestamp: Specific timestamp (YYYYMMDD_HHMMSS) or None for latest
    
    Returns:
        Path to the matching file, or None if not found
    """
    if timestamp and timestamp.lower() != 'latest':
        # Look for exact timestamp match
        candidates = list(directory.glob(f"{base_pattern}_{timestamp}.*"))
        if candidates:
            return candidates[0]
        # Also try without extension for joblib files
        candidates = list(directory.glob(f"{base_pattern}_{timestamp}.joblib"))
        if candidates:
            return candidates[0]
    
    # Find all timestamped versions and get the most recent
    pattern = re.compile(rf"{re.escape(base_pattern)}_(\d{{8}}_\d{{6}})\.\w+$")
    timestamped_files = []
    
    for f in directory.glob(f"{base_pattern}_*"):
        match = pattern.match(f.name)
        if match:
            timestamped_files.append((match.group(1), f))
    
    if timestamped_files:
        # Sort by timestamp and return most recent
        timestamped_files.sort(key=lambda x: x[0], reverse=True)
        return timestamped_files[0][1]
    
    # Fall back to non-timestamped version
    for ext in ['.csv', '.joblib', '.json']:
        fallback = directory / f"{base_pattern}{ext}"
        if fallback.exists():
            return fallback
    
    return None


def find_timestamped_submissions(directory: Path, timestamp: str = None) -> list:
    """
    Find submission files matching timestamp.
    
    Args:
        directory: Submissions directory
        timestamp: Specific timestamp or None for latest
    
    Returns:
        List of matching submission file paths
    """
    if timestamp and timestamp.lower() != 'latest':
        # Look for exact timestamp match
        files = list(directory.glob(f"submission_*_{timestamp}.csv"))
        if files:
            return files
    
    # Find all timestamped submissions, group by model type
    pattern = re.compile(r"submission_(\w+)_(\d{8}_\d{6})\.csv$")
    model_files = {}  # model_type -> [(timestamp, path), ...]
    
    for f in directory.glob("submission_*.csv"):
        if 'template' in f.name or 'example' in f.name:
            continue
        match = pattern.match(f.name)
        if match:
            model_type, ts = match.groups()
            if model_type not in model_files:
                model_files[model_type] = []
            model_files[model_type].append((ts, f))
    
    # Get most recent for each model type
    result = []
    for model_type, files in model_files.items():
        files.sort(key=lambda x: x[0], reverse=True)
        result.append(files[0][1])
    
    # Fall back to _latest files if no timestamped found
    if not result:
        result = [f for f in directory.glob("submission_*_latest.csv")]
    
    return result


def find_timestamped_model(directory: Path, base_name: str, timestamp: str = None) -> Path:
    """Find model file with optional timestamp."""
    return find_timestamped_file(directory, base_name, timestamp)


def list_available_timestamps(directory: Path, base_pattern: str) -> list:
    """List all available timestamps for a file pattern."""
    pattern = re.compile(rf"{re.escape(base_pattern)}_(\d{{8}}_\d{{6}})\.\w+$")
    timestamps = set()
    
    for f in directory.glob(f"{base_pattern}_*"):
        match = pattern.match(f.name)
        if match:
            timestamps.add(match.group(1))
    
    return sorted(timestamps, reverse=True)


# Show available timestamps
print("\n" + "="*60)
print("📅 TIMESTAMP CONFIGURATION")
print("="*60)

available_ts = list_available_timestamps(REPORTS_DIR, "model_comparison_unified")
if available_ts:
    print(f"Available unified result timestamps: {available_ts}")

if RESULTS_TIMESTAMP:
    print(f"✅ Using specified timestamp: {RESULTS_TIMESTAMP}")
else:
    print("ℹ️  Using latest files (set RESULTS_TIMESTAMP or pass as argument)")

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

# Load comparison results using timestamp-aware file resolution
comparison_s1 = None
comparison_s2 = None
comparison_unified = None
loaded_files_info = {}  # Track which files were loaded

# First check for unified training mode results (with timestamp support)
unified_file = find_timestamped_file(REPORTS_DIR, "model_comparison_unified", RESULTS_TIMESTAMP)
if unified_file and unified_file.exists():
    try:
        comparison_unified = pd.read_csv(unified_file)
        loaded_files_info['unified'] = unified_file.name
        print(f"✅ Loaded Unified training results: {unified_file.name}")
        print(comparison_unified.to_string())
        # Extract S1 and S2 scores from unified results
        if 's1_score' in comparison_unified.columns and 's2_score' in comparison_unified.columns:
            comparison_s1 = comparison_unified[['model', 's1_score']].copy()
            comparison_s1.columns = ['model', 'final_score']
            comparison_s2 = comparison_unified[['model', 's2_score']].copy()
            comparison_s2.columns = ['model', 'final_score']
            print("   → Extracted S1 and S2 scores from unified results")
    except Exception as e:
        print(f"⚠️ Could not load unified results: {e}")

# Fall back to separate scenario files if unified not found
if comparison_s1 is None:
    s1_file = find_timestamped_file(REPORTS_DIR, "model_comparison_scenario1", RESULTS_TIMESTAMP)
    if s1_file and s1_file.exists():
        try:
            comparison_s1 = pd.read_csv(s1_file)
            loaded_files_info['scenario1'] = s1_file.name
            print(f"✅ Loaded Scenario 1 results: {s1_file.name}")
            print(comparison_s1.to_string())
        except Exception as e:
            print(f"⚠️ Scenario 1 results error: {e}")
    else:
        print("⚠️ Scenario 1 results not found. Run training first.")

if comparison_s2 is None:
    s2_file = find_timestamped_file(REPORTS_DIR, "model_comparison_scenario2", RESULTS_TIMESTAMP)
    if s2_file and s2_file.exists():
        try:
            comparison_s2 = pd.read_csv(s2_file)
            loaded_files_info['scenario2'] = s2_file.name
            print(f"✅ Loaded Scenario 2 results: {s2_file.name}")
            print(comparison_s2.to_string())
        except Exception as e:
            print(f"⚠️ Scenario 2 results error: {e}")
    else:
        print("⚠️ Scenario 2 results not found. Run training first.")

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
        'scenario_2_available': comparison_s2 is not None,
        'unified_mode': comparison_unified is not None
    },
    'scenario_1': {},
    'scenario_2': {},
    'unified': {},
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

if comparison_unified is not None:
    best_unified = comparison_unified.loc[comparison_unified['final_score'].idxmin()]
    model_comparison_json['unified'] = {
        'best_model': best_unified['model'],
        'best_combined_score': float(best_unified['final_score']),
        'best_s1_score': float(best_unified['s1_score']) if 's1_score' in best_unified else None,
        'best_s2_score': float(best_unified['s2_score']) if 's2_score' in best_unified else None,
        'all_models': comparison_unified.to_dict('records'),
        'model_ranking': comparison_unified.sort_values('final_score')['model'].tolist()
    }
    save_model_csv(comparison_unified, 'fig01_model_comparison_unified.csv')

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
    'scenario_2_importance': [],
    'unified_importance': []
}

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Check for unified model first, then separate scenario models (with timestamp support)
model_paths = []
unified_model_path = find_timestamped_model(MODELS_DIR, "unified_lightgbm", RESULTS_TIMESTAMP)
if unified_model_path and unified_model_path.exists():
    # Unified mode: show same model in both panels
    model_paths = [(0, unified_model_path, 'unified'), (1, unified_model_path, 'unified')]
    loaded_files_info['model_lightgbm'] = unified_model_path.name
else:
    # Separate mode
    for idx, scenario in enumerate([1, 2]):
        model_path = find_timestamped_model(MODELS_DIR, f"scenario{scenario}_lightgbm", RESULTS_TIMESTAMP)
        if model_path:
            model_paths.append((idx, model_path, scenario))
            loaded_files_info[f'model_s{scenario}_lightgbm'] = model_path.name

for idx, model_path, scenario_label in model_paths:
    if model_path and model_path.exists():
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
        
        if scenario_label == 'unified':
            feature_importance_json['unified_importance'] = importance_list
            # Also populate scenario lists for compatibility
            if idx == 0:
                feature_importance_json['scenario_1_importance'] = importance_list
            else:
                feature_importance_json['scenario_2_importance'] = importance_list
        elif scenario_label == 1:
            feature_importance_json['scenario_1_importance'] = importance_list
        else:
            feature_importance_json['scenario_2_importance'] = importance_list
        
        # Save full importance to CSV
        save_model_csv(importance, f'fig03_feature_importance_{scenario_label}.csv')
        
        # Plot
        axes[idx].barh(top_15['feature'], top_15['importance'], 
                       color='steelblue', edgecolor='black')
        axes[idx].set_xlabel('Importance')
        title_suffix = '(Unified Model)' if scenario_label == 'unified' else f'(LightGBM)'
        scenario_num = idx + 1 if scenario_label == 'unified' else scenario_label
        axes[idx].set_title(f'Scenario {scenario_num}: Top 15 Features {title_suffix}', fontsize=12)
    else:
        axes[idx].text(0.5, 0.5, f'Model not found\nRun train_models.py first',
                       ha='center', va='center', transform=axes[idx].transAxes)
        scenario_num = idx + 1
        axes[idx].set_title(f'Scenario {scenario_num}: Feature Importance')

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

# Load and analyze submissions - use timestamp-aware resolution
submission_files = find_timestamped_submissions(SUBMISSIONS_DIR, RESULTS_TIMESTAMP)

# Fall back to older patterns if timestamp search returns nothing
if not submission_files:
    submission_files = list(SUBMISSIONS_DIR.glob('scenario*_*_final.csv'))
if not submission_files:
    submission_files = list(SUBMISSIONS_DIR.glob('submission_*_latest.csv'))
if not submission_files:
    submission_files = [f for f in SUBMISSIONS_DIR.glob('submission_*.csv') 
                       if 'template' not in f.name and 'example' not in f.name]

print(f"📁 Found {len(submission_files)} submission files:")
for f in submission_files:
    print(f"   - {f.name}")
    loaded_files_info[f'submission_{f.stem}'] = f.name

submissions = {}
submission_analysis_json = {
    'summary': {
        'files_found': len(submission_files),
        'timestamp_filter': RESULTS_TIMESTAMP or 'latest'
    },
    'scenario_1': {},
    'scenario_2': {},
    'unified_submissions': []
}

for f in submission_files:
    df = pd.read_csv(f)
    
    # Determine if it's a scenario-specific or unified submission
    if 'scenario1' in f.name:
        scenario = 1
        submissions[scenario] = df
    elif 'scenario2' in f.name:
        scenario = 2
        submissions[scenario] = df
    else:
        # Unified submission - contains all data (S1 + S2 brands)
        # Use as combined view (assign to scenario 1 for visualization)
        if 1 not in submissions:
            submissions[1] = df
            scenario = 1
        else:
            continue  # Skip duplicate unified submissions
    
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
    
    # Store in appropriate section
    if 'scenario1' in f.name:
        submission_analysis_json['scenario_1'] = stats
    elif 'scenario2' in f.name:
        submission_analysis_json['scenario_2'] = stats
    else:
        submission_analysis_json['unified_submissions'].append(stats)
        # Also use first unified as scenario 1 for compatibility
        if not submission_analysis_json['scenario_1']:
            submission_analysis_json['scenario_1'] = stats
    
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
    'loaded_files': loaded_files_info,
    'timestamp_used': RESULTS_TIMESTAMP or 'latest',
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
        'bucket2_pe': float(best_s2['bucket2_pe']) if pd.notna(best_s1.get('bucket2_pe')) else None
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

# Print summary of loaded files
print("\n" + "="*70)
print("📋 FILES USED FOR THIS ANALYSIS:")
print("="*70)
for key, filename in loaded_files_info.items():
    print(f"   {key}: {filename}")
if RESULTS_TIMESTAMP:
    print(f"\n   📅 Timestamp filter: {RESULTS_TIMESTAMP}")
else:
    print(f"\n   📅 Using latest available files")

print(f"\n✅ All data saved to: {MODEL_DATA_DIR}")
print(f"✅ All figures saved to: {MODEL_FIGURES_DIR}")
print("="*70)
print("\n📤 Ready for competition submission!")
