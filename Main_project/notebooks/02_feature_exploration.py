"""
📊 Feature Exploration Script

This script saves detailed JSON data and CSV files for each figure.
All data files are saved to reports/02_feature_data/ for later interpretation.

Usage:
    python notebooks/02_feature_exploration.py
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
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("✅ Setup complete!")
print(f"Project root: {project_root}")

# Import from src modules
from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns

print("✅ Modules imported successfully!")

# Create output directory for feature exploration data
FEATURE_DATA_DIR = REPORTS_DIR / '02_feature_data'
FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_FIGURES_DIR = FEATURE_DATA_DIR / 'figures'
FEATURE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Feature data will be saved to: {FEATURE_DATA_DIR}")

# =============================================================================
# Helper Functions
# =============================================================================

def save_feature_json(data: dict, filename: str, description: str = ""):
    """Save feature exploration data to JSON with proper type conversion."""
    
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to Python native types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
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
    
    filepath = FEATURE_DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Saved: {filepath.name}")
    return filepath


def save_feature_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV."""
    filepath = FEATURE_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"   💾 Saved: {filepath.name}")
    return filepath


# =============================================================================
# Load Data and Create Features
# =============================================================================

print("\n" + "="*60)
print("📂 LOADING DATA")
print("="*60)

volume, generics, medicine = load_all_data(train=True)
merged = merge_datasets(volume, generics, medicine)

# Create auxiliary file
aux_df = create_auxiliary_file(merged, save=False)
avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()

print(f"\n📊 Loaded {len(merged):,} records")

# Create all features
featured = create_all_features(merged, avg_j)
feature_cols = get_feature_columns(featured)

print(f"\n📊 Feature Engineering Results:")
print(f"   Total columns: {len(featured.columns)}")
print(f"   Feature columns: {len(feature_cols)}")

# =============================================================================
# FIGURE 1: Feature Correlation Matrix
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 1: Feature Correlation Matrix")
print("="*60)

# Select numeric features for correlation
numeric_features = featured[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# Compute correlation matrix (top 15 features)
top_features = numeric_features[:15]
corr_matrix = featured[top_features].corr()

# Prepare JSON data
corr_json_data = {
    'summary': {
        'total_features': len(feature_cols),
        'numeric_features': len(numeric_features),
        'features_in_heatmap': len(top_features),
        'feature_list': top_features
    },
    'correlation_stats': {
        'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
        'min_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
        'mean_abs_correlation': float(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean())
    },
    'high_correlations': []
}

# Find highly correlated pairs
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            corr_json_data['high_correlations'].append({
                'feature_1': top_features[i],
                'feature_2': top_features[j],
                'correlation': float(corr_val)
            })

save_feature_json(corr_json_data, 'fig01_feature_correlation.json',
                  'Feature correlation matrix analysis')
save_feature_csv(corr_matrix.reset_index(), 'fig01_feature_correlation.csv')

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (Top 15 Features)', fontsize=14)
plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig01_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig01_feature_correlation.png")

# =============================================================================
# FIGURE 2: Target Correlation
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 2: Feature-Target Correlation")
print("="*60)

# Correlation with target (volume)
target_corr = featured[numeric_features + ['volume']].corr()['volume'].drop('volume').sort_values(key=abs, ascending=False)

# Prepare JSON data
target_corr_json = {
    'summary': {
        'total_features_analyzed': len(target_corr),
        'strongest_positive': {
            'feature': target_corr.idxmax(),
            'correlation': float(target_corr.max())
        },
        'strongest_negative': {
            'feature': target_corr.idxmin(),
            'correlation': float(target_corr.min())
        }
    },
    'top_20_correlations': [
        {'feature': feat, 'correlation': float(corr)} 
        for feat, corr in target_corr.head(20).items()
    ],
    'interpretation': {
        'positive_correlation': 'Feature increases as volume increases',
        'negative_correlation': 'Feature increases as volume decreases',
        'strong_threshold': 0.5,
        'moderate_threshold': 0.3
    }
}

save_feature_json(target_corr_json, 'fig02_target_correlation.json',
                  'Features correlated with target variable (volume)')
save_feature_csv(pd.DataFrame({'feature': target_corr.index, 'correlation': target_corr.values}),
                 'fig02_target_correlation.csv')

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in target_corr.head(20).values]
target_corr.head(20).plot(kind='barh', ax=ax, color=colors, edgecolor='black')
ax.set_xlabel('Correlation with Volume')
ax.set_title('Top 20 Features by Correlation with Target (Volume)', fontsize=12)
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig02_target_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig02_target_correlation.png")

# =============================================================================
# FIGURE 3: Key Feature Distributions
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 3: Key Feature Distributions")
print("="*60)

key_features = ['months_postgx', 'volume_lag_1', 'volume_rolling_mean_3', 
                'n_gxs', 'months_with_generics', 'avg_vol',
                'volume_rolling_std_3', 'volume_lag_6', 'hospital_rate']

# Filter to existing features
key_features = [f for f in key_features if f in featured.columns]

# Calculate statistics for each feature
feature_stats = []
for col in key_features:
    data = featured[col].dropna()
    if len(data) > 0:
        feature_stats.append({
            'feature': col,
            'count': int(len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'q25': float(data.quantile(0.25)),
            'median': float(data.median()),
            'q75': float(data.quantile(0.75)),
            'max': float(data.max()),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis())
        })

dist_json_data = {
    'summary': {
        'features_analyzed': key_features,
        'total_records': len(featured)
    },
    'feature_statistics': feature_stats,
    'interpretation': {
        'skewness': 'Positive = right-skewed, Negative = left-skewed',
        'kurtosis': 'High = heavy tails, Low = light tails'
    }
}

save_feature_json(dist_json_data, 'fig03_feature_distributions.json',
                  'Distribution statistics for key features')
save_feature_csv(pd.DataFrame(feature_stats), 'fig03_feature_distributions.csv')

# Plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(key_features[:9]):
    data = featured[col].dropna()
    if len(data) > 0:
        # Clip outliers for better visualization
        q99 = data.quantile(0.99)
        data_clipped = data[data <= q99]
        
        axes[idx].hist(data_clipped, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{col}', fontsize=11)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        
        # Add statistics
        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}'
        axes[idx].text(0.95, 0.95, stats_text, transform=axes[idx].transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Key Feature Distributions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig03_feature_distributions.png")

# =============================================================================
# FIGURE 4: Lag Features Analysis
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 4: Lag Features Analysis")
print("="*60)

lag_features = [c for c in feature_cols if 'lag' in c]
print(f"📊 Lag Features: {lag_features}")

# Calculate lag feature correlations with volume
lag_correlations = []
for lag_col in lag_features:
    data = featured[[lag_col, 'volume']].dropna()
    if len(data) > 0:
        corr = data[lag_col].corr(data['volume'])
        lag_correlations.append({
            'feature': lag_col,
            'correlation': float(corr),
            'n_samples': int(len(data)),
            'mean': float(data[lag_col].mean()),
            'std': float(data[lag_col].std())
        })

lag_json_data = {
    'summary': {
        'total_lag_features': len(lag_features),
        'lag_feature_list': lag_features
    },
    'lag_correlations': lag_correlations,
    'interpretation': {
        'high_correlation_meaning': 'Strong persistence in volume over time',
        'lag_1_importance': 'Most recent volume is usually most predictive'
    }
}

save_feature_json(lag_json_data, 'fig04_lag_features.json',
                  'Lag features analysis and correlations')
save_feature_csv(pd.DataFrame(lag_correlations), 'fig04_lag_features.csv')

# Plot lag feature vs volume
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, lag_col in enumerate(lag_features[:6]):
    sample = featured[[lag_col, 'volume']].dropna().sample(min(5000, len(featured)), random_state=42)
    axes[idx].scatter(sample[lag_col], sample['volume'], alpha=0.3, s=10, color='steelblue')
    axes[idx].set_xlabel(lag_col)
    axes[idx].set_ylabel('Volume')
    axes[idx].set_title(f'{lag_col} vs Volume')
    
    # Add correlation
    corr = sample[lag_col].corr(sample['volume'])
    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[idx].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Lag Features vs Target Volume', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig04_lag_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig04_lag_features.png")

# =============================================================================
# FIGURE 5: Rolling Features Analysis
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 5: Rolling Features Analysis")
print("="*60)

rolling_features = [c for c in feature_cols if 'rolling' in c]
print(f"📊 Rolling Features: {rolling_features}")

# Calculate rolling feature statistics
rolling_stats = []
for roll_col in rolling_features:
    data = featured[roll_col].dropna()
    if len(data) > 0:
        feat_type = 'mean' if 'mean' in roll_col else ('std' if 'std' in roll_col else ('min' if 'min' in roll_col else 'max'))
        window_str = roll_col.split('_')[-1]
        window = int(window_str) if window_str.isdigit() else None
        rolling_stats.append({
            'feature': roll_col,
            'type': feat_type,
            'window': window,
            'mean': float(data.mean()),
            'std': float(data.std()),
            'correlation_with_volume': float(featured[[roll_col, 'volume']].dropna().corr().iloc[0, 1])
        })

rolling_json_data = {
    'summary': {
        'total_rolling_features': len(rolling_features),
        'rolling_feature_list': rolling_features,
        'window_sizes': sorted(set([s['window'] for s in rolling_stats if s['window']]))
    },
    'rolling_statistics': rolling_stats,
    'interpretation': {
        'rolling_mean': 'Smoothed average volume over window',
        'rolling_std': 'Volatility measure over window',
        'window_effect': 'Larger windows capture longer-term trends'
    }
}

save_feature_json(rolling_json_data, 'fig05_rolling_features.json',
                  'Rolling features analysis')
save_feature_csv(pd.DataFrame(rolling_stats), 'fig05_rolling_features.csv')

# Plot rolling features over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sample a few brands for visualization
sample_brands = featured[['country', 'brand_name']].drop_duplicates().sample(5, random_state=42)
sample_data = featured.merge(sample_brands, on=['country', 'brand_name'])

# Rolling mean 3
for (country, brand), group in sample_data.groupby(['country', 'brand_name']):
    group_sorted = group.sort_values('months_postgx')
    axes[0, 0].plot(group_sorted['months_postgx'], group_sorted['volume_rolling_mean_3'], 
                    label=f'{brand[:10]}', alpha=0.7, linewidth=1.5)
axes[0, 0].set_xlabel('Months Post GX')
axes[0, 0].set_ylabel('Rolling Mean (3 months)')
axes[0, 0].set_title('Rolling Mean (3 months) Over Time')
axes[0, 0].legend(fontsize=8)

# Rolling std 3
for (country, brand), group in sample_data.groupby(['country', 'brand_name']):
    group_sorted = group.sort_values('months_postgx')
    axes[0, 1].plot(group_sorted['months_postgx'], group_sorted['volume_rolling_std_3'], 
                    alpha=0.7, linewidth=1.5)
axes[0, 1].set_xlabel('Months Post GX')
axes[0, 1].set_ylabel('Rolling Std (3 months)')
axes[0, 1].set_title('Rolling Volatility (3 months) Over Time')

# Rolling mean distributions by window
mean_features = [c for c in rolling_features if 'mean' in c]
for feat in mean_features:
    data = featured[feat].dropna()
    data = data[data < data.quantile(0.99)]  # Remove outliers
    axes[1, 0].hist(data, bins=50, alpha=0.5, label=feat)
axes[1, 0].set_xlabel('Rolling Mean Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Rolling Mean Distributions')
axes[1, 0].legend(fontsize=8)

# Rolling std distributions
std_features = [c for c in rolling_features if 'std' in c]
for feat in std_features:
    data = featured[feat].dropna()
    data = data[data < data.quantile(0.99)]
    axes[1, 1].hist(data, bins=50, alpha=0.5, label=feat)
axes[1, 1].set_xlabel('Rolling Std Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Rolling Std Distributions')
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig05_rolling_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig05_rolling_features.png")

# =============================================================================
# FIGURE 6: Competition Features
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 6: Competition Features")
print("="*60)

competition_features = ['n_gxs', 'months_with_generics', 'has_generics', 'high_competition']
competition_features = [f for f in competition_features if f in featured.columns]

# Calculate competition impact
generics_bins = pd.cut(featured['n_gxs'], bins=[0, 2, 5, 10, 20, 100], labels=['1-2', '3-5', '6-10', '11-20', '20+'])
avg_by_generics = featured.groupby(generics_bins, observed=True)['volume'].agg(['mean', 'std', 'count']).reset_index()
avg_by_generics.columns = ['generics_bin', 'mean_volume', 'std_volume', 'count']

competition_json_data = {
    'summary': {
        'competition_features': competition_features,
        'n_gxs_stats': {
            'mean': float(featured['n_gxs'].mean()),
            'median': float(featured['n_gxs'].median()),
            'max': int(featured['n_gxs'].max()),
            'brands_with_high_competition': int((featured['n_gxs'] > 10).sum())
        }
    },
    'volume_by_competition_level': avg_by_generics.to_dict('records'),
    'correlation_n_gxs_volume': float(featured[['n_gxs', 'volume']].corr().iloc[0, 1]),
    'interpretation': {
        'general_trend': 'Higher generic competition tends to reduce brand volume',
        'saturation_point': 'Impact may plateau after certain number of generics'
    }
}

save_feature_json(competition_json_data, 'fig06_competition_features.json',
                  'Competition features analysis')
save_feature_csv(avg_by_generics, 'fig06_competition_features.csv')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Num generics vs volume
sample = featured[['n_gxs', 'volume']].dropna().sample(min(5000, len(featured)), random_state=42)
axes[0].scatter(sample['n_gxs'], sample['volume'], alpha=0.2, s=10)
axes[0].set_xlabel('Number of Generics')
axes[0].set_ylabel('Volume')
axes[0].set_title('Number of Generics vs Volume')

# Months with generics vs volume  
if 'months_with_generics' in featured.columns:
    sample = featured[['months_with_generics', 'volume']].dropna().sample(min(5000, len(featured)), random_state=42)
    axes[1].scatter(sample['months_with_generics'], sample['volume'], alpha=0.2, s=10, color='coral')
    axes[1].set_xlabel('Months with Generics')
    axes[1].set_ylabel('Volume')
    axes[1].set_title('Months with Generics vs Volume')

# Average volume by n_gxs bins
featured['generics_bin'] = pd.cut(featured['n_gxs'], bins=[0, 2, 5, 10, 20, 100], labels=['1-2', '3-5', '6-10', '11-20', '20+'])
avg_plot = featured.groupby('generics_bin', observed=True)['volume'].mean()
avg_plot.plot(kind='bar', ax=axes[2], color='seagreen', edgecolor='black')
axes[2].set_xlabel('Number of Generic Competitors')
axes[2].set_ylabel('Average Volume')
axes[2].set_title('Average Volume by Generic Competition Level')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig06_competition_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig06_competition_features.png")

# =============================================================================
# FIGURE 7: Time Features
# =============================================================================

print("\n" + "="*60)
print("📊 FIGURE 7: Time Features")
print("="*60)

# Calculate volume by month
volume_by_month = featured.groupby('months_postgx')['volume'].agg(['mean', 'std', 'count']).reset_index()
volume_by_month.columns = ['months_postgx', 'mean_volume', 'std_volume', 'count']

# Time period analysis
early_volume = featured[featured['months_postgx'].between(0, 5)]['volume'].mean()
mid_volume = featured[featured['months_postgx'].between(6, 11)]['volume'].mean()
late_volume = featured[featured['months_postgx'].between(12, 23)]['volume'].mean()

time_json_data = {
    'summary': {
        'months_range': [int(featured['months_postgx'].min()), int(featured['months_postgx'].max())],
        'total_time_points': int(featured['months_postgx'].nunique())
    },
    'volume_by_month': volume_by_month.to_dict('records'),
    'period_analysis': {
        'early_period_0_5': {
            'mean_volume': float(early_volume),
            'description': 'First 6 months after generic entry'
        },
        'mid_period_6_11': {
            'mean_volume': float(mid_volume),
            'description': 'Months 6-11 after generic entry'
        },
        'late_period_12_23': {
            'mean_volume': float(late_volume),
            'description': 'Months 12-23 after generic entry'
        },
        'erosion_early_to_late': float((early_volume - late_volume) / early_volume * 100) if early_volume > 0 else None
    },
    'interpretation': {
        'typical_pattern': 'Volume typically declines after generic entry',
        'steepest_decline': 'Usually in first 6 months'
    }
}

save_feature_json(time_json_data, 'fig07_time_features.json',
                  'Time features and volume trajectory analysis')
save_feature_csv(volume_by_month, 'fig07_time_features.csv')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Volume by months_postgx
axes[0].plot(volume_by_month['months_postgx'], volume_by_month['mean_volume'], 
             marker='o', color='steelblue', linewidth=2)
axes[0].fill_between(volume_by_month['months_postgx'], 
                     volume_by_month['mean_volume'] - volume_by_month['std_volume'],
                     volume_by_month['mean_volume'] + volume_by_month['std_volume'],
                     alpha=0.2)
axes[0].axvline(x=0, color='red', linestyle='--', label='Generic Entry')
axes[0].set_xlabel('Months Post Generic Entry')
axes[0].set_ylabel('Average Volume')
axes[0].set_title('Average Volume Over Time')
axes[0].legend()

# Period comparison
periods = ['Early\n(0-5)', 'Mid\n(6-11)', 'Late\n(12-23)']
volumes = [early_volume, mid_volume, late_volume]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
axes[1].bar(periods, volumes, color=colors, edgecolor='black')
axes[1].set_ylabel('Average Volume')
axes[1].set_title('Volume by Time Period')

# Volume change over time (erosion rate)
erosion_rates = []
for month in range(1, 24):
    prev = featured[featured['months_postgx'] == month - 1]['volume'].mean()
    curr = featured[featured['months_postgx'] == month]['volume'].mean()
    if prev > 0:
        erosion_rates.append({'month': month, 'erosion_pct': (curr - prev) / prev * 100})

erosion_df = pd.DataFrame(erosion_rates)
axes[2].bar(erosion_df['month'], erosion_df['erosion_pct'], color='coral', edgecolor='black')
axes[2].axhline(y=0, color='black', linewidth=0.5)
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Volume Change (%)')
axes[2].set_title('Month-over-Month Volume Change')

plt.tight_layout()
plt.savefig(FEATURE_FIGURES_DIR / 'fig07_time_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig07_time_features.png")

# =============================================================================
# COMPLETE SUMMARY
# =============================================================================

print("\n" + "="*60)
print("📊 FEATURE EXPLORATION SUMMARY")
print("="*60)

# Group features by type
lag_count = len([c for c in feature_cols if 'lag' in c])
rolling_count = len([c for c in feature_cols if 'rolling' in c])
competition_count = len([c for c in feature_cols if any(x in c for x in ['gxs', 'generic', 'competition'])])
time_count = len([c for c in feature_cols if any(x in c for x in ['month', 'time', 'period', 'early', 'late'])])
other_count = len(feature_cols) - lag_count - rolling_count - competition_count - time_count

# Save complete summary
complete_summary = {
    'metadata': {
        'generated_at': datetime.now().isoformat(),
        'total_records': len(featured),
        'total_brands': len(featured[['country', 'brand_name']].drop_duplicates())
    },
    'feature_summary': {
        'total_features': len(feature_cols),
        'lag_features': lag_count,
        'rolling_features': rolling_count,
        'competition_features': competition_count,
        'time_features': time_count,
        'other_features': other_count
    },
    'feature_list': feature_cols,
    'figures_generated': [
        'fig01_feature_correlation.png',
        'fig02_target_correlation.png',
        'fig03_feature_distributions.png',
        'fig04_lag_features.png',
        'fig05_rolling_features.png',
        'fig06_competition_features.png',
        'fig07_time_features.png'
    ],
    'missing_values': {
        'features_with_missing': int((featured[feature_cols].isnull().sum() > 0).sum()),
        'max_missing_pct': float(featured[feature_cols].isnull().sum().max() / len(featured) * 100)
    }
}

with open(FEATURE_DATA_DIR / 'feature_exploration_complete_summary.json', 'w', encoding='utf-8') as f:
    json.dump(complete_summary, f, indent=2, ensure_ascii=False)
print(f"   💾 Saved: feature_exploration_complete_summary.json")

print(f"\n📋 Feature Categories:")
print(f"   Lag features: {lag_count}")
print(f"   Rolling features: {rolling_count}")
print(f"   Competition features: {competition_count}")
print(f"   Time features: {time_count}")
print(f"   Other features: {other_count}")
print(f"   ─────────────────")
print(f"   TOTAL: {len(feature_cols)} features")

print(f"\n✅ All data saved to: {FEATURE_DATA_DIR}")
print(f"✅ All figures saved to: {FEATURE_FIGURES_DIR}")
print("="*60)
