# Setup - Add src to path
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path.cwd().parent
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
# %matplotlib inline  # notebook magic removed for .py execution

print("✅ Setup complete!")
print(f"Project root: {project_root}")

# Import from src modules
from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from feature_engineering import create_all_features, get_feature_columns

print("✅ Modules imported successfully!")

# Load data
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
print(f"\n📋 Feature List:")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# Select numeric features for correlation
numeric_features = featured[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# Compute correlation matrix (top 15 features)
top_features = numeric_features[:15]
corr_matrix = featured[top_features].corr()

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (Top 15 Features)', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved to {FIGURES_DIR / 'feature_correlation.png'}")

# Correlation with target (volume)
target_corr = featured[numeric_features + ['volume']].corr()['volume'].drop('volume').sort_values(key=abs, ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in target_corr.head(20).values]
target_corr.head(20).plot(kind='barh', ax=ax, color=colors, edgecolor='black')
ax.set_xlabel('Correlation with Volume')
ax.set_title('Top 20 Features by Correlation with Target (Volume)', fontsize=12)
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_target_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot distributions for key features
key_features = ['months_postgx', 'volume_lag_1', 'volume_rolling_mean_3', 
                'num_generics', 'months_with_generics', 'avg_vol',
                'volume_rolling_std_3', 'volume_lag_6', 'hospital_rate']

# Filter to existing features
key_features = [f for f in key_features if f in featured.columns]

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
plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze lag features
lag_features = [c for c in feature_cols if 'lag' in c]
print(f"📊 Lag Features: {lag_features}")

# Plot lag feature vs volume
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, lag_col in enumerate(lag_features[:6]):
    sample = featured[[lag_col, 'volume']].dropna().sample(min(5000, len(featured)))
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
plt.savefig(FIGURES_DIR / 'lag_features_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Analyze rolling features
rolling_features = [c for c in feature_cols if 'rolling' in c]
print(f"📊 Rolling Features: {rolling_features}")

# Plot rolling features over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sample a few brands for visualization
sample_brands = featured[['country', 'brand_name']].drop_duplicates().sample(5)
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
plt.savefig(FIGURES_DIR / 'rolling_features_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Competition features
competition_features = ['num_generics', 'months_with_generics', 'generics_growth_rate']
competition_features = [f for f in competition_features if f in featured.columns]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Num generics vs volume
sample = featured[['num_generics', 'volume']].dropna().sample(min(5000, len(featured)))
axes[0].scatter(sample['num_generics'], sample['volume'], alpha=0.2, s=10)
axes[0].set_xlabel('Number of Generics')
axes[0].set_ylabel('Volume')
axes[0].set_title('Number of Generics vs Volume')

# Months with generics vs volume  
if 'months_with_generics' in featured.columns:
    sample = featured[['months_with_generics', 'volume']].dropna().sample(min(5000, len(featured)))
    axes[1].scatter(sample['months_with_generics'], sample['volume'], alpha=0.2, s=10, color='coral')
    axes[1].set_xlabel('Months with Generics')
    axes[1].set_ylabel('Volume')
    axes[1].set_title('Months with Generics vs Volume')

# Average volume by num_generics bins
featured['generics_bin'] = pd.cut(featured['num_generics'], bins=[0, 2, 5, 10, 20, 100], labels=['1-2', '3-5', '6-10', '11-20', '20+'])
avg_by_generics = featured.groupby('generics_bin')['volume'].mean()
avg_by_generics.plot(kind='bar', ax=axes[2], color='seagreen', edgecolor='black')
axes[2].set_xlabel('Number of Generic Competitors')
axes[2].set_ylabel('Average Volume')
axes[2].set_title('Average Volume by Generic Competition Level')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'competition_features.png', dpi=150, bbox_inches='tight')
plt.show()

# Time features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Volume by months_postgx
volume_by_month = featured.groupby('months_postgx')['volume'].mean()
axes[0].plot(volume_by_month.index, volume_by_month.values, marker='o', color='steelblue', linewidth=2)
axes[0].axvline(x=0, color='red', linestyle='--', label='Generic Entry')
axes[0].set_xlabel('Months Post Generic Entry')
axes[0].set_ylabel('Average Volume')
axes[0].set_title('Average Volume Over Time')
axes[0].legend()

# Volume by month_sin/cos (seasonality)
if 'month_sin' in featured.columns:
    sample = featured[['month_sin', 'month_cos', 'volume']].dropna()
    axes[1].scatter(sample['month_sin'], sample['volume'], alpha=0.1, s=5, label='month_sin')
    axes[1].set_xlabel('Month Sin (Seasonality)')
    axes[1].set_ylabel('Volume')
    axes[1].set_title('Seasonality Effect on Volume')

# Post-entry month distribution
if 'is_early_postgx' in featured.columns:
    early_vs_late = featured.groupby('is_early_postgx')['volume'].mean()
    labels = ['Late (6-23)', 'Early (0-5)']
    axes[2].bar(range(len(early_vs_late)), early_vs_late.values, color=['coral', 'seagreen'], edgecolor='black')
    axes[2].set_xticks(range(len(early_vs_late)))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Average Volume')
    axes[2].set_title('Average Volume: Early vs Late Post-Entry')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'time_features.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature statistics summary
print("="*60)
print("📊 FEATURE ENGINEERING SUMMARY")
print("="*60)

# Group features by type
lag_count = len([c for c in feature_cols if 'lag' in c])
rolling_count = len([c for c in feature_cols if 'rolling' in c])
competition_count = len([c for c in feature_cols if any(x in c for x in ['generic', 'num_'])])
time_count = len([c for c in feature_cols if any(x in c for x in ['month', 'year', 'sin', 'cos'])])
other_count = len(feature_cols) - lag_count - rolling_count - competition_count - time_count

print(f"\n📋 Feature Categories:")
print(f"   Lag features: {lag_count}")
print(f"   Rolling features: {rolling_count}")
print(f"   Competition features: {competition_count}")
print(f"   Time features: {time_count}")
print(f"   Other features: {other_count}")
print(f"   ─────────────────")
print(f"   TOTAL: {len(feature_cols)} features")

# Missing value summary
print(f"\n📊 Missing Values:")
missing = featured[feature_cols].isnull().sum()
missing_pct = (missing / len(featured) * 100).round(1)
missing_summary = pd.DataFrame({'missing': missing, 'pct': missing_pct})
missing_summary = missing_summary[missing_summary['missing'] > 0].sort_values('pct', ascending=False)
if len(missing_summary) > 0:
    print(missing_summary.head(10))
else:
    print("   No missing values!")

print(f"\n✅ All figures saved to: {FIGURES_DIR}")
print("="*60)
