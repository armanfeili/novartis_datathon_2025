"""
📊 EDA Visualization Script

This script saves detailed JSON data for each figure BEFORE plotting.
All JSON files are saved to reports/eda_data/ for later interpretation.

Usage:
    python notebooks/01_eda_visualization.py
"""

# %% [markdown]
# # 📊 EDA Visualization Notebook
#
# This notebook saves detailed data (JSON/CSV) before each visualization.
# Data files are saved to `reports/eda_data/` for later AI interpretation.

# %%
# Setup and imports
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import create_auxiliary_file
from eda_analysis import run_full_eda

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create EDA data directory for JSON files
EDA_DATA_DIR = REPORTS_DIR / '01_eda_data'
EDA_DATA_DIR.mkdir(parents=True, exist_ok=True)
EDA_FIGURES_DIR = EDA_DATA_DIR / 'figures'
EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Setup complete!")
print(f"📁 EDA data will be saved to: {EDA_DATA_DIR}")

# %%
# Helper function to save JSON with numpy/pandas type conversion
def save_eda_json(data: dict, filename: str, description: str = ""):
    """Save EDA data to JSON with proper type conversion."""
    
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
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    output = {
        'metadata': {
            'filename': filename,
            'description': description,
            'generated_at': datetime.now().isoformat(),
            'figure_file': f"{filename.replace('.json', '.png')}"
        },
        'data': convert_to_serializable(data)
    }
    
    filepath = EDA_DATA_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   💾 Saved: {filepath.name}")
    return filepath

# %% [markdown]
# ## 1. Load Data and Run EDA Analysis

# %%
# Load data
print("\n" + "="*60)
print("📂 LOADING DATA")
print("="*60)

volume, generics, medicine = load_all_data(train=True)
merged = merge_datasets(volume, generics, medicine)

# Create auxiliary file
aux_df = create_auxiliary_file(merged, save=True)

# Run comprehensive EDA
eda_results = run_full_eda(merged, aux_df, save_report=True)

# %% [markdown]
# ## 2. Bucket Distribution Visualization

# %%
print("\n" + "="*60)
print("📊 FIGURE 1: Bucket Distribution")
print("="*60)

bucket_data = eda_results['bucket_distribution']

# Prepare detailed data for JSON
bucket_json_data = {
    'summary': {
        'total_brands': int(bucket_data['count'].sum()),
        'bucket_1_count': int(bucket_data[bucket_data['bucket'] == 1]['count'].values[0]),
        'bucket_2_count': int(bucket_data[bucket_data['bucket'] == 2]['count'].values[0]),
        'bucket_1_percentage': float(bucket_data[bucket_data['bucket'] == 1]['percentage'].values[0]),
        'bucket_2_percentage': float(bucket_data[bucket_data['bucket'] == 2]['percentage'].values[0]),
    },
    'interpretation': {
        'bucket_1_definition': 'High erosion brands - mean normalized volume <= 0.25 in months 18-23',
        'bucket_2_definition': 'Lower erosion brands - mean normalized volume > 0.25 in months 18-23',
        'weight_info': 'Bucket 1 has 2x weight in the evaluation metric',
        'imbalance_ratio': float(bucket_data[bucket_data['bucket'] == 2]['count'].values[0] / 
                                 bucket_data[bucket_data['bucket'] == 1]['count'].values[0])
    },
    'raw_data': bucket_data.to_dict('records')
}

save_eda_json(bucket_json_data, 'fig01_bucket_distribution.json',
              'Distribution of brands across erosion buckets')

# Save CSV data
bucket_data.to_csv(EDA_DATA_DIR / 'fig01_bucket_distribution.csv', index=False)
print(f"   💾 Saved: fig01_bucket_distribution.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#ff6b6b', '#4ecdc4']

axes[0].pie(bucket_data['count'], labels=[f"Bucket {int(b)}" for b in bucket_data['bucket']], 
            autopct='%1.1f%%', colors=colors, explode=[0.05, 0])
axes[0].set_title('Bucket Distribution\n(Bucket 1 = High Erosion, Weight 2×)', fontsize=12)

axes[1].bar(bucket_data['bucket'].astype(int).astype(str), bucket_data['count'], color=colors)
axes[1].set_xlabel('Bucket')
axes[1].set_ylabel('Number of Brands')
axes[1].set_title('Brands per Bucket')
for i, (b, c) in enumerate(zip(bucket_data['bucket'], bucket_data['count'])):
    axes[1].text(i, c + 5, str(int(c)), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig(EDA_FIGURES_DIR / 'fig01_bucket_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved figure: fig01_bucket_distribution.png")

# %% [markdown]
# ## 3. Erosion Curves by Bucket

# %%
print("\n" + "="*60)
print("📊 FIGURE 2: Erosion Curves by Bucket")
print("="*60)

erosion_curves = eda_results['erosion_curves']

# Compute key metrics for JSON
bucket1_curves = erosion_curves[erosion_curves['bucket'] == 1]
bucket2_curves = erosion_curves[erosion_curves['bucket'] == 2]

erosion_json_data = {
    'summary': {
        'bucket_1': {
            'initial_vol_norm_month0': float(bucket1_curves[bucket1_curves['months_postgx'] == 0]['mean_vol_norm'].values[0]) if len(bucket1_curves[bucket1_curves['months_postgx'] == 0]) > 0 else None,
            'final_vol_norm_month23': float(bucket1_curves[bucket1_curves['months_postgx'] == 23]['mean_vol_norm'].values[0]) if len(bucket1_curves[bucket1_curves['months_postgx'] == 23]) > 0 else None,
            'total_erosion_pct': None,  # Will calculate
            'mean_monthly_decline': float(bucket1_curves['mean_vol_norm'].diff().mean()) if len(bucket1_curves) > 1 else None,
        },
        'bucket_2': {
            'initial_vol_norm_month0': float(bucket2_curves[bucket2_curves['months_postgx'] == 0]['mean_vol_norm'].values[0]) if len(bucket2_curves[bucket2_curves['months_postgx'] == 0]) > 0 else None,
            'final_vol_norm_month23': float(bucket2_curves[bucket2_curves['months_postgx'] == 23]['mean_vol_norm'].values[0]) if len(bucket2_curves[bucket2_curves['months_postgx'] == 23]) > 0 else None,
            'total_erosion_pct': None,
            'mean_monthly_decline': float(bucket2_curves['mean_vol_norm'].diff().mean()) if len(bucket2_curves) > 1 else None,
        }
    },
    'interpretation': {
        'vol_norm_meaning': 'Normalized volume = actual volume / avg_j (pre-entry average)',
        'vol_norm_1.0': 'Value of 1.0 means volume equals pre-entry baseline',
        'bucket_1_threshold': 'Bucket 1 = final vol_norm <= 0.25 (75%+ volume loss)',
        'key_observation': 'Steeper decline = faster generic erosion'
    },
    'raw_data': {
        'bucket_1': bucket1_curves.to_dict('records'),
        'bucket_2': bucket2_curves.to_dict('records')
    },
    'statistics': {
        'bucket_1_vol_norm_stats': {
            'mean': float(bucket1_curves['mean_vol_norm'].mean()),
            'min': float(bucket1_curves['mean_vol_norm'].min()),
            'max': float(bucket1_curves['mean_vol_norm'].max()),
            'std_range': [float(bucket1_curves['std_vol_norm'].min()), 
                         float(bucket1_curves['std_vol_norm'].max())]
        },
        'bucket_2_vol_norm_stats': {
            'mean': float(bucket2_curves['mean_vol_norm'].mean()),
            'min': float(bucket2_curves['mean_vol_norm'].min()),
            'max': float(bucket2_curves['mean_vol_norm'].max()),
            'std_range': [float(bucket2_curves['std_vol_norm'].min()), 
                         float(bucket2_curves['std_vol_norm'].max())]
        }
    }
}

# Calculate total erosion percentages
if erosion_json_data['summary']['bucket_1']['initial_vol_norm_month0'] and erosion_json_data['summary']['bucket_1']['final_vol_norm_month23']:
    erosion_json_data['summary']['bucket_1']['total_erosion_pct'] = round(
        (1 - erosion_json_data['summary']['bucket_1']['final_vol_norm_month23'] / 
         erosion_json_data['summary']['bucket_1']['initial_vol_norm_month0']) * 100, 2)

if erosion_json_data['summary']['bucket_2']['initial_vol_norm_month0'] and erosion_json_data['summary']['bucket_2']['final_vol_norm_month23']:
    erosion_json_data['summary']['bucket_2']['total_erosion_pct'] = round(
        (1 - erosion_json_data['summary']['bucket_2']['final_vol_norm_month23'] / 
         erosion_json_data['summary']['bucket_2']['initial_vol_norm_month0']) * 100, 2)

save_eda_json(erosion_json_data, 'fig02_erosion_curves.json',
              'Erosion curves showing volume decline over time by bucket')

# Save CSV data
erosion_curves.to_csv(EDA_DATA_DIR / 'fig02_erosion_curves.csv', index=False)
print(f"   💾 Saved: fig02_erosion_curves.csv")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

for bucket in [1, 2]:
    data = erosion_curves[erosion_curves['bucket'] == bucket]
    label = f"Bucket {bucket}" + (" (High Erosion - 2× weight)" if bucket == 1 else " (Lower Erosion)")
    color = '#ff6b6b' if bucket == 1 else '#4ecdc4'
    
    ax.plot(data['months_postgx'], data['mean_vol_norm'], label=label, linewidth=2, color=color)
    ax.fill_between(data['months_postgx'], 
                    data['mean_vol_norm'] - data['std_vol_norm'],
                    data['mean_vol_norm'] + data['std_vol_norm'],
                    alpha=0.2, color=color)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Pre-entry baseline')
ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5, label='Bucket 1 threshold')
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, label='Generic entry')

ax.set_xlabel('Months Since Generic Entry', fontsize=12)
ax.set_ylabel('Normalized Volume (Volume / Avg_j)', fontsize=12)
ax.set_title('Generic Erosion Curves by Bucket', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim(-1, 24)
ax.set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig(EDA_FIGURES_DIR / 'fig02_erosion_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Sample Brand Trajectories

# %%
print("\n" + "="*60)
print("📊 FIGURE 3: Sample Brand Trajectories")
print("="*60)

sample_brands = merged[['country', 'brand_name']].drop_duplicates().sample(6, random_state=42)

# Prepare detailed data for JSON
sample_trajectories_data = {
    'summary': {
        'n_samples': 6,
        'random_seed': 42,
        'purpose': 'Visual examples of individual brand volume trajectories'
    },
    'brands': []
}

for idx, (_, row) in enumerate(sample_brands.iterrows()):
    brand_data = merged[(merged['country'] == row['country']) & 
                        (merged['brand_name'] == row['brand_name'])].sort_values('months_postgx')
    
    # Get bucket info
    brand_aux = aux_df[(aux_df['country'] == row['country']) & 
                       (aux_df['brand_name'] == row['brand_name'])]
    bucket = int(brand_aux['bucket'].values[0]) if len(brand_aux) > 0 else None
    avg_vol = float(brand_aux['avg_vol'].values[0]) if len(brand_aux) > 0 else None
    
    brand_info = {
        'country': row['country'],
        'brand_name': row['brand_name'],
        'bucket': bucket,
        'avg_vol_pre_entry': avg_vol,
        'n_months': len(brand_data),
        'months_range': [int(brand_data['months_postgx'].min()), int(brand_data['months_postgx'].max())],
        'volume_stats': {
            'min': float(brand_data['volume'].min()),
            'max': float(brand_data['volume'].max()),
            'mean': float(brand_data['volume'].mean()),
            'pre_entry_mean': float(brand_data[brand_data['months_postgx'] < 0]['volume'].mean()) if len(brand_data[brand_data['months_postgx'] < 0]) > 0 else None,
            'post_entry_mean': float(brand_data[brand_data['months_postgx'] >= 0]['volume'].mean()) if len(brand_data[brand_data['months_postgx'] >= 0]) > 0 else None,
        },
        'trajectory': brand_data[['months_postgx', 'volume']].to_dict('records')
    }
    sample_trajectories_data['brands'].append(brand_info)

save_eda_json(sample_trajectories_data, 'fig03_sample_trajectories.json',
              'Individual brand volume trajectories as examples')

# Save CSV data - all sample brand trajectories combined
sample_traj_list = []
for brand_info in sample_trajectories_data['brands']:
    for point in brand_info['trajectory']:
        sample_traj_list.append({
            'country': brand_info['country'],
            'brand_name': brand_info['brand_name'],
            'bucket': brand_info['bucket'],
            'months_postgx': point['months_postgx'],
            'volume': point['volume']
        })
pd.DataFrame(sample_traj_list).to_csv(EDA_DATA_DIR / 'fig03_sample_trajectories.csv', index=False)
print(f"   💾 Saved: fig03_sample_trajectories.csv")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (_, row) in enumerate(sample_brands.iterrows()):
    brand_data = merged[(merged['country'] == row['country']) & 
                        (merged['brand_name'] == row['brand_name'])].sort_values('months_postgx')
    
    ax = axes[idx]
    ax.plot(brand_data['months_postgx'], brand_data['volume'], 'b-', linewidth=2)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Generic Entry')
    ax.set_title(f"{row['country']} - {row['brand_name'][:15]}", fontsize=10)
    ax.set_xlabel('Months Post GX')
    ax.set_ylabel('Volume')
    
    if idx == 0:
        ax.legend()

plt.suptitle('Sample Brand Volume Trajectories', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(EDA_FIGURES_DIR / 'fig03_sample_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. n_gxs Impact on Erosion

# %%
print("\n" + "="*60)
print("📊 FIGURE 4: Generic Competition Impact")
print("="*60)

n_gxs_impact = eda_results['n_gxs_impact']
comp_traj = eda_results['competition_trajectory']

# Prepare detailed data for JSON
competition_json_data = {
    'summary': {
        'n_gxs_range': [int(n_gxs_impact['n_gxs'].min()), int(n_gxs_impact['n_gxs'].max())],
        'mean_competitors_at_entry': float(comp_traj[comp_traj['months_postgx'] == 0]['mean_n_gxs'].values[0]) if len(comp_traj[comp_traj['months_postgx'] == 0]) > 0 else None,
        'mean_competitors_at_month23': float(comp_traj[comp_traj['months_postgx'] == 23]['mean_n_gxs'].values[0]) if len(comp_traj[comp_traj['months_postgx'] == 23]) > 0 else None,
    },
    'interpretation': {
        'n_gxs_meaning': 'Number of generic competitors in the market',
        'expected_pattern': 'More competitors typically leads to lower normalized volume (more erosion)',
        'trajectory_meaning': 'Shows how competition builds up over time after generic entry'
    },
    'erosion_by_n_gxs': n_gxs_impact.to_dict('records'),
    'competition_trajectory': comp_traj.to_dict('records'),
    'correlation_analysis': {
        'correlation_n_gxs_vs_vol_norm': float(np.corrcoef(n_gxs_impact['n_gxs'], n_gxs_impact['mean_vol_norm'])[0, 1]) if len(n_gxs_impact) > 1 else None,
        'vol_norm_at_0_competitors': float(n_gxs_impact[n_gxs_impact['n_gxs'] == 0]['mean_vol_norm'].values[0]) if len(n_gxs_impact[n_gxs_impact['n_gxs'] == 0]) > 0 else None,
        'vol_norm_at_max_competitors': float(n_gxs_impact[n_gxs_impact['n_gxs'] == n_gxs_impact['n_gxs'].max()]['mean_vol_norm'].values[0]) if len(n_gxs_impact) > 0 else None,
    }
}

save_eda_json(competition_json_data, 'fig04_competition_impact.json',
              'Impact of generic competition on brand volume erosion')

# Save CSV data
n_gxs_impact.to_csv(EDA_DATA_DIR / 'fig04_n_gxs_impact.csv', index=False)
comp_traj.to_csv(EDA_DATA_DIR / 'fig04_competition_trajectory.csv', index=False)
print(f"   💾 Saved: fig04_n_gxs_impact.csv")
print(f"   💾 Saved: fig04_competition_trajectory.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(n_gxs_impact['n_gxs'], n_gxs_impact['mean_vol_norm'], color='steelblue', alpha=0.7)
axes[0].set_xlabel('Number of Generics (n_gxs)', fontsize=12)
axes[0].set_ylabel('Mean Normalized Volume', fontsize=12)
axes[0].set_title('Erosion vs Number of Generic Competitors', fontsize=12)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

axes[1].plot(comp_traj['months_postgx'], comp_traj['mean_n_gxs'], 'g-', linewidth=2, label='Mean')
axes[1].fill_between(comp_traj['months_postgx'],
                     comp_traj['mean_n_gxs'] - comp_traj['std_n_gxs'],
                     comp_traj['mean_n_gxs'] + comp_traj['std_n_gxs'],
                     alpha=0.2, color='green')
axes[1].set_xlabel('Months Since Generic Entry', fontsize=12)
axes[1].set_ylabel('Number of Generics', fontsize=12)
axes[1].set_title('Generic Competition Over Time', fontsize=12)

plt.tight_layout()
plt.savefig(EDA_FIGURES_DIR / 'fig04_competition_impact.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Therapeutic Area Analysis

# %%
print("\n" + "="*60)
print("📊 FIGURE 5: Therapeutic Area Analysis")
print("="*60)

ther_analysis = eda_results['ther_area_analysis']

if len(ther_analysis) > 0:
    # Sort by mean erosion
    ther_analysis_sorted = ther_analysis.sort_values('mean_erosion')
    
    # Prepare detailed data for JSON
    ther_json_data = {
        'summary': {
            'n_therapeutic_areas': len(ther_analysis),
            'most_eroded_area': ther_analysis_sorted.iloc[0]['ther_area'],
            'most_eroded_value': float(ther_analysis_sorted.iloc[0]['mean_erosion']),
            'least_eroded_area': ther_analysis_sorted.iloc[-1]['ther_area'],
            'least_eroded_value': float(ther_analysis_sorted.iloc[-1]['mean_erosion']),
            'erosion_range': [float(ther_analysis['mean_erosion'].min()), 
                             float(ther_analysis['mean_erosion'].max())]
        },
        'interpretation': {
            'mean_erosion_meaning': 'Average normalized volume (lower = more erosion)',
            'bucket_1_threshold': 0.25,
            'areas_below_threshold': ther_analysis_sorted[ther_analysis_sorted['mean_erosion'] <= 0.25]['ther_area'].tolist()
        },
        'raw_data': ther_analysis_sorted.to_dict('records'),
        'statistics': {
            'mean_across_areas': float(ther_analysis['mean_erosion'].mean()),
            'std_across_areas': float(ther_analysis['mean_erosion'].std()),
            'median_across_areas': float(ther_analysis['mean_erosion'].median())
        }
    }
    
    save_eda_json(ther_json_data, 'fig05_therapeutic_areas.json',
                  'Erosion patterns by therapeutic area')
    
    # Save CSV data
    ther_analysis_sorted.to_csv(EDA_DATA_DIR / 'fig05_therapeutic_areas.csv', index=False)
    print(f"   💾 Saved: fig05_therapeutic_areas.csv")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ther_analysis_sorted)))
    ax.barh(ther_analysis_sorted['ther_area'], ther_analysis_sorted['mean_erosion'], color=colors)
    ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='Bucket 1 threshold')
    ax.set_xlabel('Mean Normalized Volume (Lower = More Erosion)', fontsize=12)
    ax.set_title('Erosion by Therapeutic Area', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / 'fig05_therapeutic_areas.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️ No therapeutic area data available")

# %% [markdown]
# ## 7. Biological vs Small Molecule

# %%
print("\n" + "="*60)
print("📊 FIGURE 6: Biological vs Small Molecule")
print("="*60)

bio_vs_small = eda_results['bio_vs_small']

if len(bio_vs_small) > 0:
    # Prepare detailed data for JSON
    bio_json_data = {
        'summary': {
            'drug_types_compared': list(bio_vs_small.columns),
            'months_range': [int(bio_vs_small.index.min()), int(bio_vs_small.index.max())],
        },
        'interpretation': {
            'biological_drugs': 'Large molecule drugs (proteins, antibodies) - typically harder to replicate',
            'small_molecule_drugs': 'Traditional chemical drugs - easier to create generics',
            'expected_pattern': 'Biologicals may show slower erosion due to biosimilar complexity'
        },
        'statistics': {},
        'raw_data': bio_vs_small.reset_index().to_dict('records')
    }
    
    # Calculate stats for each drug type
    for col in bio_vs_small.columns:
        bio_json_data['statistics'][col] = {
            'mean_vol_norm': float(bio_vs_small[col].mean()),
            'final_vol_norm': float(bio_vs_small[col].iloc[-1]) if len(bio_vs_small) > 0 else None,
            'initial_vol_norm': float(bio_vs_small[col].iloc[0]) if len(bio_vs_small) > 0 else None,
            'total_erosion_pct': float((1 - bio_vs_small[col].iloc[-1] / bio_vs_small[col].iloc[0]) * 100) if len(bio_vs_small) > 0 and bio_vs_small[col].iloc[0] != 0 else None
        }
    
    save_eda_json(bio_json_data, 'fig06_biological_vs_small.json',
                  'Comparison of erosion between biological and small molecule drugs')
    
    # Save CSV data
    bio_vs_small.reset_index().to_csv(EDA_DATA_DIR / 'fig06_biological_vs_small.csv', index=False)
    print(f"   💾 Saved: fig06_biological_vs_small.csv")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bio_vs_small.plot(ax=ax, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Months Since Generic Entry', fontsize=12)
    ax.set_ylabel('Mean Normalized Volume', fontsize=12)
    ax.set_title('Erosion: Biological vs Small Molecule Drugs', fontsize=14)
    ax.legend(title='Drug Type')
    
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / 'fig06_biological_vs_small.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️ No biological/small molecule data available")

# %% [markdown]
# ## 8. Hospital Rate Impact

# %%
print("\n" + "="*60)
print("📊 FIGURE 7: Hospital Rate Impact")
print("="*60)

hospital_analysis = eda_results['hospital_rate_analysis']

if len(hospital_analysis) > 0:
    # Prepare detailed data for JSON
    hospital_json_data = {
        'summary': {
            'hospital_rate_bins': list(hospital_analysis.columns),
            'months_range': [int(hospital_analysis.index.min()), int(hospital_analysis.index.max())],
        },
        'interpretation': {
            'hospital_rate_meaning': 'Percentage of drug sales through hospital channel',
            '0-25%': 'Primarily retail/pharmacy distribution',
            '75-100%': 'Primarily hospital distribution',
            'expected_pattern': 'Hospital drugs may have different erosion patterns due to tender processes'
        },
        'statistics': {},
        'raw_data': hospital_analysis.reset_index().to_dict('records')
    }
    
    # Calculate stats for each hospital rate bin
    for col in hospital_analysis.columns:
        if hospital_analysis[col].notna().any():
            bio_json_data['statistics'][str(col)] = {
                'mean_vol_norm': float(hospital_analysis[col].mean()),
                'final_vol_norm': float(hospital_analysis[col].dropna().iloc[-1]) if len(hospital_analysis[col].dropna()) > 0 else None,
            }
    
    save_eda_json(hospital_json_data, 'fig07_hospital_rate.json',
                  'Impact of hospital distribution rate on erosion')
    
    # Save CSV data
    hospital_analysis.reset_index().to_csv(EDA_DATA_DIR / 'fig07_hospital_rate.csv', index=False)
    print(f"   💾 Saved: fig07_hospital_rate.csv")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    hospital_analysis.plot(ax=ax, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Months Since Generic Entry', fontsize=12)
    ax.set_ylabel('Mean Normalized Volume', fontsize=12)
    ax.set_title('Erosion by Hospital Rate', fontsize=14)
    ax.legend(title='Hospital Rate')
    
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / 'fig07_hospital_rate.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("   ⚠️ No hospital rate data available")

# %% [markdown]
# ## 9. Erosion Speed Analysis

# %%
print("\n" + "="*60)
print("📊 FIGURE 8: Erosion Speed Analysis")
print("="*60)

erosion_speed = eda_results['erosion_speed']

# Prepare detailed data for JSON
speed_json_data = {
    'summary': {
        'total_brands_analyzed': len(erosion_speed),
        'bucket_1_brands': int((erosion_speed['bucket'] == 1).sum()),
        'bucket_2_brands': int((erosion_speed['bucket'] == 2).sum()),
    },
    'metrics_explained': {
        'time_to_50pct': 'Months until normalized volume drops to 50% (0.5)',
        'erosion_first_6m': 'Average erosion rate in first 6 months (1 - mean_vol_norm)',
        'final_equilibrium': 'Mean normalized volume in months 18-23 (final stable level)'
    },
    'bucket_1_statistics': {
        'time_to_50pct': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].mean()) if erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].median()) if erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].notna().any() else None,
            'min': float(erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].min()) if erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].notna().any() else None,
            'max': float(erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].max()) if erosion_speed[erosion_speed['bucket'] == 1]['time_to_50pct'].notna().any() else None,
        },
        'erosion_first_6m': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 1]['erosion_first_6m'].mean()) if erosion_speed[erosion_speed['bucket'] == 1]['erosion_first_6m'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 1]['erosion_first_6m'].median()) if erosion_speed[erosion_speed['bucket'] == 1]['erosion_first_6m'].notna().any() else None,
        },
        'final_equilibrium': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 1]['final_equilibrium'].mean()) if erosion_speed[erosion_speed['bucket'] == 1]['final_equilibrium'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 1]['final_equilibrium'].median()) if erosion_speed[erosion_speed['bucket'] == 1]['final_equilibrium'].notna().any() else None,
        }
    },
    'bucket_2_statistics': {
        'time_to_50pct': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].mean()) if erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].median()) if erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].notna().any() else None,
            'min': float(erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].min()) if erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].notna().any() else None,
            'max': float(erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].max()) if erosion_speed[erosion_speed['bucket'] == 2]['time_to_50pct'].notna().any() else None,
        },
        'erosion_first_6m': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 2]['erosion_first_6m'].mean()) if erosion_speed[erosion_speed['bucket'] == 2]['erosion_first_6m'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 2]['erosion_first_6m'].median()) if erosion_speed[erosion_speed['bucket'] == 2]['erosion_first_6m'].notna().any() else None,
        },
        'final_equilibrium': {
            'mean': float(erosion_speed[erosion_speed['bucket'] == 2]['final_equilibrium'].mean()) if erosion_speed[erosion_speed['bucket'] == 2]['final_equilibrium'].notna().any() else None,
            'median': float(erosion_speed[erosion_speed['bucket'] == 2]['final_equilibrium'].median()) if erosion_speed[erosion_speed['bucket'] == 2]['final_equilibrium'].notna().any() else None,
        }
    },
    'interpretation': {
        'bucket_1_pattern': 'Fast erosion: quick time to 50%, high early erosion, low final equilibrium',
        'bucket_2_pattern': 'Slower erosion: longer time to 50% (or never), lower early erosion, higher final equilibrium',
        'key_difference': 'Bucket 1 brands lose volume much faster and stabilize at lower levels'
    }
}

# Save CSV with full erosion speed data
erosion_speed.to_csv(EDA_DATA_DIR / 'fig08_erosion_speed_full.csv', index=False)
print(f"   💾 Saved: fig08_erosion_speed_full.csv")

save_eda_json(speed_json_data, 'fig08_erosion_speed.json',
              'Erosion speed metrics by bucket')

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Time to 50% erosion
for bucket in [1, 2]:
    data = erosion_speed[erosion_speed['bucket'] == bucket]['time_to_50pct'].dropna()
    color = '#ff6b6b' if bucket == 1 else '#4ecdc4'
    axes[0].hist(data, bins=20, alpha=0.6, label=f'Bucket {bucket}', color=color)
axes[0].set_xlabel('Months to 50% Erosion')
axes[0].set_title('Time to 50% Volume Loss')
axes[0].legend()

# Erosion in first 6 months
for bucket in [1, 2]:
    data = erosion_speed[erosion_speed['bucket'] == bucket]['erosion_first_6m'].dropna()
    color = '#ff6b6b' if bucket == 1 else '#4ecdc4'
    axes[1].hist(data, bins=20, alpha=0.6, label=f'Bucket {bucket}', color=color)
axes[1].set_xlabel('Erosion Rate (1 - mean vol_norm)')
axes[1].set_title('Erosion in First 6 Months')
axes[1].legend()

# Final equilibrium
for bucket in [1, 2]:
    data = erosion_speed[erosion_speed['bucket'] == bucket]['final_equilibrium'].dropna()
    color = '#ff6b6b' if bucket == 1 else '#4ecdc4'
    axes[2].hist(data, bins=20, alpha=0.6, label=f'Bucket {bucket}', color=color)
axes[2].set_xlabel('Final Normalized Volume (months 18-23)')
axes[2].set_title('Final Equilibrium Level')
axes[2].legend()

plt.tight_layout()
plt.savefig(EDA_FIGURES_DIR / 'fig08_erosion_speed.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Summary Statistics

# %%
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

# Create comprehensive summary JSON
summary_json_data = {
    'dataset_overview': {
        'shape': list(eda_results['data_summary']['shape']),
        'total_brands': int(eda_results['data_summary']['n_brands']),
        'countries': int(eda_results['data_summary']['n_countries']),
        'months_range': [int(x) for x in eda_results['data_summary']['months_postgx_range']],
    },
    'bucket_distribution': {
        'bucket_1_count': int(bucket_data[bucket_data['bucket'] == 1]['count'].values[0]),
        'bucket_1_pct': float(bucket_data[bucket_data['bucket'] == 1]['percentage'].values[0]),
        'bucket_2_count': int(bucket_data[bucket_data['bucket'] == 2]['count'].values[0]),
        'bucket_2_pct': float(bucket_data[bucket_data['bucket'] == 2]['percentage'].values[0]),
        'imbalance_note': 'Highly imbalanced - Bucket 2 dominates but Bucket 1 has 2x weight'
    },
    'key_insights': [
        'Bucket 1 (high erosion) contains only ~6.7% of brands but has 2x weight in metric',
        'Bucket 1 brands lose ~75%+ of volume within 24 months post generic entry',
        'Bucket 2 brands retain more volume, stabilizing above 25% normalized volume',
        'More generic competitors correlates with lower normalized volume',
        'Competition builds up gradually over 24 months post generic entry',
        'Erosion speed varies significantly between buckets'
    ],
    'modeling_recommendations': [
        'Use stratified sampling to handle bucket imbalance',
        'Consider separate models for each bucket',
        'Focus optimization on Bucket 1 predictions (2x weight)',
        'Include n_gxs as important feature for predictions',
        'Model time dynamics (months_postgx) carefully'
    ],
    'files_generated': {
        'json_data_files': [
            'fig01_bucket_distribution.json',
            'fig02_erosion_curves.json', 
            'fig03_sample_trajectories.json',
            'fig04_competition_impact.json',
            'fig05_therapeutic_areas.json',
            'fig06_biological_vs_small.json',
            'fig07_hospital_rate.json',
            'fig08_erosion_speed.json',
            'eda_complete_summary.json'
        ],
        'csv_data_files': [
            'fig01_bucket_distribution.csv',
            'fig02_erosion_curves.csv',
            'fig03_sample_trajectories.csv',
            'fig04_n_gxs_impact.csv',
            'fig04_competition_trajectory.csv',
            'fig05_therapeutic_areas.csv',
            'fig06_biological_vs_small.csv',
            'fig07_hospital_rate.csv',
            'fig08_erosion_speed_full.csv'
        ],
        'figure_files': [
            'bucket_distribution.png',
            'erosion_curves_by_bucket.png',
            'sample_brand_trajectories.png',
            'n_gxs_impact.png',
            'ther_area_erosion.png',
            'biological_vs_small_molecule.png',
            'hospital_rate_erosion.png',
            'erosion_speed_analysis.png'
        ]
    }
}

save_eda_json(summary_json_data, 'eda_complete_summary.json',
              'Complete EDA summary with all key insights')

# Print summary
print(f"\nDataset Shape: {eda_results['data_summary']['shape']}")
print(f"Total Brands: {eda_results['data_summary']['n_brands']}")
print(f"Countries: {eda_results['data_summary']['n_countries']}")
print(f"Months Range: {eda_results['data_summary']['months_postgx_range']}")

print("\n" + "-" * 40)
print("\nBucket Distribution:")
for _, row in bucket_data.iterrows():
    print(f"  Bucket {int(row['bucket'])}: {int(row['count'])} brands ({row['percentage']:.1f}%)")

print("\n⚠️ KEY INSIGHT: Bucket 1 (high erosion) is weighted 2× in the metric!")
print("   Focus optimization on Bucket 1 predictions!")

print("\n" + "="*60)
print(f"✅ All figures saved to: {FIGURES_DIR}")
print(f"✅ All JSON data saved to: {EDA_DATA_DIR}")
print("="*60)

# %%
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EDA Visualization Script Complete!")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"JSON data saved to: {EDA_DATA_DIR}")
    print("\nYou can ask me to interpret any figure by referencing its JSON file!")
