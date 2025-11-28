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

print("✅ Config imported!")
print(f"Models directory: {MODELS_DIR}")
print(f"Reports directory: {REPORTS_DIR}")

# Load comparison results
try:
    comparison_s1 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario1.csv')
    print("✅ Loaded Scenario 1 results")
    display(comparison_s1)
except FileNotFoundError:
    print("⚠️ Scenario 1 results not found. Run scripts/train_models.py --scenario 1 first.")
    comparison_s1 = None

try:
    comparison_s2 = pd.read_csv(REPORTS_DIR / 'model_comparison_scenario2.csv')
    print("✅ Loaded Scenario 2 results")
    display(comparison_s2)
except FileNotFoundError:
    print("⚠️ Scenario 2 results not found. Run scripts/train_models.py --scenario 2 first.")
    comparison_s2 = None

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

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Saved to {FIGURES_DIR / 'model_comparison.png'}")

# Bucket-level analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scenario 1 bucket breakdown
if comparison_s1 is not None and 'bucket1_pe' in comparison_s1.columns:
    x = np.arange(len(comparison_s1))
    width = 0.35
    
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

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bucket_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# Load LightGBM model and get feature importance
import joblib

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
        }).sort_values('importance', ascending=True).tail(15)
        
        axes[idx].barh(importance['feature'], importance['importance'], 
                       color='steelblue', edgecolor='black')
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'Scenario {scenario}: Top 15 Features (LightGBM)', fontsize=12)
    else:
        axes[idx].text(0.5, 0.5, f'Model not found\nRun train_models.py first',
                       ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'Scenario {scenario}: Feature Importance')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Load and analyze submissions
submission_files = list(SUBMISSIONS_DIR.glob('scenario*_final.csv'))

print(f"📁 Found {len(submission_files)} submission files:")
for f in submission_files:
    print(f"   - {f.name}")

submissions = {}
for f in submission_files:
    df = pd.read_csv(f)
    scenario = 1 if 'scenario1' in f.name else 2
    submissions[scenario] = df
    print(f"\n📊 {f.name}:")
    print(f"   Rows: {len(df):,}")
    print(f"   Brands: {df[['country', 'brand_name']].drop_duplicates().shape[0]}")
    print(f"   Volume range: [{df['volume'].min():.2f}, {df['volume'].max():.2f}]")

# Visualize submission predictions
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
plt.savefig(FIGURES_DIR / 'submission_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (scenario, df) in enumerate(submissions.items()):
    # Log scale for better visualization
    log_volume = np.log1p(df['volume'])
    
    axes[idx].hist(log_volume, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[idx].axvline(x=log_volume.median(), color='red', linestyle='--', 
                      label=f'Median: {np.expm1(log_volume.median()):,.0f}')
    axes[idx].set_xlabel('Log(Volume + 1)')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Scenario {scenario}: Predicted Volume Distribution', fontsize=12)
    axes[idx].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Final summary
print("="*70)
print("📊 MODEL RESULTS SUMMARY")
print("="*70)

if comparison_s1 is not None:
    best_s1 = comparison_s1.loc[comparison_s1['final_score'].idxmin()]
    print(f"\n🏆 SCENARIO 1 BEST MODEL:")
    print(f"   Model: {best_s1['model']}")
    print(f"   Final Score: {best_s1['final_score']:.4f}")
    if 'bucket1_pe' in best_s1:
        print(f"   Bucket 1 PE: {best_s1['bucket1_pe']:.4f}")
        print(f"   Bucket 2 PE: {best_s1['bucket2_pe']:.4f}")

if comparison_s2 is not None:
    best_s2 = comparison_s2.loc[comparison_s2['final_score'].idxmin()]
    print(f"\n🏆 SCENARIO 2 BEST MODEL:")
    print(f"   Model: {best_s2['model']}")
    print(f"   Final Score: {best_s2['final_score']:.4f}")
    if 'bucket1_pe' in best_s2:
        print(f"   Bucket 1 PE: {best_s2['bucket1_pe']:.4f}")
        print(f"   Bucket 2 PE: {best_s2['bucket2_pe']:.4f}")

print(f"\n📁 SUBMISSIONS:")
for scenario, df in submissions.items():
    print(f"   Scenario {scenario}: {len(df):,} predictions")

print(f"\n✅ All figures saved to: {FIGURES_DIR}")
print("="*70)
print("\n📤 Ready for competition submission!")
