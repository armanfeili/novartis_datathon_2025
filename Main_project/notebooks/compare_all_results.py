"""
📊 Compare ALL Model Results Across All Runs

This script loads all timestamped model comparison files and creates
comprehensive visualizations comparing:
- All separate training runs (S1 and S2)
- All unified training runs
- Best results from each approach

Usage:
    python notebooks/compare_all_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Setup paths
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from config import REPORTS_DIR

# Output directory for this script's figures
OUTPUT_DIR = REPORTS_DIR / "compare_all_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("=" * 70)
print("📊 COMPARING ALL MODEL RESULTS")
print("=" * 70)

# =============================================================================
# Load ALL result files
# =============================================================================

def load_all_comparisons(pattern: str) -> dict:
    """Load all comparison files matching pattern."""
    results = {}
    files = sorted(REPORTS_DIR.glob(pattern))
    
    for f in files:
        # Extract timestamp from filename
        match = re.search(r'(\d{8}_\d{6})', f.name)
        if match:
            ts = match.group(1)
            # Format: YYYYMMDD_HHMMSS -> HH:MM:SS
            time_str = f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
            label = time_str
        else:
            label = "latest"
        
        try:
            df = pd.read_csv(f)
            results[label] = df
            print(f"  ✅ Loaded: {f.name} ({len(df)} models)")
        except Exception as e:
            print(f"  ⚠️ Error loading {f.name}: {e}")
    
    return results

print("\n📂 Loading Scenario 1 results...")
s1_results = load_all_comparisons("model_comparison_scenario1*.csv")

print("\n📂 Loading Scenario 2 results...")
s2_results = load_all_comparisons("model_comparison_scenario2*.csv")

print("\n📂 Loading Unified results...")
unified_results = load_all_comparisons("model_comparison_unified*.csv")

# =============================================================================
# FIGURE 1: All Separate Training Runs - Best Model per Run
# =============================================================================

print("\n" + "=" * 70)
print("📊 FIGURE 1: Best Model Score Per Run (Separate Training)")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# S1 best scores over time
if s1_results:
    runs = []
    for label, df in s1_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        runs.append({
            'run': label,
            'best_model': best_row['model'],
            'best_score': best_row['final_score']
        })
    
    runs_df = pd.DataFrame(runs)
    colors = ['#2ecc71' if s == runs_df['best_score'].min() else '#3498db' 
              for s in runs_df['best_score']]
    
    bars = axes[0].bar(range(len(runs_df)), runs_df['best_score'], color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(runs_df)))
    axes[0].set_xticklabels(runs_df['run'], rotation=45, ha='right')
    axes[0].set_ylabel('Best Score (PE, lower is better)')
    axes[0].set_title('Scenario 1: Best Score Per Run', fontsize=14)
    
    # Add model name labels
    for i, (bar, row) in enumerate(zip(bars, runs_df.itertuples())):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{row.best_model}\n{row.best_score:.4f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Mark overall best
    best_idx = runs_df['best_score'].idxmin()
    axes[0].annotate('🏆 BEST', xy=(best_idx, runs_df.loc[best_idx, 'best_score']),
                    xytext=(0, 30), textcoords='offset points', ha='center',
                    fontsize=12, color='green', fontweight='bold')

# S2 best scores over time
if s2_results:
    runs = []
    for label, df in s2_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        runs.append({
            'run': label,
            'best_model': best_row['model'],
            'best_score': best_row['final_score']
        })
    
    runs_df = pd.DataFrame(runs)
    colors = ['#2ecc71' if s == runs_df['best_score'].min() else '#e74c3c' 
              for s in runs_df['best_score']]
    
    bars = axes[1].bar(range(len(runs_df)), runs_df['best_score'], color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(runs_df)))
    axes[1].set_xticklabels(runs_df['run'], rotation=45, ha='right')
    axes[1].set_ylabel('Best Score (PE, lower is better)')
    axes[1].set_title('Scenario 2: Best Score Per Run', fontsize=14)
    
    # Add model name labels
    for i, (bar, row) in enumerate(zip(bars, runs_df.itertuples())):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{row.best_model}\n{row.best_score:.4f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Mark overall best
    best_idx = runs_df['best_score'].idxmin()
    axes[1].annotate('🏆 BEST', xy=(best_idx, runs_df.loc[best_idx, 'best_score']),
                    xytext=(0, 30), textcoords='offset points', ha='center',
                    fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'all_runs_best_scores.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: all_runs_best_scores.png")
plt.show()

# =============================================================================
# FIGURE 2: All Models Across All Runs (Heatmap)
# =============================================================================

print("\n" + "=" * 70)
print("📊 FIGURE 2: Model Performance Heatmap Across Runs")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# S1 Heatmap
if s1_results:
    # Build matrix: rows = models, columns = runs
    all_models = set()
    for df in s1_results.values():
        all_models.update(df['model'].tolist())
    all_models = sorted(all_models)
    
    matrix = pd.DataFrame(index=all_models, columns=list(s1_results.keys()))
    for label, df in s1_results.items():
        for _, row in df.iterrows():
            matrix.loc[row['model'], label] = row['final_score']
    
    matrix = matrix.astype(float)
    
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[0],
                cbar_kws={'label': 'Score (PE)'}, annot_kws={'size': 8})
    axes[0].set_title('Scenario 1: Model Scores Across All Runs', fontsize=12)
    axes[0].set_xlabel('Run Time')
    axes[0].set_ylabel('Model')

# S2 Heatmap  
if s2_results:
    all_models = set()
    for df in s2_results.values():
        all_models.update(df['model'].tolist())
    all_models = sorted(all_models)
    
    matrix = pd.DataFrame(index=all_models, columns=list(s2_results.keys()))
    for label, df in s2_results.items():
        for _, row in df.iterrows():
            matrix.loc[row['model'], label] = row['final_score']
    
    matrix = matrix.astype(float)
    
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[1],
                cbar_kws={'label': 'Score (PE)'}, annot_kws={'size': 8})
    axes[1].set_title('Scenario 2: Model Scores Across All Runs', fontsize=12)
    axes[1].set_xlabel('Run Time')
    axes[1].set_ylabel('Model')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'all_runs_heatmap.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: all_runs_heatmap.png")
plt.show()

# =============================================================================
# FIGURE 3: Separate vs Unified Comparison
# =============================================================================

print("\n" + "=" * 70)
print("📊 FIGURE 3: Separate vs Unified Training Comparison")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get best from each approach
comparison_data = []

# Best from separate S1
if s1_results:
    best_s1_score = float('inf')
    best_s1_model = None
    best_s1_run = None
    for label, df in s1_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_s1_score:
            best_s1_score = best_row['final_score']
            best_s1_model = best_row['model']
            best_s1_run = label
    comparison_data.append({
        'approach': f'Separate\n({best_s1_run})',
        'model': best_s1_model,
        's1_score': best_s1_score,
        's2_score': None
    })

# Best from separate S2
if s2_results:
    best_s2_score = float('inf')
    best_s2_model = None
    best_s2_run = None
    for label, df in s2_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_s2_score:
            best_s2_score = best_row['final_score']
            best_s2_model = best_row['model']
            best_s2_run = label
    # Update the separate entry with S2 score
    if comparison_data:
        comparison_data[0]['s2_score'] = best_s2_score

# Best from unified
if unified_results:
    best_unified_score = float('inf')
    best_unified_row = None
    best_unified_run = None
    for label, df in unified_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_unified_score:
            best_unified_score = best_row['final_score']
            best_unified_row = best_row
            best_unified_run = label
    
    if best_unified_row is not None:
        comparison_data.append({
            'approach': f'Unified\n({best_unified_run})',
            'model': best_unified_row['model'],
            's1_score': best_unified_row.get('s1_score', best_unified_score),
            's2_score': best_unified_row.get('s2_score', best_unified_score)
        })

# Plot comparison
if comparison_data:
    comp_df = pd.DataFrame(comparison_data)
    
    x = np.arange(len(comp_df))
    width = 0.35
    
    # S1 scores
    s1_scores = comp_df['s1_score'].fillna(0)
    s2_scores = comp_df['s2_score'].fillna(0)
    
    bars1 = axes[0].bar(x - width/2, s1_scores, width, label='Scenario 1', color='#3498db', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, s2_scores, width, label='Scenario 2', color='#e74c3c', edgecolor='black')
    
    axes[0].set_ylabel('Score (PE, lower is better)')
    axes[0].set_title('Best Scores: Separate vs Unified Training', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comp_df['approach'])
    axes[0].legend()
    
    # Add value labels
    for bar, val in zip(bars1, s1_scores):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, s2_scores):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Show model names
    axes[1].axis('off')
    table_data = []
    for _, row in comp_df.iterrows():
        table_data.append([
            row['approach'].replace('\n', ' '),
            row['model'],
            f"{row['s1_score']:.4f}" if pd.notna(row['s1_score']) else 'N/A',
            f"{row['s2_score']:.4f}" if pd.notna(row['s2_score']) else 'N/A'
        ])
    
    table = axes[1].table(
        cellText=table_data,
        colLabels=['Approach', 'Best Model', 'S1 Score', 'S2 Score'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    axes[1].set_title('Summary Table', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'separate_vs_unified.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: separate_vs_unified.png")
plt.show()

# =============================================================================
# FIGURE 4: Unified Results Over Time
# =============================================================================

if unified_results:
    print("\n" + "=" * 70)
    print("📊 FIGURE 4: Unified Training Results Over Time")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Collect all model scores across runs
    model_scores = {}
    for label, df in unified_results.items():
        for _, row in df.iterrows():
            model = row['model']
            if model not in model_scores:
                model_scores[model] = {'runs': [], 'scores': []}
            model_scores[model]['runs'].append(label)
            model_scores[model]['scores'].append(row['final_score'])
    
    # Plot each model as a line
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_scores)))
    for (model, data), color in zip(model_scores.items(), colors):
        ax.plot(data['runs'], data['scores'], 'o-', label=model, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Run Time', fontsize=12)
    ax.set_ylabel('Score (PE, lower is better)', fontsize=12)
    ax.set_title('Unified Training: Model Scores Across All Runs', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'unified_over_time.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: unified_over_time.png")
    plt.show()

# =============================================================================
# Summary Report
# =============================================================================

print("\n" + "=" * 70)
print("📋 SUMMARY REPORT")
print("=" * 70)

print("\n🏆 BEST RESULTS:")

if s1_results:
    best_s1_score = float('inf')
    best_s1_info = None
    for label, df in s1_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_s1_score:
            best_s1_score = best_row['final_score']
            best_s1_info = (label, best_row['model'], best_row['final_score'])
    print(f"\n  Scenario 1 (Separate):")
    print(f"    Run: {best_s1_info[0]}")
    print(f"    Model: {best_s1_info[1]}")
    print(f"    Score: {best_s1_info[2]:.4f}")

if s2_results:
    best_s2_score = float('inf')
    best_s2_info = None
    for label, df in s2_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_s2_score:
            best_s2_score = best_row['final_score']
            best_s2_info = (label, best_row['model'], best_row['final_score'])
    print(f"\n  Scenario 2 (Separate):")
    print(f"    Run: {best_s2_info[0]}")
    print(f"    Model: {best_s2_info[1]}")
    print(f"    Score: {best_s2_info[2]:.4f}")

if unified_results:
    best_uni_score = float('inf')
    best_uni_info = None
    for label, df in unified_results.items():
        best_row = df.loc[df['final_score'].idxmin()]
        if best_row['final_score'] < best_uni_score:
            best_uni_score = best_row['final_score']
            best_uni_info = (label, best_row['model'], best_row['final_score'],
                           best_row.get('s1_score'), best_row.get('s2_score'))
    print(f"\n  Unified Training:")
    print(f"    Run: {best_uni_info[0]}")
    print(f"    Model: {best_uni_info[1]}")
    print(f"    Combined Score: {best_uni_info[2]:.4f}")
    if best_uni_info[3]:
        print(f"    S1 Score: {best_uni_info[3]:.4f}")
    if best_uni_info[4]:
        print(f"    S2 Score: {best_uni_info[4]:.4f}")

print("\n" + "=" * 70)
print("✅ All visualizations saved to: reports/compare_all_results/")
print("=" * 70)
