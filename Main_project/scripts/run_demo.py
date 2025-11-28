# =============================================================================
# File: scripts/run_demo.py
# Description: Quick demo to verify entire pipeline works
# =============================================================================

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import load_all_data, merge_datasets
from bucket_calculator import compute_avg_j, create_auxiliary_file
from models import BaselineModels
from evaluation import evaluate_model
from submission import generate_submission, save_submission


def run_demo():
    """Run a quick demo of the entire pipeline."""
    
    print("=" * 70)
    print("🎯 NOVARTIS DATATHON 2025 - QUICK DEMO")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n📂 Step 1: Loading data...")
    
    volume_train, generics_train, medicine_train = load_all_data(train=True)
    merged_train = merge_datasets(volume_train, generics_train, medicine_train)
    
    # Use subset for demo
    brands = merged_train[['country', 'brand_name']].drop_duplicates().head(30)
    demo_data = merged_train.merge(brands, on=['country', 'brand_name'])
    print(f"   Using {len(brands)} brands for demo")
    
    # =========================================================================
    # Step 2: Create auxiliary file
    # =========================================================================
    print("\n📊 Step 2: Creating auxiliary file...")
    
    aux_df = create_auxiliary_file(demo_data, save=False)
    avg_j = aux_df[['country', 'brand_name', 'avg_vol']].copy()
    
    # =========================================================================
    # Step 3: Generate baseline predictions
    # =========================================================================
    print("\n📈 Step 3: Generating baseline predictions...")
    
    # Scenario 1: Predict months 0-23
    pred_s1 = BaselineModels.exponential_decay(
        avg_j,
        months_to_predict=list(range(0, 24)),
        decay_rate=0.05
    )
    print(f"   Scenario 1 predictions: {len(pred_s1)} rows")
    
    # Scenario 2: Predict months 6-23
    pred_s2 = BaselineModels.exponential_decay(
        avg_j,
        months_to_predict=list(range(6, 24)),
        decay_rate=0.05
    )
    print(f"   Scenario 2 predictions: {len(pred_s2)} rows")
    
    # =========================================================================
    # Step 4: Evaluate on training data (as demo)
    # =========================================================================
    print("\n📊 Step 4: Evaluating predictions...")
    
    # Get actual volumes
    actual_s1 = demo_data[demo_data['months_postgx'].between(0, 23)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ]
    
    actual_s2 = demo_data[demo_data['months_postgx'].between(6, 23)][
        ['country', 'brand_name', 'months_postgx', 'volume']
    ]
    
    # Evaluate Scenario 1
    print("\n--- Scenario 1 Evaluation ---")
    results_s1 = evaluate_model(actual_s1, pred_s1, aux_df, scenario=1)
    
    # Evaluate Scenario 2
    print("\n--- Scenario 2 Evaluation ---")
    results_s2 = evaluate_model(actual_s2, pred_s2, aux_df, scenario=2)
    
    # =========================================================================
    # Step 5: Generate sample submission
    # =========================================================================
    print("\n📝 Step 5: Generating sample submissions...")
    
    submission_s1 = generate_submission(pred_s1, scenario=1)
    submission_s2 = generate_submission(pred_s2, scenario=2)
    
    print(f"   Scenario 1 submission: {len(submission_s1)} rows")
    print(f"   Scenario 2 submission: {len(submission_s2)} rows")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results Summary:")
    print(f"   Scenario 1 Final Score: {results_s1['final_score']:.4f}")
    print(f"   Scenario 2 Final Score: {results_s2['final_score']:.4f}")
    print(f"\n🚀 Ready to run full pipeline!")
    print(f"   Command: python src/pipeline.py --scenario 1 --model lightgbm")
    
    return results_s1, results_s2


if __name__ == "__main__":
    run_demo()
