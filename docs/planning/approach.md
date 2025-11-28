# Novartis Datathon 2025: Implementation Approach

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Understanding](#2-project-understanding)
3. [Time Allocation Strategy](#3-time-allocation-strategy)
4. [Data Pipeline Architecture](#4-data-pipeline-architecture)
5. [Exploratory Data Analysis Plan](#5-exploratory-data-analysis-plan)
6. [Feature Engineering Strategy](#6-feature-engineering-strategy)
7. [Modeling Approach](#7-modeling-approach)
8. [Validation Framework](#8-validation-framework)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Submission Strategy](#10-submission-strategy)
11. [Business Story & Presentation](#11-business-story--presentation)
12. [Risk Mitigation](#12-risk-mitigation)
13. [Code Organization](#13-code-organization)
14. [Deliverables Checklist](#14-deliverables-checklist)

---

## 1. Executive Summary

### 1.1 Business Objective

**One-sentence goal**: Forecast monthly sales volumes for branded drugs after generic entry over a 24-month horizon, with special focus on high-erosion brands (Bucket 1), to help Novartis anticipate revenue loss, plan post-patent strategies, and optimize product and country-level decisions.

### 1.2 Technical Framing

This is a **panel time-series forecasting problem** with:
- **1,953 training series** (country-brand combinations)
- **340 test series** (228 Scenario 1, 112 Scenario 2)
- **Two forecasting scenarios** with different information availability
- **Weighted evaluation** prioritizing Bucket 1 (high erosion) and early months

### 1.3 Success Criteria

| Criterion | Target |
|-----------|--------|
| Phase 1-a (Scenario 1) | Top 10 ranking → Advance |
| Phase 1-b (Scenario 2) | Top 5 ranking → Advance to finals |
| Phase 2 (Jury) | Clear methodology + business insights → Top 3 |

### 1.4 Key Constraints

- **Submission limits**: 3 submissions per 8 hours
- **Time constraint**: ~48-hour competition window
- **Public/Private split**: 30% public leaderboard / 70% private final
- **Format**: CSV with `country, brand_name, months_postgx, volume`

### 1.5 Three-Phase Work Structure

We structure the work into three phases, mirroring how winning teams operate:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 – UNDERSTANDING & DESIGN                                    │
│  ├─ EDA, metric understanding, validation strategy                   │
│  ├─ Data pipeline design                                             │
│  └─ Sections: 2, 4, 5, 8 (Understanding, Pipeline, EDA, Validation)  │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2 – CORE IMPLEMENTATION ("Guaranteed Good" Solution)          │
│  ├─ Baseline + hero model + end-to-end pipeline                      │
│  ├─ First draft of business story + key plots                        │
│  └─ Sections: 6, 7, 10, 13, 14 (Features, Modeling, Submission, etc.)│
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3 – IMPROVEMENT, EDGE & POLISH                                │
│  ├─ Feature/model refinements, robustness checks                     │
│  ├─ Final presentation and storytelling                              │
│  └─ Sections: 9, 11, 12 (Tracking, Story, Risk Mitigation)           │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: This structure ensures you always have a working solution before attempting risky improvements. Phase 2 output is your "safety net" – a robust baseline that would be competitive even without Phase 3 refinements.

---

## 2. Project Understanding

### 2.1 The Two Scenarios

```
┌─────────────────────────────────────────────────────────────────┐
│                         SCENARIO 1                               │
│                 (Right After Generic Entry)                      │
├─────────────────────────────────────────────────────────────────┤
│  Position: Month 0 (generic entry)                               │
│  Available: Pre-entry history + static drug info                 │
│  Forecast:  Months 0-23 (24 months)                              │
│  Challenge: Pure ex-ante prediction, no post-entry signal        │
│  Weight:    Months 0-5 = 50%, 6-11 = 20%, 12-23 = 10%           │
│             Monthly = 20%                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         SCENARIO 2                               │
│                 (Six Months After Entry)                         │
├─────────────────────────────────────────────────────────────────┤
│  Position: Month 6                                               │
│  Available: Pre-entry + months 0-5 actuals                       │
│  Forecast:  Months 6-23 (18 months)                              │
│  Challenge: Leverage early erosion signal                        │
│  Weight:    Months 6-11 = 50%, 12-23 = 30%                       │
│             Monthly = 20%                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Bucket Classification

| Bucket | Mean Erosion Range | Description | Metric Weight |
|--------|-------------------|-------------|---------------|
| **B1** | [0, 0.25] | High erosion (severe drop) | **2×** |
| **B2** | (0.25, 1] | Medium/Low erosion | 1× |

### 2.3 Data Assets

| File | Purpose | Key Columns |
|------|---------|-------------|
| `df_volume_train.csv` | Target time series | `country`, `brand_name`, `month`, `months_postgx`, `volume` |
| `df_generics_train.csv` | Competition dynamics | `country`, `brand_name`, `months_postgx`, `n_gxs` |
| `df_medicine_info_train.csv` | Static drug attributes | `ther_area`, `hospital_rate`, `main_package`, `biological`, `small_molecule` |
| `metric_calculation.py` | Official metric code | `compute_metric1()`, `compute_metric2()` |
| `submission_template.csv` | Submission format | All required (country, brand, month) combinations |

---

## 3. Time Allocation Strategy

### 3.1 Competition Timeline

```
┌────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Understanding + EDA              [Day 1 + Early Day 2]   │
│  ├─ Data loading & validation                        (~2 hours)    │
│  ├─ Initial EDA & visualization                      (~4 hours)    │
│  ├─ Metric understanding & baseline                  (~2 hours)    │
│  └─ Feature brainstorming                            (~2 hours)    │
├────────────────────────────────────────────────────────────────────┤
│  PHASE 2: Modeling + Validation            [Day 2 + Early Day 3]   │
│  ├─ Feature pipeline implementation                  (~4 hours)    │
│  ├─ Baseline model training                          (~2 hours)    │
│  ├─ Hero model (GBM) training & tuning               (~6 hours)    │
│  ├─ Scenario 1 & 2 specific models                   (~4 hours)    │
│  └─ Ensemble exploration                             (~2 hours)    │
├────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Analysis + Storytelling                     [Late Day 3] │
│  ├─ Error analysis by bucket/segment                 (~2 hours)    │
│  ├─ Feature importance & SHAP                        (~2 hours)    │
│  ├─ Business narrative development                   (~2 hours)    │
│  └─ Visualization polish                             (~2 hours)    │
├────────────────────────────────────────────────────────────────────┤
│  PHASE 4: Final Submission + Slides               [Sunday Morning] │
│  ├─ Final model training on full data                (~1 hour)     │
│  ├─ Generate final predictions                       (~1 hour)     │
│  ├─ Slide deck completion                            (~3 hours)    │
│  └─ Code cleanup & documentation                     (~2 hours)    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Critical Deadlines

| Time | Event | Action Required |
|------|-------|-----------------|
| Competition Start | Begin | Load data, verify format |
| First 2 hours | Early | Submit test prediction to verify format |
| Saturday Evening | Checkpoint | Have working model for both scenarios |
| Sunday 9:30 AM | Selection Opens | Review all submissions |
| Sunday 10:30 AM | **DEADLINE** | Mark final submission |
| Sunday 12:00 PM | Finalist | Upload slides + code |

### 3.3 Phase Milestones & Story Draft

**End of Phase 1 (Day 1 Evening)**:
- [ ] Data loaded and validated
- [ ] EDA complete with key insights documented
- [ ] Validation strategy defined and tested
- [ ] Feature list brainstormed

**End of Phase 2 (Day 2 Evening)** – **CRITICAL CHECKPOINT**:
- [ ] Hero model trained for both scenarios
- [ ] End-to-end submission pipeline working
- [ ] **First draft of business story ready**:
  - 1–2 example country–brand series (actual vs predicted)
  - First version of B1 vs B2 erosion comparison plots
  - Short narrative explaining what the model learned
- [ ] At least one valid submission uploaded

**End of Phase 3 (Sunday Morning)**:
- [ ] Robustness checks complete (adversarial validation, alternative splits)
- [ ] Final model selected and submission marked
- [ ] Presentation slides polished
- [ ] Code package ready for upload

---

## 4. Data Pipeline Architecture

### 4.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ df_volume    │    │ df_generics  │    │df_medicine   │           │
│  │   _train     │    │   _train     │    │  _info_train │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                    │                    │                  │
│         └────────────┬───────┴────────────┬──────┘                  │
│                      ▼                    ▼                         │
│         ┌────────────────────────────────────────────┐              │
│         │           prepare_base_panel()              │              │
│         │  Join on (country, brand_name, months_postgx)│             │
│         └────────────────────┬───────────────────────┘              │
│                              ▼                                       │
│         ┌────────────────────────────────────────────┐              │
│         │      compute_pre_entry_stats()              │              │
│         │  - avg_vol (12-month pre-entry average)     │              │
│         │  - bucket (compute mean erosion → classify) │              │
│         └────────────────────┬───────────────────────┘              │
│                              ▼                                       │
│         ┌────────────────────────────────────────────┐              │
│         │    make_features(panel, scenario)           │              │
│         │  - Scenario 1: only pre-entry data          │              │
│         │  - Scenario 2: pre-entry + months 0-5       │              │
│         └────────────────────┬───────────────────────┘              │
│                              ▼                                       │
│         ┌────────────────────────────────────────────┐              │
│         │           MODEL TRAINING                    │              │
│         └────────────────────────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Loading Module

```python
# src/data.py - Key functions to implement

def load_raw_data(data_dir: str, split: str = "train") -> dict:
    """Load all three datasets for train or test split."""
    return {
        'volume': pd.read_csv(f"{data_dir}/{split.upper()}/df_volume_{split}.csv"),
        'generics': pd.read_csv(f"{data_dir}/{split.upper()}/df_generics_{split}.csv"),
        'medicine_info': pd.read_csv(f"{data_dir}/{split.upper()}/df_medicine_info_{split}.csv")
    }

def prepare_base_panel(volume_df, generics_df, medicine_info_df) -> pd.DataFrame:
    """Create unified panel with all features joined."""
    # Join volume with generics on (country, brand_name, months_postgx)
    # Join with medicine_info on (country, brand_name)
    pass

def compute_pre_entry_stats(panel_df) -> pd.DataFrame:
    """Compute avg_vol and bucket for each (country, brand_name)."""
    # 1. Filter to months_postgx in [-12, -1]
    # 2. Compute avg_vol = mean(volume)
    # 3. Filter to months_postgx in [0, 23]
    # 4. Compute normalized volume = volume / avg_vol
    # 5. Compute mean_erosion = mean(normalized_volume)
    # 6. Assign bucket: 1 if mean_erosion <= 0.25 else 2
    pass
```

### 4.3 Missing Value Strategy

| Column | Strategy | Justification |
|--------|----------|---------------|
| `volume` | Keep as-is (very rare NaN) | Target variable, must not impute |
| `n_gxs` | Forward-fill, then 0 | Generics count is cumulative |
| `hospital_rate` | Median by `ther_area` + flag | Domain-aware imputation |
| `ther_area` | "Unknown" category | Preserve information |
| `main_package` | "Unknown" category | Preserve information |
| `biological` / `small_molecule` | False if missing + flag | Conservative default |

---

## 5. Exploratory Data Analysis Plan

### 5.1 EDA Checklist

#### A. Data Quality Checks
- [ ] Row counts per dataset (train vs test)
- [ ] Missing values per column
- [ ] Duplicates check on `(country, brand_name, months_postgx)`
- [ ] Value ranges (negative volumes? impossible n_gxs?)
- [ ] Distribution of `months_postgx` (do all series have full 24+24?)

#### B. Target Distribution
- [ ] Histogram of `volume` (likely right-skewed)
- [ ] Histogram of `avg_vol` (pre-entry baseline)
- [ ] Histogram of mean normalized erosion
- [ ] Bucket distribution (B1 vs B2 count)

#### C. Erosion Pattern Analysis
- [ ] **Critical**: Plot normalized volume vs `months_postgx` for B1 vs B2
- [ ] Speed of erosion (months 0-5 vs 6-11 vs 12-23)
- [ ] Impact of `n_gxs` on erosion trajectory
- [ ] Erosion by `ther_area`, `biological`, `hospital_rate`

#### D. Feature Distributions
- [ ] `n_gxs` distribution over time
- [ ] `hospital_rate` histogram
- [ ] `ther_area` bar chart
- [ ] `main_package` distribution
- [ ] `biological` vs `small_molecule` counts

### 5.2 Key Visualizations to Create

```
1. EROSION CURVE COMPARISON
   ┌────────────────────────────────────────┐
   │  Normalized Volume                      │
   │  1.0 ─── B2 (low erosion)              │
   │      ╲                                  │
   │  0.5  ╲_______________                 │
   │        ╲                                │
   │  0.25 ──╲─── Bucket threshold          │
   │          ╲____ B1 (high erosion)       │
   │  0.0                                    │
   │      0    6    12    18    24          │
   │         months_postgx                   │
   └────────────────────────────────────────┘

2. N_GXS IMPACT ON EROSION
   - Dual-axis plot: volume trajectory + n_gxs over time
   - Faceted by bucket or therapeutic area

3. SEGMENT COMPARISON HEATMAP
   - Rows: ther_area categories
   - Columns: time periods (0-5, 6-11, 12-23)
   - Values: mean normalized erosion
```

### 5.3 EDA Questions to Answer

1. **How distinct are B1 and B2 erosion patterns?**
   - If very distinct → consider separate models
   - If overlapping → unified model with bucket-aware features

2. **Which features most differentiate erosion severity?**
   - Candidates: `n_gxs`, `biological`, `hospital_rate`, `ther_area`
   - This informs feature engineering priority

3. **Are there country-level systematic differences?**
   - If yes → use country as feature or stratify
   - If no → country mainly for grouping

4. **What's the typical erosion speed?**
   - Fast (most drop in months 0-5) → prioritize early month accuracy
   - Gradual → smooth trend models may work

---

## 6. Feature Engineering Strategy

### 6.1 Feature Categories

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CATEGORY 1: PRE-ENTRY STATISTICS (Available in Both Scenarios)     │
│  ├─ avg_vol_12m: Mean volume over months [-12, -1]                  │
│  ├─ avg_vol_6m: Mean volume over months [-6, -1]                    │
│  ├─ avg_vol_3m: Mean volume over months [-3, -1]                    │
│  ├─ pre_entry_trend: Linear slope over pre-entry period             │
│  ├─ pre_entry_volatility: Std dev of volume / avg_vol               │
│  ├─ pre_entry_max: Maximum volume in pre-entry period               │
│  └─ log_avg_vol: log1p(avg_vol_12m) for scale normalization         │
│                                                                      │
│  CATEGORY 2: TIME FEATURES                                           │
│  ├─ months_postgx: Direct time index (-24 to 23)                    │
│  ├─ months_postgx_sq: Squared for non-linear decay                  │
│  ├─ is_post_entry: Binary (months_postgx >= 0)                      │
│  ├─ time_bucket: Categorical (pre, 0-5, 6-11, 12-23)                │
│  └─ month_of_year: Seasonality from calendar month                   │
│                                                                      │
│  CATEGORY 3: GENERICS COMPETITION                                    │
│  ├─ n_gxs: Current number of generics                               │
│  ├─ has_generic: Binary (n_gxs > 0)                                 │
│  ├─ multiple_generics: Binary (n_gxs >= 2)                          │
│  ├─ n_gxs_cummax: Maximum n_gxs up to current month                 │
│  ├─ n_gxs_change: Change in n_gxs from previous month               │
│  └─ first_generic_month: Month when n_gxs first > 0                 │
│                                                                      │
│  CATEGORY 4: DRUG CHARACTERISTICS                                    │
│  ├─ ther_area: Therapeutic area (encoded)                           │
│  ├─ biological: Boolean                                              │
│  ├─ small_molecule: Boolean                                          │
│  ├─ hospital_rate: Percentage (0-100)                                │
│  ├─ hospital_rate_bin: Low/Medium/High                               │
│  ├─ main_package: Dosage form (encoded)                              │
│  └─ is_injection: Binary from main_package                           │
│                                                                      │
│  CATEGORY 5: SCENARIO 2 ONLY (Early Post-Entry Signal)              │
│  ├─ volume_month_0_to_5: Actual volumes for months 0-5              │
│  ├─ avg_vol_0_5: Mean volume over months [0, 5]                     │
│  ├─ erosion_0_5: avg_vol_0_5 / avg_vol_12m                          │
│  ├─ trend_0_5: Linear slope over months 0-5                         │
│  └─ drop_month_0: volume[0] / avg_vol_12m (initial drop)            │
│                                                                      │
│  CATEGORY 6: INTERACTION FEATURES                                    │
│  ├─ n_gxs_x_biological: Interaction term                            │
│  ├─ hospital_rate_x_time: Time-varying hospital effect               │
│  └─ ther_area_x_n_gxs: Therapeutic-specific competition effect      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Feature Engineering Code Structure

```python
# src/features.py

def make_features(panel_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Build features respecting scenario-specific information constraints.
    
    Args:
        panel_df: Base panel with all raw data
        scenario: "scenario1" or "scenario2"
    
    Returns:
        DataFrame with engineered features
    """
    features = panel_df.copy()
    
    # Determine cutoff based on scenario
    if scenario == "scenario1":
        cutoff_month = 0  # Can only use months_postgx < 0
    else:  # scenario2
        cutoff_month = 6  # Can use months_postgx < 6
    
    # 1. Pre-entry statistics (both scenarios)
    features = add_pre_entry_features(features)
    
    # 2. Time features
    features = add_time_features(features)
    
    # 3. Generics features (respecting cutoff)
    features = add_generics_features(features, cutoff_month)
    
    # 4. Drug characteristics (static)
    features = add_drug_features(features)
    
    # 5. Scenario 2 only: early post-entry signal
    if scenario == "scenario2":
        features = add_early_erosion_features(features)
    
    # 6. Interactions
    features = add_interaction_features(features)
    
    return features
```

### 6.3 Leakage Prevention Rules

| Rule | Implementation |
|------|----------------|
| **Never use future months** | Filter `months_postgx < cutoff` before computing features |
| **Never use bucket as feature** | Bucket is computed from target; use only for analysis |
| **Separate train/test pipelines** | Same code, but test has no access to train statistics |
| **No global statistics on test** | Statistics (mean, std) computed only on training data |

### 6.4 Target Definition & Metric Alignment

The official metric uses **normalized volume** (volume / avg_vol) with time-window and bucket weights. Our training should align with this.

#### Target Choice

**Decision: Train on normalized volume (Option 1)**

$$y_{j,i} = \frac{\text{volume}_{j,i}}{\text{Avg}_j}$$

| Option | Pros | Cons |
|--------|------|------|
| **Normalized volume** ✓ | Directly aligns with metric; scale-invariant | Need to handle small Avg_j |
| Log volume | Handles skewness | Doesn't match metric normalization |
| Raw volume | Simple | Large brands dominate; metric mismatch |

**Handling small Avg_j**:
- Clip `Avg_j` to minimum of 100 (or 1st percentile) to avoid division instability
- Flag series with very small `Avg_j` for conservative prediction fallback

#### Sample Weights to Reflect Metric

Instead of treating all rows equally, weight samples to approximate the official PE:

```python
def compute_sample_weights(df: pd.DataFrame, scenario: str) -> pd.Series:
    """
    Compute sample weights that approximate official metric weighting.
    """
    weights = pd.Series(1.0, index=df.index)
    
    # Time-window weights
    if scenario == "scenario1":
        # Months 0-5: 50% weight, 6-11: 20%, 12-23: 10%, monthly: 20%
        weights.loc[df['months_postgx'].between(0, 5)] = 3.0   # Higher priority
        weights.loc[df['months_postgx'].between(6, 11)] = 1.5
        weights.loc[df['months_postgx'].between(12, 23)] = 1.0
    else:  # scenario2
        # Months 6-11: 50%, 12-23: 30%, monthly: 20%
        weights.loc[df['months_postgx'].between(6, 11)] = 2.5  # Higher priority
        weights.loc[df['months_postgx'].between(12, 23)] = 1.0
    
    # Bucket weights (B1 = 2×)
    weights.loc[df['bucket'] == 1] *= 2.0
    
    return weights
```

**Key insight**: This makes the training loss approximate the official PE, rather than optimizing generic MAE/MSE equally across all rows.

---

## 7. Modeling Approach

### 7.1 Model Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TIER 1: BASELINES (Implement First)                                 │
│  ├─ Naive Flat: Predict avg_vol_12m for all future months           │
│  ├─ Global Erosion Curve: Apply average erosion by months_postgx    │
│  └─ Linear Trend: Extrapolate pre-entry trend                       │
│                                                                      │
│  TIER 2: HERO MODEL (Primary Focus)                                  │
│  ├─ Gradient Boosting (CatBoost / LightGBM / XGBoost)               │
│  ├─ Train separate models for Scenario 1 and Scenario 2             │
│  └─ Features: Full engineered feature set                            │
│                                                                      │
│  TIER 3: ENSEMBLE (If Time Permits)                                  │
│  ├─ Average of 2-3 best GBM variants                                │
│  └─ Weighted average prioritizing Bucket 1 performance              │
│                                                                      │
│  TIER 4: ALTERNATIVES (Optional Exploration)                         │
│  ├─ Ridge/Lasso on normalized volume                                 │
│  ├─ Per-series ARIMA (for comparison)                                │
│  └─ Neural network (if data supports)                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Baseline Implementations

```python
# src/models/baselines.py

def baseline_flat(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 1: Predict pre-entry average for all future months.
    
    For Scenario 1: Predict avg_vol_12m for months 0-23
    For Scenario 2: Predict avg_vol_0_5 for months 6-23
    """
    # Compute avg_vol per (country, brand_name)
    pre_entry = train_df[train_df['months_postgx'].between(-12, -1)]
    avg_vol = pre_entry.groupby(['country', 'brand_name'])['volume'].mean().reset_index()
    avg_vol.columns = ['country', 'brand_name', 'pred_volume']
    
    # Merge with test
    predictions = test_df.merge(avg_vol, on=['country', 'brand_name'])
    predictions['volume'] = predictions['pred_volume']
    
    return predictions[['country', 'brand_name', 'months_postgx', 'volume']]

def baseline_global_erosion(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 2: Apply average erosion curve to each brand.
    
    E[m] = mean(volume_m / avg_vol) across all training brands
    Predict: volume_m = avg_vol_brand × E[m]
    """
    # Compute global erosion curve
    # For each brand, compute normalized volume
    # Average across brands for each months_postgx
    pass

def baseline_linear_trend(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 3: Fit linear trend on pre-entry data, extrapolate.
    
    For Scenario 2: Fit on months 0-5, extrapolate to 6-23
    """
    pass
```

### 7.3 Hero Model Configuration

```python
# configs/model_config.yaml equivalent

HERO_MODEL_CONFIG = {
    'catboost': {
        'scenario1': {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'cat_features': ['ther_area', 'main_package', 'country'],
            'loss_function': 'MAE',  # or custom loss
        },
        'scenario2': {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'cat_features': ['ther_area', 'main_package', 'country'],
            'loss_function': 'MAE',
        }
    },
    'lightgbm': {
        # Similar configuration
    }
}
```

### 7.4 Model Training Strategy

```python
# src/train.py

def train_scenario_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    val_features: pd.DataFrame,
    val_target: pd.Series,
    scenario: str,
    model_type: str = 'catboost'
) -> Tuple[Model, dict]:
    """
    Train model for specific scenario with early stopping.
    
    Returns:
        Trained model and training metrics
    """
    config = HERO_MODEL_CONFIG[model_type][scenario]
    
    if model_type == 'catboost':
        model = CatBoostRegressor(**config)
        model.fit(
            train_features, train_target,
            eval_set=(val_features, val_target),
            verbose=100
        )
    
    # Compute validation metrics
    val_pred = model.predict(val_features)
    metrics = compute_validation_metrics(val_target, val_pred, scenario)
    
    return model, metrics
```

### 7.5 Hyperparameter Tuning Strategy

Given time constraints, use **targeted tuning**:

| Parameter | Search Range | Priority |
|-----------|--------------|----------|
| `learning_rate` | [0.01, 0.05, 0.1] | High |
| `depth` | [4, 5, 6, 7] | High |
| `l2_leaf_reg` | [1, 3, 5, 10] | Medium |
| `iterations` | Use early stopping | Auto |
| `subsample` | [0.7, 0.8, 0.9] | Medium |
| `colsample_bylevel` | [0.7, 0.8, 0.9] | Low |

**Strategy**: 
1. Start with defaults
2. Run ~20-30 random combinations
3. Pick best based on validation metric
4. Verify stability across 2-3 random seeds

---

## 8. Validation Framework

### 8.1 Validation Design Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VALIDATION PRINCIPLES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SERIES-LEVEL SPLITS (Not Row-Level)                              │
│     ✓ Hold out entire (country, brand) combinations                 │
│     ✗ Never mix months of same series across train/val              │
│                                                                      │
│  2. SCENARIO-AWARE FEATURE CONSTRUCTION                              │
│     ✓ For validation series, use only scenario-allowed data         │
│     ✓ Scenario 1: Only pre-entry (months_postgx < 0)                │
│     ✓ Scenario 2: Pre-entry + months 0-5                             │
│                                                                      │
│  3. STRATIFIED BY BUCKET                                             │
│     ✓ Maintain similar B1/B2 ratio in train vs validation           │
│     ✓ Ensures validation reflects test distribution                  │
│                                                                      │
│  4. USE OFFICIAL METRIC                                              │
│     ✓ Evaluate with compute_metric1() and compute_metric2()         │
│     ✓ Track bucket-specific performance                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Validation Split Implementation

```python
# src/validation.py

def create_validation_split(
    panel_df: pd.DataFrame,
    val_fraction: float = 0.2,
    stratify_by: str = 'bucket',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split at series level.
    
    Args:
        panel_df: Full training panel
        val_fraction: Fraction of series for validation
        stratify_by: Column to stratify by (usually 'bucket')
        random_state: For reproducibility
    
    Returns:
        train_df, val_df (both full series, not just rows)
    """
    # Get unique series with their bucket
    series_info = panel_df.groupby(['country', 'brand_name']).agg({
        'bucket': 'first',  # Assuming bucket is pre-computed
        'ther_area': 'first'
    }).reset_index()
    
    # Stratified split
    train_series, val_series = train_test_split(
        series_info,
        test_size=val_fraction,
        stratify=series_info[stratify_by],
        random_state=random_state
    )
    
    # Filter panel to get full train/val data
    train_keys = set(zip(train_series['country'], train_series['brand_name']))
    val_keys = set(zip(val_series['country'], val_series['brand_name']))
    
    train_df = panel_df[panel_df.apply(
        lambda x: (x['country'], x['brand_name']) in train_keys, axis=1
    )]
    val_df = panel_df[panel_df.apply(
        lambda x: (x['country'], x['brand_name']) in val_keys, axis=1
    )]
    
    return train_df, val_df
```

### 8.3 Validation Metrics Dashboard

Track these metrics for every model experiment:

| Metric | Scenario 1 | Scenario 2 |
|--------|------------|------------|
| **Official PE** | `compute_metric1()` | `compute_metric2()` |
| **PE Bucket 1** | Subset to B1 | Subset to B1 |
| **PE Bucket 2** | Subset to B2 | Subset to B2 |
| **MAE Overall** | Unnormalized | Unnormalized |
| **MAE Normalized** | MAE / avg_vol | MAE / avg_vol |

### 8.4 Cross-Validation Strategy

Given time constraints, use **1-2 splits** rather than full K-fold:

```python
def quick_cv(panel_df, model_fn, n_splits=2):
    """
    Quick 2-fold CV for robustness check.
    """
    scores = []
    
    for fold in range(n_splits):
        train_df, val_df = create_validation_split(
            panel_df, 
            val_fraction=0.2,
            random_state=42 + fold
        )
        
        # Train and evaluate
        model = model_fn(train_df)
        score = evaluate_model(model, val_df)
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }
```

### 8.5 Adversarial Validation (Optional but Recommended)

Adversarial validation detects **distribution shift** between train and test, which helps avoid overfitting to quirks of the training data.

**Implementation**:

```python
def adversarial_validation(train_features: pd.DataFrame, 
                           test_features: pd.DataFrame) -> dict:
    """
    Train classifier to distinguish train vs test rows.
    High AUC indicates distribution shift.
    """
    # Combine train and test (features only, no target)
    train_features['is_test'] = 0
    test_features['is_test'] = 1
    combined = pd.concat([train_features, test_features])
    
    # Train simple classifier
    X = combined.drop(columns=['is_test'])
    y = combined['is_test']
    
    model = LGBMClassifier(n_estimators=100, max_depth=4)
    model.fit(X, y)
    
    # Compute AUC
    from sklearn.model_selection import cross_val_score
    auc_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'mean_auc': np.mean(auc_scores),
        'auc_scores': auc_scores,
        'top_shift_features': importances.head(10)
    }
```

**Interpretation**:
- **AUC ≈ 0.5**: Train and test are similar → Good!
- **AUC > 0.7**: Significant distribution shift → Caution!
- **AUC > 0.8**: Severe shift → Consider:
  - Dropping high-importance features that cause shift
  - Increasing regularization
  - Mentioning this in presentation as a "robustness consideration"

**What to do with results**:
1. If shift detected, inspect the top features driving the shift
2. Consider whether those features are "safe" (real signal) or "dangerous" (artifacts)
3. Optionally train models with/without suspect features and compare validation scores
4. **Mention adversarial validation in presentation** – shows methodological rigor

---

## 9. Experiment Tracking

### 9.1 Experiment Log Template

Create a simple CSV or spreadsheet:

| timestamp | scenario | model | features | depth | lr | metric1 | metric2 | pe_b1 | pe_b2 | notes |
|-----------|----------|-------|----------|-------|-----|---------|---------|-------|-------|-------|
| 2025-11-28 10:00 | both | catboost_v1 | fe_v1 | 6 | 0.05 | 1.23 | 0.98 | 1.45 | 1.01 | baseline |
| 2025-11-28 12:00 | both | catboost_v2 | fe_v2 | 5 | 0.03 | 1.15 | 0.92 | 1.32 | 0.98 | added n_gxs features |

### 9.2 Model Versioning

```
models/
├── baselines/
│   ├── flat_baseline.pkl
│   ├── global_erosion.pkl
│   └── linear_trend.pkl
├── scenario1/
│   ├── catboost_v1.pkl
│   ├── catboost_v2.pkl
│   └── catboost_final.pkl
├── scenario2/
│   ├── catboost_v1.pkl
│   ├── catboost_v2.pkl
│   └── catboost_final.pkl
└── ensemble/
    └── weighted_avg_config.json
```

### 9.3 What to Log for Each Experiment

```python
EXPERIMENT_LOG = {
    'timestamp': datetime.now().isoformat(),
    'scenario': 'scenario1',
    'model_type': 'catboost',
    'model_version': 'v2',
    'features_version': 'fe_v2',
    
    # Hyperparameters
    'hyperparams': {
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 847,  # actual iterations with early stopping
        'l2_leaf_reg': 3
    },
    
    # Validation metrics
    'metrics': {
        'metric1': 1.15,
        'metric2': 0.92,
        'pe_bucket1': 1.32,
        'pe_bucket2': 0.98,
        'mae': 15234.5
    },
    
    # Notes
    'notes': 'Added n_gxs cummax feature, improved B1 by 5%',
    
    # Artifacts
    'model_path': 'models/scenario1/catboost_v2.pkl',
    'submission_path': 'submissions/submission_v2.csv'
}
```

---

## 10. Submission Strategy

### 10.1 Submission Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SUBMISSION WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: GENERATE PREDICTIONS                                        │
│  ├─ Load final trained models (Scenario 1 & 2)                      │
│  ├─ Load test data                                                   │
│  ├─ Build features with scenario-appropriate cutoffs                 │
│  ├─ Generate predictions                                             │
│  └─ Inverse transform if using log target                            │
│                                                                      │
│  STEP 2: FORMAT SUBMISSION                                           │
│  ├─ Ensure columns: country, brand_name, months_postgx, volume       │
│  ├─ Ensure all required rows are present                             │
│  ├─ No missing values in volume                                       │
│  ├─ No negative volumes (clip to 0 if needed)                        │
│  └─ Match exact order of submission_template.csv                      │
│                                                                      │
│  STEP 3: VALIDATE LOCALLY                                            │
│  ├─ Check row count matches template                                 │
│  ├─ Check no duplicate keys                                           │
│  ├─ Run metric_calculation.py on validation hold-out                 │
│  └─ Compare with previous submissions                                 │
│                                                                      │
│  STEP 4: UPLOAD & MONITOR                                            │
│  ├─ Upload to competition platform                                   │
│  ├─ Check for errors (wrong format → error message)                  │
│  ├─ Wait for leaderboard update                                       │
│  └─ Log public score                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Scenario Detection Logic

**Critical**: Correctly identifying which test series belong to Scenario 1 vs Scenario 2 is essential.

#### Detection Rules

| Scenario | Detection Criterion | Expected Count |
|----------|--------------------|-----------------|
| **Scenario 1** | Series where `min(months_postgx) == 0` | 228 |
| **Scenario 2** | Series where `min(months_postgx) == 6` | 112 |

```python
def detect_test_scenarios(test_volume: pd.DataFrame) -> dict:
    """
    Identify which test series belong to Scenario 1 vs 2.
    
    Returns:
        dict with 'scenario1' and 'scenario2' lists of (country, brand_name) tuples
    """
    # Find minimum months_postgx for each series
    series_min = test_volume.groupby(['country', 'brand_name'])['months_postgx'].min()
    
    scenario1_series = series_min[series_min == 0].index.tolist()
    scenario2_series = series_min[series_min == 6].index.tolist()
    
    return {
        'scenario1': scenario1_series,
        'scenario2': scenario2_series
    }

def validate_scenario_detection(scenarios: dict) -> None:
    """
    Sanity checks for scenario detection.
    """
    n_s1 = len(scenarios['scenario1'])
    n_s2 = len(scenarios['scenario2'])
    
    print(f"Scenario 1 series: {n_s1} (expected: 228)")
    print(f"Scenario 2 series: {n_s2} (expected: 112)")
    
    # Check no overlap
    s1_set = set(scenarios['scenario1'])
    s2_set = set(scenarios['scenario2'])
    overlap = s1_set & s2_set
    
    assert len(overlap) == 0, f"ERROR: {len(overlap)} series appear in both scenarios!"
    assert n_s1 + n_s2 == 340, f"ERROR: Total series {n_s1 + n_s2} != 340"
    
    print("✓ Scenario detection validated!")
```

### 10.3 Submission Generation Code

```python
# src/inference.py

def generate_submission(
    model_scenario1: Model,
    model_scenario2: Model,
    test_volume: pd.DataFrame,
    test_generics: pd.DataFrame,
    test_medicine_info: pd.DataFrame,
    submission_template: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate final submission file.
    """
    # Detect scenarios using validated logic
    scenarios = detect_test_scenarios(test_volume)
    validate_scenario_detection(scenarios)
    
    # Build features for Scenario 1 series
    scenario1_features = make_features(
        scenario1_panel, scenario='scenario1'
    )
    scenario1_pred = model_scenario1.predict(scenario1_features)
    
    # Build features for Scenario 2 series
    scenario2_features = make_features(
        scenario2_panel, scenario='scenario2'
    )
    scenario2_pred = model_scenario2.predict(scenario2_features)
    
    # Combine predictions
    all_predictions = pd.concat([scenario1_pred, scenario2_pred])
    
    # Post-processing
    all_predictions['volume'] = all_predictions['volume'].clip(lower=0)
    
    # Merge with template to ensure correct format
    submission = submission_template.merge(
        all_predictions,
        on=['country', 'brand_name', 'months_postgx'],
        how='left'
    )
    
    assert submission['volume'].notna().all(), "Missing predictions!"
    
    return submission[['country', 'brand_name', 'months_postgx', 'volume']]
```

### 10.4 Submission Schedule

| Submission # | Purpose | Timing |
|--------------|---------|--------|
| 1 | Format test | Within first 2 hours |
| 2 | Baseline score | After baseline implementation |
| 3-5 | Feature iterations | Day 2 |
| 6-8 | Model tuning | Day 2-3 |
| 9 | Final candidate | Saturday evening |
| 10 | Buffer/emergency | Sunday morning |

### 10.5 Final Submission Selection

On Sunday 9:30-10:30 AM:

1. Review all submission scores (public leaderboard)
2. **Do NOT just pick highest public score** (risk of overfitting to 30%)
3. Consider:
   - Consistency across multiple submissions
   - Local validation performance
   - Stability across seeds/splits
4. Select submission that balances public score + robustness

---

## 11. Business Story & Presentation

### 11.1 Presentation Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRESENTATION OUTLINE                              │
│                      (15-20 minutes)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PROBLEM FRAMING (2-3 min)                                        │
│     ├─ Business context: Generic erosion challenge                   │
│     ├─ Why it matters: Revenue planning, strategic decisions         │
│     └─ Our objective: Accurate forecasts for high-erosion brands     │
│                                                                      │
│  2. DATA UNDERSTANDING (3-4 min)                                     │
│     ├─ Key patterns discovered in EDA                                │
│     │   └─ B1 vs B2 erosion curves visualization                    │
│     ├─ Important features identified                                 │
│     └─ Data challenges and how we addressed them                     │
│                                                                      │
│  3. METHODOLOGY (4-5 min)                                            │
│     ├─ Feature engineering approach                                  │
│     │   └─ Key features: pre-entry stats, n_gxs dynamics, etc.      │
│     ├─ Model selection rationale                                     │
│     ├─ Scenario-specific adaptations                                 │
│     └─ Validation strategy (avoiding leakage)                        │
│                                                                      │
│  4. RESULTS & INSIGHTS (4-5 min)                                     │
│     ├─ Performance metrics (vs baseline)                             │
│     ├─ Bucket-specific results (B1 focus)                            │
│     ├─ Feature importance interpretation                             │
│     └─ Example predictions: good vs challenging cases                │
│                                                                      │
│  5. BUSINESS IMPLICATIONS (2-3 min)                                  │
│     ├─ How finance teams can use these forecasts                     │
│     ├─ Decision support examples                                      │
│     │   └─ "Brand X: High erosion predicted → recommend action"     │
│     └─ Limitations and when to override                              │
│                                                                      │
│  6. NEXT STEPS (1-2 min)                                             │
│     ├─ Model improvements with more time                              │
│     ├─ Integration with existing processes                            │
│     └─ Uncertainty quantification                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 Key Visualizations for Slides

1. **Erosion Pattern Comparison**
   - B1 vs B2 normalized volume curves
   - Show where the main drop happens

2. **Feature Importance**
   - Bar chart of top 10 features
   - Explain in business terms

3. **Prediction vs Actual Examples**
   - 2-3 representative brands
   - Show where model works well and where it struggles

4. **Business Decision Framework**
   - Simple flowchart: Forecast → Risk Level → Recommended Action

5. **Killer Comparison Slide** ⭐ (High Impact)

   Show on ONE compelling B1 example:

   ```
   ┌──────────────────────────────────────────────────────────────┐
   │  "Why Our Model Matters for High-Erosion Brands"             │
   ├──────────────────────────────────────────────────────────────┤
   │                                                               │
   │  [Chart: Actual vs Baseline vs Global Curve vs Our Model]   │
   │                                                               │
   │  Brand: [Example B1 brand]                                   │
   │                                                               │
   │  ┌─────────────────────────────────────────────────────┐     │
   │  │  Method              │  PE (Validation)            │     │
   │  ├─────────────────────────────────────────────────────┤     │
   │  │  Flat Baseline       │  2.34                       │     │
   │  │  Global Erosion Curve│  1.67                       │     │
   │  │  Our Model           │  0.89  ✓                    │     │
   │  └─────────────────────────────────────────────────────┘     │
   │                                                               │
   │  "62% reduction in prediction error vs baseline"             │
   │  "For the brands Novartis cares most about"                  │
   │                                                               │
   └──────────────────────────────────────────────────────────────┘
   ```

   **Key message**: "We are not just slightly better; we make a *meaningful* difference on the worst cases you care about."

### 11.3 Key Messages to Convey

| Message | Supporting Evidence |
|---------|---------------------|
| "We understand the business problem" | Clear EDA connecting patterns to erosion |
| "Our approach is principled" | No leakage, proper validation, scenario-aware |
| "We focus on what matters" | B1 performance, early months accuracy |
| "Results are interpretable" | Feature importance aligns with domain |
| "We know limitations" | Honest about uncertainty and edge cases |

### 11.4 Anticipated Jury Questions

| Question | Prepared Answer |
|----------|-----------------|
| "Why this model over alternatives?" | Gradient boosting handles tabular data + interactions well; fast to train; interpretable via feature importance |
| "How confident are you in B1 predictions?" | Show B1-specific error analysis; acknowledge higher uncertainty for severe erosion |
| "What drives high erosion?" | Top features: early n_gxs increase, biological=False, certain ther_areas |
| "How would you deploy this?" | Sketch: Monthly retrain → Dashboard → Brand team alerts |

### 11.5 Case Study Requirements

For a compelling presentation, prepare **at least 3 case studies**:

| Case Type | Purpose | Selection Criteria |
|-----------|---------|---------------------|
| **B1 Success** | Show model works on high-erosion | B1 brand where prediction closely matches actual |
| **B1 Challenge** | Show honesty about limitations | B1 brand where model struggles (explain why) |
| **B2 Contrast** | Show model differentiates buckets | B2 brand with different erosion pattern |

**For each case study, show**:

1. Time series plot: actual vs predicted volume (months 0-23)
2. Key features for this brand (n_gxs trajectory, biological, ther_area)
3. Brief narrative: "This brand shows [pattern] because [features]"

### 11.6 Decision-Support Layer (Optional but Attractive)

Convert predictions into **actionable prioritization** for finance teams:

**Priority Score Formula**:
$$\text{priority\_score}_j = \text{avg\_vol}_j \times (1 - \text{predicted\_mean\_erosion}_j)$$

- High score = Large brand with severe erosion = **High priority for attention**
- Low score = Small brand or mild erosion = Lower priority

**Alternative formulation** (if predicting early drop):
$$\text{priority\_score}_j = \text{avg\_vol}_j \times \text{predicted\_early\_drop}_{0\text{-}5}$$

**What to show**:

```text
┌──────────────────────────────────────────────────────────────────┐
│  TOP 10 HIGH-PRIORITY BRANDS (Illustrative Example)              │
├───────┬─────────┬─────────────────┬─────────────┬───────────────┤
│ Rank  │ Country │ Brand           │ Pred Erosion│ Priority Score│
├───────┼─────────┼─────────────────┼─────────────┼───────────────┤
│ 1     │ DE      │ DrugA           │ 0.12        │ 8,500,000     │
│ 2     │ FR      │ DrugB           │ 0.18        │ 6,200,000     │
│ 3     │ ES      │ DrugC           │ 0.15        │ 5,100,000     │
│ ...   │ ...     │ ...             │ ...         │ ...           │
└───────┴─────────┴─────────────────┴─────────────┴───────────────┘
```

**Key message to jury**: "This is not a prescriptive tool, but a way to **flag at-risk revenues** so finance teams can prioritize which brands need strategic attention."

---

## 12. Risk Mitigation

### 12.1 Technical Risks

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Overfitting to public LB | Trust local validation; ensemble for stability | Use most stable submission |
| Feature leakage | Strict cutoff rules; code review | Simplify to pre-entry features only |
| Model instability | Multiple seeds; regularization | Use simple baseline |
| Wrong submission format | Early test submission; validation script | Fix immediately if error |
| Code breaks on test data | Defensive coding; null checks | Manual prediction if needed |

### 12.2 Time Risks

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Too much time on EDA | Set strict time box (4-6 hours) | Move to modeling with partial EDA |
| Endless model tuning | Set tuning budget (20-30 runs) | Use best-so-far config |
| Slides not ready | Start slides by Saturday evening | Minimal viable deck |
| Last-minute bugs | Submit "safe" version early | Use last working submission |

### 12.3 Competition Risks

| Risk | Mitigation |
|------|------------|
| Platform down | Have local predictions ready; monitor announcements |
| Submission limit hit | Plan submissions carefully; save attempts for final day |
| Public/Private mismatch | Don't over-optimize public; trust local CV |
| Train/test distribution shift | Adversarial validation; monitor AUC; simplify or regularize features that overfit to test |

### 12.4 Edge Case Handling (Tricky Series)

Some series will be problematic. Identify and handle them explicitly:

#### Identification Criteria

| Edge Case | Detection | Count in EDA |
|-----------|-----------|---------------|
| **Very low baseline** | `Avg_j < P5(Avg_j)` or `Avg_j < 100` | TBD |
| **Short history** | `< 6 months` of pre-entry data | TBD |
| **High volatility** | `std(volume) / Avg_j > 0.5` | TBD |
| **Zero months** | Any month with `volume == 0` in pre-entry | TBD |

#### Fallback Policy

```python
def apply_edge_case_fallback(predictions: pd.DataFrame, 
                              edge_case_series: list,
                              global_erosion_curve: pd.Series) -> pd.DataFrame:
    """
    For problematic series, use conservative fallback prediction.
    """
    for country, brand in edge_case_series:
        mask = (predictions['country'] == country) & \
               (predictions['brand_name'] == brand)
        
        # Fall back to global erosion curve × brand's avg_vol
        avg_vol = predictions.loc[mask, 'avg_vol'].iloc[0]
        predictions.loc[mask, 'volume'] = (
            global_erosion_curve * avg_vol
        ).values
    
    return predictions
```

#### Presentation Talking Point

> "For extremely small or newly launched brands, model uncertainty is high. We recommend manual review for brands with baseline volume below [X]. Our model correctly identifies these as high-uncertainty cases."

This shows the jury you understand model limitations.

### 12.5 Work Prioritization (If Time Runs Short)

The 48-hour window is tight. If time pressure hits, follow this priority ranking:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRIORITY 1: NON-NEGOTIABLE (Must Complete)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ✓ Clean data pipeline (load, join, preprocess)                     │
│  ✓ Baseline model + hero GBM for BOTH scenarios                     │
│  ✓ Proper validation with official metric                           │
│  ✓ At least ONE valid submission uploaded                            │
│  ✓ Basic slides structure                                            │
├─────────────────────────────────────────────────────────────────────┤
│  PRIORITY 2: HIGH VALUE (Strong Impact If Done)                      │
├─────────────────────────────────────────────────────────────────────┤
│  ○ Metric-aligned sample weights in training                         │
│  ○ 2-3 case study plots (B1 success, B1 challenge, B2 contrast)     │
│  ○ Bucket-1 focused error analysis                                   │
│  ○ Feature importance interpretation                                 │
│  ○ Edge case handling policy                                         │
├─────────────────────────────────────────────────────────────────────┤
│  PRIORITY 3: NICE-TO-HAVE (Drop If Pressed)                          │
├─────────────────────────────────────────────────────────────────────┤
│  △ Ensemble of multiple models                                       │
│  △ Adversarial validation                                            │
│  △ SHAP values                                                       │
│  △ Hyperparameter tuning beyond 10-15 runs                           │
│  △ Decision-support priority score                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Rule of thumb**: If it's Saturday 8 PM and you don't have Priority 1 complete, stop everything else and focus only on Priority 1.

---

## 13. Code Organization

### 13.1 Project Structure

```
novartis_datathon_2025/
├── README.md                    # Project overview, setup instructions
├── requirements.txt             # Python dependencies
│
├── configs/                     # Configuration files
│   ├── data.yaml               # Data paths
│   ├── features.yaml           # Feature definitions
│   └── model_*.yaml            # Model configurations
│
├── data/
│   ├── raw/                    # Original data (READ ONLY)
│   │   ├── TRAIN/
│   │   └── TEST/
│   ├── processed/              # Cleaned, joined data
│   └── interim/                # Intermediate outputs
│
├── docs/
│   ├── NOVARTIS_DATATHON_2025_COMPLETE_GUIDE.md
│   ├── problem.md              # 1-2 sentence problem statement + business context
│   ├── data_schema.md          # Tables, keys, columns, relationships
│   ├── validation.md           # Split strategy and rationale
│   ├── experiments_log.md      # Short log of experiments (or CSV)
│   └── planning/
│       └── approach.md         # This document
│
├── notebooks/
│   ├── 00_eda.ipynb           # Exploratory analysis
│   ├── 01_feature_prototype.ipynb  # Feature engineering tests
│   ├── 02_model_sanity.ipynb  # Model sanity checks
│   └── colab/
│       └── main.ipynb         # Full pipeline for Colab
│
├── src/
│   ├── __init__.py
│   ├── data.py                # Data loading & preprocessing
│   ├── features.py            # Feature engineering
│   ├── validation.py          # Validation framework
│   ├── train.py               # Model training
│   ├── inference.py           # Prediction generation
│   ├── evaluate.py            # Metric computation
│   └── utils.py               # Utilities
│
├── models/                     # Trained model artifacts
│   ├── baselines/
│   ├── scenario1/
│   └── scenario2/
│
├── submissions/                # Generated submissions
│   ├── submission_v1.csv
│   └── submission_final.csv
│
└── tests/                      # Unit tests
    └── test_smoke.py
```

### 13.2 Documentation Files

Maintain these concise docs during the competition:

#### `docs/problem.md`
```markdown
# Problem Statement

**Objective**: Forecast monthly sales volumes for branded drugs over 24 months 
after generic entry, with emphasis on high-erosion brands (Bucket 1).

**Business Use**:
- Anticipate revenue loss from generic competition
- Plan post-patent strategies (pricing, production, inventory)
- Prioritize portfolio management attention

**Evaluation**: Weighted percentage error (PE) with 2× weight on Bucket 1.
```

#### `docs/data_schema.md`
```markdown
# Data Schema

## Tables
| Table | Primary Key | Description |
|-------|-------------|-------------|
| df_volume_* | (country, brand_name, months_postgx) | Monthly volumes |
| df_generics_* | (country, brand_name, months_postgx) | Generic counts |
| df_medicine_info_* | (country, brand_name) | Static drug info |

## Joins
- Volume ← Generics on (country, brand_name, months_postgx)
- Volume ← Medicine_info on (country, brand_name)
```

#### `docs/validation.md`
```markdown
# Validation Strategy

**Split**: Series-level (entire country-brand held out), 80/20
**Stratification**: By bucket (maintain B1/B2 ratio)
**Metric**: Official compute_metric1() and compute_metric2()
**Why**: Mimics real scenario where we predict new brands, not partial series
```

#### `docs/experiments_log.md`
```markdown
# Experiment Log

| Date | Model | Features | Val Metric1 | Val Metric2 | Notes |
|------|-------|----------|-------------|-------------|-------|
| 11/28 | baseline_flat | - | 2.34 | 1.89 | Naive baseline |
| 11/28 | catboost_v1 | fe_v1 | 1.45 | 1.12 | First GBM |
```

### 13.3 Core Module Specifications

#### `src/data.py`
```python
"""Data loading and preprocessing."""

def load_raw_data(data_dir: str, split: str) -> dict: ...
def prepare_base_panel(volume, generics, medicine_info) -> pd.DataFrame: ...
def compute_pre_entry_stats(panel) -> pd.DataFrame: ...
def handle_missing_values(panel) -> pd.DataFrame: ...
def adversarial_validation(train_features, test_features) -> dict: ...
```

#### `src/features.py`
```python
"""Feature engineering with scenario awareness."""

def make_features(panel: pd.DataFrame, scenario: str) -> pd.DataFrame: ...
def add_pre_entry_features(df) -> pd.DataFrame: ...
def add_time_features(df) -> pd.DataFrame: ...
def add_generics_features(df, cutoff_month) -> pd.DataFrame: ...
def add_drug_features(df) -> pd.DataFrame: ...
def add_early_erosion_features(df) -> pd.DataFrame: ...  # Scenario 2 only
```

#### `src/validation.py`
```python
"""Validation framework."""

def create_validation_split(panel, val_fraction, stratify_by) -> Tuple: ...
def simulate_scenario(train_df, val_df, scenario: str) -> Tuple: ...
def compute_validation_metrics(actuals, predictions, scenario) -> dict: ...
```

#### `src/train.py`
```python
"""Model training."""

def train_model(train_features, train_target, config) -> Model: ...
def train_with_early_stopping(train, val, config) -> Tuple[Model, dict]: ...
def save_model(model, path): ...
def load_model(path) -> Model: ...
```

#### `src/inference.py`
```python
"""Prediction generation."""

def generate_submission(models, test_data, template) -> pd.DataFrame: ...
def post_process_predictions(predictions) -> pd.DataFrame: ...
def validate_submission_format(submission, template) -> bool: ...
```

#### `src/evaluate.py`
```python
"""Evaluation metrics."""

# Wrapper around metric_calculation.py
def compute_metric1(actual, pred, aux) -> float: ...
def compute_metric2(actual, pred, aux) -> float: ...
def compute_bucket_metrics(actual, pred, aux) -> dict: ...
```

---

## 14. Deliverables Checklist

### 14.1 Competition Deliverables

#### For All Teams
- [ ] **Submission file** (`submission_final.csv`)
  - Format: `country, brand_name, months_postgx, volume`
  - All required rows present
  - Selected as final by 10:30 AM Sunday

#### For Top 5 Finalists (by 12:00 PM Sunday)
- [ ] **Presentation slides** (follow template)
  - Problem understanding
  - Methodology
  - Results & insights
  - Business implications
  - Limitations & next steps

- [ ] **Code package**
  - All code used to generate final submission
  - README with run instructions
  - Requirements/dependencies

### 14.2 Internal Quality Checklist

#### Data Pipeline
- [ ] Data loads correctly for train and test
- [ ] Joins produce expected row counts
- [ ] No data leakage in feature engineering
- [ ] Missing values handled consistently

#### Modeling
- [ ] Baseline implemented and scored
- [ ] Hero model trained for both scenarios
- [ ] Validation metrics computed correctly
- [ ] Model artifacts saved

#### Submission
- [ ] Format validated against template
- [ ] No missing predictions
- [ ] No negative volumes
- [ ] Test submission uploaded and scored

#### Documentation
- [ ] Experiment log maintained
- [ ] Key decisions documented
- [ ] Reproducible from README

### 14.3 Pre-Submission Sanity Checks

```python
def sanity_check_submission(submission_df, template_df):
    """Final checks before submission."""
    
    # 1. Row count
    assert len(submission_df) == len(template_df), "Row count mismatch!"
    
    # 2. Column names
    assert list(submission_df.columns) == ['country', 'brand_name', 'months_postgx', 'volume']
    
    # 3. No missing values
    assert submission_df['volume'].notna().all(), "Missing predictions!"
    
    # 4. No negative volumes
    assert (submission_df['volume'] >= 0).all(), "Negative volumes!"
    
    # 5. Keys match template
    template_keys = set(zip(template_df['country'], template_df['brand_name'], template_df['months_postgx']))
    submission_keys = set(zip(submission_df['country'], submission_df['brand_name'], submission_df['months_postgx']))
    assert template_keys == submission_keys, "Key mismatch!"
    
    # 6. No duplicates
    assert not submission_df.duplicated(subset=['country', 'brand_name', 'months_postgx']).any()
    
    print("✓ All sanity checks passed!")
```

---

## Appendix A: Quick Reference Commands

### Environment Setup
```bash
# Create environment
conda create -n novartis python=3.10
conda activate novartis
pip install -r requirements.txt

# Or using pip only
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Key Commands
```bash
# Run EDA notebook
jupyter notebook notebooks/00_eda.ipynb

# Train models
python src/train.py --scenario scenario1 --config configs/model_cat.yaml
python src/train.py --scenario scenario2 --config configs/model_cat.yaml

# Generate submission
python src/inference.py --output submissions/submission_v1.csv

# Validate submission
python -c "from src.inference import validate_submission; validate_submission('submissions/submission_v1.csv')"
```

---

## Appendix B: Key Formulas

### Mean Generic Erosion
$$\text{Mean Erosion} = \frac{1}{24} \sum_{i=0}^{23} \frac{Vol_i}{Avg_j}$$

### Pre-Entry Average
$$Avg_j = \frac{1}{12} \sum_{i=-12}^{-1} Vol_{j,i}$$

### Bucket Classification
$$\text{Bucket} = \begin{cases} 1 & \text{if Mean Erosion} \leq 0.25 \\ 2 & \text{if Mean Erosion} > 0.25 \end{cases}$$

### Final Metric
$$PE = \frac{2}{n_{B1}} \sum_{j \in B1} PE_j + \frac{1}{n_{B2}} \sum_{j \in B2} PE_j$$

---

## Appendix C: Important Numbers

| Item | Value |
|------|-------|
| Training series | 1,953 |
| Test series | 340 |
| Scenario 1 test | 228 |
| Scenario 2 test | 112 |
| Pre-entry window | 12 months (for avg_vol) |
| Forecast horizon | 24 months |
| Bucket 1 threshold | ≤ 0.25 |
| Bucket 1 weight | 2× |
| Submission limit | 3 per 8 hours |
| Public LB split | 30% |
| Final selection window | 9:30-10:30 AM Sunday |

---

## Appendix D: Ideas Ported from Strategy Archive

The following valuable concepts were incorporated from our initial strategy document (`approach_old.md`):

1. **Three-Phase Structure** (Section 1.5)
   - Phase 1: Understanding & Design
   - Phase 2: Core Implementation ("Guaranteed Good")
   - Phase 3: Improvement, Edge & Polish

2. **Design Documentation** (Section 13.2)
   - `problem.md`, `data_schema.md`, `validation.md`, `experiments_log.md`
   - Quick references during Q&A and for reproducibility

3. **Adversarial Validation** (Section 8.5)
   - Detect train/test distribution shift
   - Identify unstable features
   - Shows methodological rigor to jury

4. **Story Draft Milestone** (Section 3.3)
   - Lock first draft of business story by end of Phase 2
   - Prevents "good model, rushed slides" failure mode

5. **Case Study Focus** (Section 11.5)
   - At least 3 case studies: B1 success, B1 challenge, B2 contrast
   - Makes presentation concrete and memorable

6. **Decision-Support Layer** (Section 11.6)
   - Priority score to rank at-risk brands
   - Converts predictions into actionable finance decisions

7. **Target Definition & Metric Alignment** (Section 6.4)
   - Train on normalized volume (volume / Avg_j)
   - Sample weights to approximate official PE

8. **Scenario Detection Logic** (Section 10.2)
   - Explicit rules: Scenario 1 = min(months_postgx) == 0
   - Validation function with expected counts

9. **Edge Case Handling** (Section 12.4)
   - Identify problematic series (low volume, short history)
   - Fallback policy for conservative predictions

10. **Work Prioritization** (Section 12.5)
    - Clear ranking: Non-negotiable → High value → Nice-to-have
    - Decision rule for time pressure

11. **Killer Comparison Slide** (Section 11.2)
    - Baseline vs Global Curve vs Our Model on B1 example
    - Quantified improvement for jury impact

The original `approach_old.md` is archived for reference but this document (`approach.md`) is the **single source of truth** for the competition.

---

*Document Version: 1.2*
*Last Updated: November 2025*
*Purpose: Implementation guide for Novartis Datathon 2025*
