# Data Cleaning, Feature Engineering, and Model Selection Plan

This document summarizes the recommended approach to **clean**, **preprocess**, **engineer features**, and **select models** for the LOE/generic erosion forecasting project, based on the EDA findings.

---

## 1. Data Cleaning & Preprocessing

### 1.1 Basic Sanity Checks

- Drop exact duplicates on:  
  `country, brand_name, months_postgx`  
- Verify for each brand:
  - No brand has multiple rows for the same month.
  - `months_postgx` covers the expected range (e.g. -24 to +23).  
- Decide how to handle brands with very short history (e.g. only a few pre-LOE months):
  - Keep them but rely on global models that share information across brands.

---

### 1.2 Missing Values

**`time_to_50pct`** (train only):

- Create two columns:
  - `reached_50pct` = 1 if `time_to_50pct` is defined, else 0.
  - `time_to_50pct_imputed`:
    - Set to `time_to_50pct` where available.
    - Set to `24` (or final observed month) where the 50% erosion threshold is never reached.
- Use these as **brand-level features** for training and analysis (avoid leaking to test post-entry periods).

**`avg_vol` (`avg_j`)**:

- Recompute where possible as the mean of volume over months -12..-1 (pre-LOE).
- For brands without 12 months of pre-LOE:
  - Impute using median `avg_vol` by therapeutic area or
  - Fit a simple regression model for log(avg_vol) using country, therapeutic area, hospital_rate, etc.

**`n_gxs` (number of generics)**:

- Where missing, treat as `0` (no generics yet), provided this matches the data structure.
- Check the frequency and distribution of NAs to ensure this assumption is safe.

---

### 1.3 Outlier Treatment

- For **`n_gxs`**:
  - Cap at a maximum value, e.g. `15`: `n_gxs_capped = min(n_gxs, 15)`.
  - Optionally create `log1p(n_gxs_capped)` to capture diminishing returns.

- For **`volume` and `vol_norm`**:
  - Use `vol_norm` as the main signal for modeling; keep `volume` primarily for revenue/ROI calculations.
  - Do **not** remove `vol_norm > 1` (growth or special contracting effects); instead, add a flag:
    - `vol_norm_gt1` = 1 if `vol_norm > 1` else 0.

---

### 1.4 Scaling

- Tree-based models do not require scaling, but neural and linear models benefit from it.

Suggested scaling:

- `months_postgx`: leave raw or apply StandardScaler (mean 0, std 1).
- `n_gxs_capped`: apply `log1p` then StandardScaler.
- `hospital_rate`: already in [0, 1]; use raw or MinMaxScaler.
- `avg_vol`: apply `log1p` then StandardScaler.
- `vol_norm`: keep raw for interpretability.

---

### 1.5 Categorical Encoding (Avoiding Leakage)

Key categorical variables: `country`, `therapeutic_area`, `main_package`, etc.

Approach:

- Use **target encoding** for high-cardinality categories:
  - Encode `country`, `therapeutic_area`, `main_package` with mean `vol_norm` or mean bucket probability.
- Implement target encoding **within cross-validation folds**:
  - Compute encodings on training folds only.
  - Apply encodings to validation fold to avoid target leakage.
- For additional structure, define:
  - `ther_area_erosion_rank` = rank of each therapeutic area by its average erosion (1 = highest erosion).

---

## 2. Feature Engineering

The goal is to exploit time structure, competition, and brand characteristics while respecting what is known at forecast time for each scenario.

### 2.1 Time Features

The metric weights time windows differently (0–5, 6–11, 12–23), so the model should know where in the trajectory a point is.

Create:

- `months_postgx`
- `months_postgx_sq` = `months_postgx ** 2` (captures curvature).
- Period flags:
  - `is_early_period` = 1 if `0 <= months_postgx <= 5` else 0.
  - `is_mid_period` = 1 if `6 <= months_postgx <= 11` else 0.
  - `is_late_period` = 1 if `12 <= months_postgx <= 23` else 0.

For pre-LOE data, similar buckets can be defined (e.g. `pre_early`, `pre_late`), especially for Scenario 1 feature creation.

---

### 2.2 Competition Features (`n_gxs`)

Number of generic competitors is a core driver of erosion.

Create features like:

- `n_gxs_capped` = `min(n_gxs, 15)`.
- `log_n_gxs` = `log1p(n_gxs_capped)`.
- `has_competition` = 1 if `n_gxs > 0` else 0.
- `high_competition` = 1 if `n_gxs >= 5` else 0.
- `competition_intensity` = `n_gxs / (months_postgx + 1)` for post-LOE periods.

Brand-level competition features:

- `max_n_gxs_post` = maximum `n_gxs` observed post-LOE (for training / analysis).
- `time_to_first_generic` (if launch dates are available): time between brand launch and first generic.

---

### 2.3 Lag & Rolling Features

Time-series structure is crucial.

For each `(country, brand_name)`:

- Lagged normalized volumes:
  - `vol_norm_lag1`
  - `vol_norm_lag3`
  - `vol_norm_lag6`
- Differences and changes:
  - `vol_norm_diff1` = `vol_norm - vol_norm_lag1`
  - `vol_norm_diff3` = `vol_norm - vol_norm_lag3`
  - `vol_norm_pct_change` = `(vol_norm - vol_norm_lag1) / (vol_norm_lag1 + ε)`
- Rolling statistics:
  - `vol_norm_roll_mean_3`
  - `vol_norm_roll_mean_6`
  - `vol_norm_roll_std_3`
  - `erosion_rate_3m` = `vol_norm_lag3 - vol_norm`

**Scenario considerations:**

- Scenario 1:
  - At forecast origin (month 0), only pre-LOE history is available for test brands.  
  - Design lag and rolling features that are definable from pre-LOE data.
- Scenario 2:
  - At forecast origin (month 6), months 0–5 are known, so lags and rolling stats over 0–5 are valid and should be used.

---

### 2.4 Brand-Level Static Features

From pre-LOE history:

- `avg_vol` / `avg_j` (recomputed baseline volume).
- **Pre-LOE trend**:
  - Slope of a linear regression of `vol_norm` on time for months [-12, -1].
- **Pre-LOE volatility**:
  - Standard deviation of `vol_norm` in months [-12, -1].
- `pre_loe_growth_flag`:
  - 1 if pre-LOE trend slope > 0, else 0.

From EDA-derived metrics (train only, for analysis and optional modeling):

- `time_to_50pct_imputed`
- `reached_50pct`

From categories and segmentation:

- `ther_area_erosion_rank`
- `is_high_erosion_area` (e.g. Anti-infectives, Oncology, MSK flagged as high erosion).
- `hospital_rate_bucket` ∈ {0–25, 25–50, 50–75, 75–100}.

---

## 3. Model Selection Strategy

Data profile: ~94k rows, ~2k brands, ~40–60 features. This is ideal for **global tabular models** plus simple time-series components.

### 3.1 Global Models Instead of Per-Brand ARIMA

Given:
- Many brands,
- Moderate history length per brand,
- Need to generalize to new brands,

Prefer **global models** that learn patterns across all series:

- Gradient boosting (LightGBM / XGBoost / CatBoost).
- Optional neural models (LSTM / Temporal Fusion Transformer) if time allows.

Per-brand ARIMA is less attractive due to scale and complexity.

---

### 3.2 Separate Pipelines by Scenario

#### Scenario 1 (No Post-Entry Data)

- At forecast origin (t = 0): only pre-LOE history and static features are known.
- Need to forecast months 0–23.

Training strategies:

1. **Horizon-as-row**:
   - Each row represents: (brand, forecast_origin=0, horizon `h`, features_at_origin, `h`).
   - Target: `vol_norm_h` (normalized volume at month `h`).
   - Features at origin: brand-level static features, pre-LOE patterns, `n_gxs` trajectory assumptions if used, and horizon `h` itself as a feature.

2. **Separate model per horizon**:
   - Train 24 models, one for each forecast month (0–23).
   - Each model uses the same origin features but targets different `vol_norm_h`.

Use **LightGBM** (or similar) with sample weights that reflect bucket importance and metric weighting.

#### Scenario 2 (First 6 Months Known)

- At forecast origin (t = 6): months 0–5 are known.
- Need to forecast months 6–23.

Same general structure as Scenario 1, but:

- Features also include post-entry early behavior:
  - mean `vol_norm` in months 0–5,
  - slope of `vol_norm` from 0–5,
  - last observed value at month 5,
  - rolling stats over 0–5.

Use a separate model (or model head) for Scenario 2, with similar horizon-as-row or per-horizon design.

---

### 3.3 Handling Bucket 1 vs Bucket 2

Bucket 1 (high erosion) is minority but weighted more heavily in the competition metric.

Strategies:

1. **Bucket classifier (optional)**:
   - Train a classifier to predict Bucket 1 vs Bucket 2 from pre-LOE features (and early post-entry features for Scenario 2).
   - Use:
     - For analysis (where erosion is likely to be severe).
     - To route brands into specialized regressors if desired.

2. **Bucket-aware regressors**:

- **Option A**: Single global regressor with sample weights:
  - Weight Bucket 1 observations ×2.
  - Weight Bucket 2 observations ×1.

- **Option B**: Two separate regressors:
  - `Model_B1` trained on Bucket 1 brands (with oversampling or weights).
  - `Model_B2` trained on Bucket 2 brands.
  - Use predicted bucket probabilities to blend the two at prediction time.

Given time constraints, starting with **sample weights in one global model** is recommended.

---

### 3.4 Aligning Training Loss with the Competition Metric

The competition metric emphasizes:

- Scenario 1:
  - All months 0–23 (20%),
  - Sum(0–5) (50%),
  - Sum(6–11) (20%),
  - Sum(12–23) (10%).

- Scenario 2:
  - Months 6–23 (20%),
  - Sum(6–11) (50%),
  - Sum(12–23) (30%).

Approximate this within a supervised learning loss by using **sample weights** for each horizon:

- For each training row (brand, horizon `h`), assign a **time-window weight** based on its bucket:
  - Scenario 1 example:
    - `h in [0–5]`: highest weight (reflecting its dominance in the metric).
    - `h in [6–11]`: medium weight.
    - `h in [12–23]`: lower weight.
- Multiply by **bucket weight**:
  - Bucket 1: ×2
  - Bucket 2: ×1
- Use these as `sample_weight` in LightGBM/XGBoost.

This encourages the model to focus on the horizons and erosion profiles that matter most for the final score.

---

### 3.5 Hybrid / Ensemble Approach

A practical and strong ensemble approach:

1. **Physics baseline**:
   - Per brand, fit a simple exponential decay curve for post-LOE normalized volume:
     - `vol_norm(t) ≈ a * exp(-b * t) + c` for `t >= 0`.
   - Provides an interpretable baseline and often reasonable shape.

2. **Global ML model (LightGBM)**:
   - Either predict `vol_norm` directly or predict the **residual**:
     - `residual = true_vol_norm - physics_baseline(t)`.

3. **Optional ARHOW / classic TS component**:
   - For brands with long stable history, fit ARIMA + Holt-Winters hybrid (ARHOW) and use its forecasts as an additional feature or another ensemble leg.

Final prediction could be a weighted combination:

```text
y_pred = w_phys * y_phys + w_ml * y_ml (+ w_ts * y_ts)
```

Weights (`w_phys`, `w_ml`, `w_ts`) can be tuned on a validation set.

Given dataset size and complexity, a strong baseline is:

- exponential-decay baseline +
- global LightGBM regressor.

---

## 4. Cross-Validation Design

To avoid leakage and ensure realistic brand-level generalization:

- Use **GroupKFold** or **StratifiedGroupKFold**:
  - `group = brand_name`
  - If possible, `stratify = bucket`.

This ensures:

- All months for a given brand are either in training or validation (no leakage across time for that brand).
- Each fold has a mix of Bucket 1 and Bucket 2.

For scenario-based evaluation:

- Design folds over brands, then within each fold:
  - Simulate Scenario 1 and Scenario 2 forecast origins (month 0, month 6) and measure errors as the datathon metric approximations.

---

## 5. High-ROI Minimal Stack (If Time is Limited)

If time is tight, prioritize the following:

1. **Cleaning**:
   - Cap `n_gxs`, create `log_n_gxs`.
   - Recompute `avg_vol` from -12..-1.
   - Encode `therapeutic_area` into an erosion-based rank.

2. **Features**:
   - Time indicators: `months_postgx`, early/mid/late period flags.
   - Competition: `log_n_gxs`, `has_competition`, `high_competition`.
   - Lags: `vol_norm_lag1`, `lag3`, `diff1`, `roll_mean_3`.
   - Brand static: `log_avg_vol`, `hospital_rate_bucket`, `is_high_erosion_area`.

3. **Models**:
   - Scenario 1:
     - Horizon-as-row LightGBM with time-window and bucket-based sample weights.
   - Scenario 2:
     - Same structure plus early post-LOE summary features (0–5).

4. **Ensemble**:
   - Add a simple exponential-decay baseline, combine with LightGBM outputs.

This gives a pipeline that:
- Respects the metric structure,
- Leverages the strongest EDA insights,
- And remains feasible under Datathon time pressure.

