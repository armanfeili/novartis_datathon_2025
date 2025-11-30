# Model Comparison Results

## Experiment Summary

**Date:** November 29, 2025  
**Scenarios Tested:** S1 (full forecast months 0-23), S2 (months 6-23 with early actuals)  
**Models Tested:** CatBoost (3 configs), LightGBM, XGBoost, Neural Network, Historical Curve Baseline  
**Environment Note:** LightGBM and XGBoost required `OMP_NUM_THREADS=1` due to OpenMP library conflicts.

---

## Scenario 1 Results (Sorted by Official Metric - Lower is Better)

| Rank | Model | Config | Official Metric | RMSE | MAE | Train Time (s) |
|------|-------|--------|-----------------|------|-----|----------------|
| 1 | CatBoost | depth8 | **0.7509** | 0.2476 | 0.1773 | 29.47 |
| 2 | LightGBM | default | 0.7526 | 0.2489 | 0.1776 | 3.47 |
| 3 | CatBoost | baseline_depth6 | 0.7692 | 0.2488 | 0.1795 | 18.93 |
| 4 | XGBoost | default | 0.7754 | 0.2480 | 0.1768 | 5.51 |
| 5 | CatBoost | depth4 | 0.7817 | 0.2488 | 0.1808 | 20.08 |
| 6 | Historical Curve | default | 0.9165 | 0.2981 | 0.2375 | 0.11 |
| 7 | Neural Network | default | 0.9648 | 0.2999 | 0.2329 | 134.76 |

---

## Scenario 2 Results (Sorted by Official Metric - Lower is Better)

| Rank | Model | Config | Official Metric | RMSE | MAE | Train Time (s) |
|------|-------|--------|-----------------|------|-----|----------------|
| 1 | CatBoost | baseline_depth6 | **0.2742** | 0.2055 | 0.1265 | 15.39 |
| 2 | CatBoost | depth4 | 0.2748 | 0.2058 | 0.1276 | 13.64 |
| 3 | CatBoost | depth8 | 0.2940 | 0.2052 | 0.1276 | 15.39 |
| 4 | LightGBM | default | 0.2941 | 0.2095 | 0.1293 | 2.79 |
| 5 | XGBoost | default | 0.2993 | 0.2100 | 0.1293 | 5.54 |
| 6 | Neural Network | default | 0.5815 | 0.2884 | 0.2015 | 243.82 |
| 7 | Historical Curve | default | 1.0840 | 0.2997 | 0.2371 | 0.11 |

---

## Analysis & Interpretation

### Best Models by Scenario

| Scenario | Best Model | Best Config | Official Metric | Runner-up |
|----------|------------|-------------|-----------------|-----------|
| S1 | CatBoost | depth8 | 0.7509 | LightGBM (0.7526) |
| S2 | CatBoost | baseline_depth6 | 0.2742 | CatBoost depth4 (0.2748) |

### Key Observations

1. **CatBoost Remains the Hero Model**
   - CatBoost variants dominate both scenarios
   - S1: CatBoost depth8 wins (0.7509), 2.4% better than baseline depth6 (0.7692)
   - S2: CatBoost baseline depth6 wins (0.2742), with depth4 very close (0.2748)

2. **LightGBM is a Strong Contender**
   - **S1:** LightGBM (0.7526) is nearly as good as CatBoost depth8 (0.7509) - only 0.2% worse
   - **S1:** LightGBM beats CatBoost baseline depth6 (0.7692) by 2.2%
   - **S2:** LightGBM (0.2941) is competitive with CatBoost depth8 (0.2940)
   - **Training speed:** 5-8x faster than CatBoost

3. **XGBoost Performance**
   - Slightly behind LightGBM and CatBoost on both scenarios
   - S1: 0.7754 (3.3% worse than best)
   - S2: 0.2993 (9.2% worse than best)

4. **Depth Configuration Impact**
   - **S1:** Deeper trees (depth=8) perform best - the full forecast problem benefits from higher model capacity
   - **S2:** Shallower trees (depth=6) perform best - early actuals reduce need for complex extrapolation
   - **Tradeoff:** Depth 4 underperforms on S1 but is competitive on S2

5. **S2 is Inherently Easier**
   - S2 official metrics (0.27-0.58) are much lower than S1 (0.75-0.97)
   - Having early actuals (months 0-5) provides strong signal for forecasting months 6-23
   - Gradient boosting methods show ~60-65% improvement from S1 to S2

6. **Neural Network and Baseline Performance**
   - NN performs poorly: worst gradient-boosting-class model on both scenarios
   - Training time 10-50x longer than boosting methods with worse results
   - Historical curve baseline performs worst on S2 - falls back to global predictions

### Relative Performance Analysis

**Scenario 1 - vs CatBoost depth8 (best):**
| Model | Official Metric | Relative Difference |
|-------|-----------------|---------------------|
| CatBoost depth8 | 0.7509 | baseline |
| LightGBM | 0.7526 | +0.2% |
| CatBoost depth6 | 0.7692 | +2.4% |
| XGBoost | 0.7754 | +3.3% |
| CatBoost depth4 | 0.7817 | +4.1% |

**Scenario 2 - vs CatBoost depth6 (best):**
| Model | Official Metric | Relative Difference |
|-------|-----------------|---------------------|
| CatBoost depth6 | 0.2742 | baseline |
| CatBoost depth4 | 0.2748 | +0.2% |
| CatBoost depth8 | 0.2940 | +7.2% |
| LightGBM | 0.2941 | +7.3% |
| XGBoost | 0.2993 | +9.2% |

### Recommendations

1. **Hero Models for Production:**
   - **Scenario 1:** CatBoost depth=8
   - **Scenario 2:** CatBoost depth=6 (baseline config)

2. **Ensemble Candidates:**
   - CatBoost depth8 + LightGBM for S1 (complementary, similar performance)
   - CatBoost depth6 + CatBoost depth4 for S2 (both strong, may capture different patterns)

3. **Next Experiments to Try:**
   - CatBoost depth=7 as middle ground for both scenarios
   - LightGBM hyperparameter tuning (currently using defaults)
   - Weighted ensemble of CatBoost + LightGBM

4. **Skip:**
   - Neural network (too slow, worse performance)
   - Historical curve baseline (broken implementation, poor results)

### Metric Interpretation

- **Official Metric:** Weighted RMSE with bucket adjustments (competition target, lower is better)
- **RMSE/MAE:** Normalized to avg_vol_12m scale, measuring prediction error on y_norm

---

## Raw Data

See `model_comparison_metrics.csv` for the complete dataset.
