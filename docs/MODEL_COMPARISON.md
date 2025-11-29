# Model Comparison and Hyperparameter Reference

**Novartis Datathon 2025 - Generic Erosion Forecasting**

Generated: November 29, 2025

---

## Table of Contents
1. [Model Overview](#model-overview)
2. [Performance Comparison](#performance-comparison)
3. [Detailed Hyperparameters by Model](#detailed-hyperparameters-by-model)
4. [Sweep Configurations](#sweep-configurations)
5. [Best Practices](#best-practices)

---

## Model Overview

| # | Model | Type | Priority | Config File | Status |
|---|-------|------|----------|-------------|--------|
| 1 | **CatBoost** | Gradient Boosting | 3 (Tertiary) | `model_cat.yaml` | ✅ Working |
| 2 | **LightGBM** | Gradient Boosting | 2 (Secondary) | `model_lgbm.yaml` | ⚠️ Segfault |
| 3 | **XGBoost** | Gradient Boosting | 1 (Primary) | `model_xgb.yaml` | ⚠️ Segfault |
| 4 | **Linear** | Linear Regression | 4 (Baseline) | `model_linear.yaml` | ❌ Categorical Error |
| 5 | **Neural Network** | Deep Learning | 5 (Experimental) | `model_nn.yaml` | ✅ Working |
| 6 | **Historical Curve** | KNN-based | 4 (Baseline) | N/A | ✅ Working |
| 7 | **Global Mean** | Baseline | 4 (Baseline) | N/A | ❌ Missing Feature |
| 8 | **Flat** | Baseline | 4 (Baseline) | N/A | ✅ Working |
| 9 | **Trend** | Baseline | 4 (Baseline) | N/A | ✅ Working |
| 10 | **Hybrid Physics+ML** | Physics-informed | 2 (Secondary) | `model_hybrid.yaml` | ❌ API Mismatch |
| 11 | **ARIHOW** | Time Series | 4 (Specialized) | `model_arihow.yaml` | ❌ API Mismatch |
| 12 | **LSTM** | Deep Learning | 5 (Ablation) | `model_lstm.yaml` | Untested |
| 13 | **CNN-LSTM** | Deep Learning | 4 (Experimental) | `model_cnn_lstm.yaml` | Untested |
| 14 | **KG-GCN-LSTM** | Graph Neural Net | 4 (Experimental) | `model_kg_gcn_lstm.yaml` | Untested |

---

## Performance Comparison

### Validation Results (November 29, 2025)

| Model | Scenario | Official Metric | RMSE | MAE | Training Time |
|-------|----------|----------------|------|-----|---------------|
| **CatBoost** | S1 | **0.7692** | 0.2488 | 0.1795 | 17.83s |
| **CatBoost** | S2 | **0.2742** | 0.2055 | 0.1265 | 11.35s |
| Neural Network | S1 | 0.9648 | 0.2999 | 0.2329 | 125.47s |
| Neural Network | S2 | 0.5815 | 0.2884 | 0.2015 | 235.59s |
| Historical Curve | S1 | 0.9165 | 0.2981 | 0.2375 | 0.13s |
| Historical Curve | S2 | 1.0840 | 0.2997 | 0.2371 | 0.12s |
| Flat Baseline | S1 | 1.8007 | 0.4965 | 0.4094 | 0.00s |
| Flat Baseline | S2 | 2.1748 | 0.5363 | 0.4559 | 0.00s |
| Trend Baseline | S1 | 1.8007 | 0.4965 | 0.4094 | 0.00s |
| Trend Baseline | S2 | 2.1748 | 0.5363 | 0.4559 | 0.00s |

**Note:** Lower Official Metric is better (PE = Prediction Error). Best performing model is CatBoost.

---

## Detailed Hyperparameters by Model

### 1. CatBoost (`model_cat.yaml`)

**Core Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `iterations` | 3000 | 100-10000 | Number of boosting rounds |
| `learning_rate` | 0.03 | 0.01-0.1 | Step size shrinkage |
| `depth` | 6 | 4-10 | Tree depth |
| `l2_leaf_reg` | 3.0 | 0.1-10 | L2 regularization |
| `random_strength` | 1.0 | 0-5 | Random strength for splits |
| `bagging_temperature` | 1.0 | 0-2 | Bayesian bootstrap temperature |
| `min_data_in_leaf` | 20 | 1-100 | Minimum samples in leaf |
| `early_stopping_rounds` | 100 | 10-200 | Patience for early stopping |

**Sweep Configurations:**
- `default`: depth=6, lr=0.03, l2=3.0
- `shallow`: depth=4, lr=0.05, l2=1.0
- `conservative`: depth=5, lr=0.02, l2=10.0
- `s1_best`: depth=6, lr=0.03, l2=3.0
- `s2_best`: depth=6, lr=0.03, l2=3.0

**Categorical Features:** `ther_area`, `main_package`, `time_bucket`, `hospital_rate_bin`, `n_gxs_bin`

---

### 2. LightGBM (`model_lgbm.yaml`)

**Core Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `boosting_type` | gbdt | gbdt, dart, goss | Boosting algorithm |
| `num_leaves` | 63 | 15-255 | Max leaves per tree |
| `max_depth` | -1 | -1, 4-15 | Tree depth (-1=unlimited) |
| `learning_rate` | 0.05 | 0.01-0.1 | Step size shrinkage |
| `n_estimators` | 3000 | 100-10000 | Number of trees |
| `min_data_in_leaf` | 20 | 5-100 | Minimum samples in leaf |
| `lambda_l1` | 0.0 | 0-10 | L1 regularization |
| `lambda_l2` | 0.0 | 0-10 | L2 regularization |
| `feature_fraction` | 0.8 | 0.5-1.0 | Feature subsampling |
| `bagging_fraction` | 0.8 | 0.5-1.0 | Row subsampling |
| `bagging_freq` | 5 | 0-10 | Bagging frequency |
| `cat_smooth` | 10 | 1-100 | Categorical smoothing |
| `early_stopping_rounds` | 50 | 10-100 | Early stopping patience |

**Sweep Configurations:**
- `default`: num_leaves=63, lr=0.05, min_data=20
- `low_lr`: num_leaves=63, lr=0.02, n_estimators=5000
- `high_leaves`: num_leaves=127, lr=0.03, min_data=10
- `conservative`: num_leaves=31, lr=0.03, min_data=40
- `regularized`: num_leaves=63, lr=0.05, l1=0.1, l2=0.1
- `s1_best`: num_leaves=63, lr=0.05
- `s2_best`: num_leaves=31, lr=0.05

---

### 3. XGBoost (`model_xgb.yaml`)

**Core Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `booster` | gbtree | gbtree, gblinear, dart | Booster type |
| `max_depth` | 6 | 3-12 | Maximum tree depth |
| `learning_rate` | 0.03 | 0.01-0.1 | Eta/learning rate |
| `n_estimators` | 3000 | 100-10000 | Number of trees |
| `min_child_weight` | 1 | 1-10 | Min sum of instance weight |
| `gamma` | 0 | 0-5 | Min loss reduction for split |
| `reg_alpha` | 0 | 0-10 | L1 regularization |
| `reg_lambda` | 1 | 0-10 | L2 regularization |
| `subsample` | 0.8 | 0.5-1.0 | Row subsampling |
| `colsample_bytree` | 0.8 | 0.5-1.0 | Column subsampling per tree |
| `colsample_bylevel` | 1.0 | 0.5-1.0 | Column subsampling per level |
| `early_stopping_rounds` | 50 | 10-100 | Early stopping patience |

**Sweep Configurations:**
- `default`: max_depth=6, lr=0.03, lambda=1
- `low_lr`: max_depth=6, lr=0.02, n_estimators=5000
- `shallow`: max_depth=4, lr=0.05, lambda=3
- `deep`: max_depth=8, lr=0.02, lambda=5
- `regularized`: max_depth=5, lr=0.03, lambda=10, alpha=1
- `s1_best`: max_depth=6, lr=0.03, lambda=1
- `s2_best`: max_depth=4, lr=0.05, lambda=1

**Best Known Results:**
- Scenario 1: max_depth=6, lr=0.03 → Official: 0.7499
- Scenario 2: max_depth=4, lr=0.05 → Official: 0.2659

---

### 4. Linear Model (`model_linear.yaml`)

**Model Types:**
| Type | Algorithm | Key Parameters |
|------|-----------|----------------|
| `ridge` | Ridge Regression | `alpha` (L2 regularization) |
| `lasso` | Lasso Regression | `alpha` (L1 regularization) |
| `elasticnet` | ElasticNet | `alpha`, `l1_ratio` |
| `huber` | Huber Regressor | `epsilon`, `alpha` |

**Core Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `alpha` | 1.0 | 0.001-1000 | Regularization strength |
| `l1_ratio` | 0.5 | 0-1 | ElasticNet mixing (1=L1, 0=L2) |
| `epsilon` | 1.35 | 1-2 | Huber threshold |
| `fit_intercept` | true | - | Include intercept |
| `max_iter` | 1000 | 100-10000 | Max iterations |

**Sweep Configurations:**
- `default`: ridge, alpha=1.0
- `ridge_strong`: ridge, alpha=10.0
- `lasso`: lasso, alpha=1.0
- `lasso_strong`: lasso, alpha=10.0
- `elasticnet`: elasticnet, alpha=1.0, l1_ratio=0.5
- `huber`: huber, epsilon=1.35

---

### 5. Neural Network (`model_nn.yaml`)

**Architecture Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hidden_layers` | [256, 128, 64] | Various | Layer dimensions |
| `activation` | relu | relu, gelu, silu | Activation function |
| `dropout` | 0.2 | 0.0-0.5 | Dropout rate |
| `batch_norm` | true | - | Batch normalization |

**Training Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 100 | 10-500 | Maximum epochs |
| `batch_size` | 256 | 32-512 | Batch size |
| `learning_rate` | 0.001 | 0.0001-0.01 | Initial learning rate |
| `weight_decay` | 0.00001 | 0-0.001 | L2 regularization |
| `optimizer` | adam | adam, adamw, sgd | Optimizer |

**Scheduler Settings:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | reduce_on_plateau | LR scheduler type |
| `patience` | 10 | Scheduler patience |
| `factor` | 0.5 | LR reduction factor |
| `min_lr` | 0.000001 | Minimum learning rate |

**Early Stopping:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | 20 | Early stopping patience |
| `min_delta` | 0.000001 | Minimum improvement |
| `restore_best_weights` | true | Use best weights |

**Sweep Configurations:**
- `default`: layers=[256,128,64], lr=0.001, dropout=0.2
- `small`: layers=[128,64], lr=0.001, dropout=0.1
- `large`: layers=[512,256,128,64], lr=0.0005, dropout=0.3
- `deep`: layers=[256,256,128,128,64], lr=0.0001, dropout=0.2

---

### 6. Hybrid Physics+ML (`model_hybrid.yaml`)

**Physics Baseline Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `decay_rate` | 0.05 | 0.03-0.10 | Exponential decay rate |

**ML Residual Model (LightGBM):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_leaves` | 31 | Tree complexity |
| `learning_rate` | 0.05 | Learning rate |
| `n_estimators` | 500 | Number of trees |
| `min_data_in_leaf` | 20 | Min samples per leaf |

**Prediction Settings:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_min` | 0.0 | Minimum prediction |
| `clip_max` | 2.0 | Maximum prediction |

**Sweep Configurations:**
- `default`: decay=0.05, lr=0.05, leaves=31
- `slow_decay`: decay=0.03, lr=0.05, leaves=63
- `fast_decay`: decay=0.10, lr=0.03, leaves=31
- `ml_heavy`: decay=0.05, lr=0.07, leaves=63

---

### 7. ARIHOW (`model_arihow.yaml`)

**ARIMA Component:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | [1, 1, 1] | (p, d, q) ARIMA order |
| `seasonal_order` | [1, 0, 1, 12] | (P, D, Q, s) seasonal order |
| `trend` | 'c' | Trend component |
| `enforce_stationarity` | false | Enforce stationarity |

**Holt-Winters Component:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend` | 'add' | Trend type (add/mul/null) |
| `seasonal` | null | Seasonal type |
| `seasonal_periods` | 12 | Seasonal periods |
| `damped_trend` | true | Damped trend |

**Weight Learning:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_beta` | 0.5 | Initial ARIMA weight |
| `method` | 'grid' | Optimization method |
| `val_fraction` | 0.2 | Validation split |

**Sweep Configurations:**
- `default`: order=[1,1,1], hw_trend='add', damped=true
- `arima_heavy`: order=[2,1,1], beta=0.7, hw_trend=null
- `hw_heavy`: order=[1,0,0], beta=0.3, hw_trend='add'
- `simple`: order=[1,1,0], hw_trend='add', no seasonal

---

### 8. LSTM (`model_lstm.yaml`)

**Architecture Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lstm_hidden_dim` | 128 | 32-256 | Hidden dimension |
| `lstm_num_layers` | 3 | 1-4 | Number of LSTM layers |
| `lstm_dropout` | 0.3 | 0.0-0.5 | Dropout rate |
| `lstm_bidirectional` | true | - | Bidirectional LSTM |
| `attention_enabled` | true | - | Temporal attention |
| `attention_dim` | 64 | 16-128 | Attention dimension |
| `lookback_window` | 12 | 6-24 | Input sequence length |

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0005 | Initial learning rate |
| `weight_decay` | 0.0001 | L2 regularization |
| `batch_size` | 64 | Batch size |
| `max_epochs` | 100 | Maximum epochs |
| `early_stopping_patience` | 15 | Patience |

**Sweep Configurations:**
- `default`: hidden=128, layers=3, bidirectional=true
- `small`: hidden=64, layers=1, bidirectional=false
- `large`: hidden=256, layers=4, bidirectional=true
- `attention`: hidden=128, layers=2, attention=true

---

### 9. CNN-LSTM (`model_cnn_lstm.yaml`)

**CNN Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `cnn_channels` | [32, 64] | Channel progression |
| `cnn_kernel_sizes` | [3, 3] | Kernel sizes |
| `cnn_pool_sizes` | [2, 2] | Pooling sizes |
| `cnn_dropout` | 0.2 | CNN dropout |

**LSTM Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lstm_hidden_dim` | 64 | Hidden dimension |
| `lstm_num_layers` | 2 | Number of layers |
| `lstm_dropout` | 0.2 | LSTM dropout |
| `lstm_bidirectional` | false | Bidirectional |

**Attention:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `attention_enabled` | true | Enable attention |
| `attention_dim` | 32 | Attention dimension |

**Sweep Configurations:**
- `default`: lstm_hidden=64, lr=0.001, dropout=0.2
- `small`: channels=[16,32], hidden=32, layers=1
- `large`: channels=[64,128], hidden=128, layers=3
- `bidirectional`: bidirectional=true, hidden=64

---

### 10. KG-GCN-LSTM (`model_kg_gcn_lstm.yaml`)

**GCN Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gcn_type` | gcn | gcn or gat |
| `gcn_hidden_dims` | [64, 32] | GCN layers |
| `gcn_dropout` | 0.2 | GCN dropout |
| `gcn_activation` | relu | Activation |
| `gcn_skip_connection` | false | Skip connections |
| `gat_heads` | 4 | GAT attention heads |

**LSTM Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lstm_hidden_dim` | 64 | Hidden dimension |
| `lstm_num_layers` | 2 | Number of layers |
| `lstm_dropout` | 0.2 | Dropout |

**Fusion Settings:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `fusion_type` | concat | concat/attention/gate |
| `fusion_hidden_dim` | 64 | Fusion layer size |

**Graph Construction:**
| Edge Type | Weight | Description |
|-----------|--------|-------------|
| `therapeutic_area` | 1.0 | Same therapeutic area |
| `manufacturer` | 0.5 | Same manufacturer |
| `package_form` | 0.3 | Same dosage form |

**Sweep Configurations:**
- `default`: gcn_type=gcn, hidden=[64,32], lstm=64
- `gat`: gcn_type=gat, heads=4, lstm=64
- `small`: hidden=[32,16], lstm=32, layers=1
- `deep_gcn`: hidden=[64,64,32,32], skip_connection=true

---

## Sweep Configurations

### Running Sweeps

```bash
# Single model sweep
python -m src.train --scenario 1 --model xgboost --sweep

# Sweep with cross-validation
python -m src.train --scenario 1 --model catboost --sweep-cv --n-folds 3

# All models sweep
python -m src.train --all-models --sweep

# Quick sweep (first 3 configs only)
python -m src.train --scenario 1 --model lightgbm --sweep --quick-sweep

# Specific configuration
python -m src.train --scenario 1 --model xgboost --config-id s1_best
```

### Sweep Modes

| Mode | Description | Usage |
|------|-------------|-------|
| `configs` | Iterate over named configs | Default mode |
| `grid` | Cartesian product of parameters | Full exploration |

### Selection Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| `official_metric` | Competition PE metric | **Primary** - use for selection |
| `rmse` | Root Mean Square Error | Early stopping only |

---

## Best Practices

### Model Selection Guidelines

1. **Start with XGBoost** - Highest priority, best official_metric
2. **Use CatBoost for categoricals** - Native categorical handling
3. **Add LightGBM for diversity** - Different tree construction
4. **Neural networks for patterns** - When sufficient data/time

### Hyperparameter Tuning Order

1. `learning_rate` - Most impactful
2. `max_depth` / `num_leaves` - Tree complexity
3. `n_estimators` + `early_stopping_rounds` - Use early stopping
4. Regularization (`lambda`, `alpha`) - Prevent overfitting
5. Subsampling (`subsample`, `colsample`) - Add randomness

### Recommended Settings by Scenario

**Scenario 1 (No post-entry actuals):**
- Higher regularization (less overfitting risk)
- Conservative tree depth (6)
- Focus on pre-entry features

**Scenario 2 (First 6 months available):**
- Can use early erosion features
- Slightly shallower trees (4-5)
- More data → can be more aggressive

### Ensemble Strategy

1. Train XGBoost + LightGBM separately
2. Optimize weights using validation set
3. Target 50/50 or optimize based on official_metric
4. Add CatBoost if improves ensemble

---

## Quick Reference Card

### CLI Commands

```bash
# Basic training
python -m src.train --scenario 1 --model catboost

# With cross-validation
python -m src.train --scenario 1 --model xgboost --cv --n-folds 5

# Hyperparameter optimization
python -m src.train --scenario 1 --model xgboost --hpo --hpo-trials 50

# Full pipeline
python -m src.train --full-pipeline --model catboost --parallel

# Ensemble training
python -m src.train --scenario 1 --ensemble
```

### Key Metrics

| Scenario | Metric | Key Weights |
|----------|--------|-------------|
| S1 (Metric 1) | PE(0-23) | 50% acc(0-5), 20% monthly, 20% acc(6-11), 10% acc(12-23) |
| S2 (Metric 2) | PE(6-23) | 50% acc(6-11), 30% acc(12-23), 20% monthly |

### Bucket Weights
- **Bucket 1** (high erosion, mean_erosion ≤ 0.25): **2x weight**
- **Bucket 2** (low erosion): **1x weight**

---

*This document is auto-generated based on config files and training runs.*
