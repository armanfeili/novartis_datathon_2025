# Sweep Results: 2025-11-29_06-12_xgboost_scenario1

- **Scenario**: 1
- **Model**: xgboost
- **Total Runs**: 4
- **Sweep Axes**: ['params.max_depth', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                               | axes                                                  |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:-----------------------------------------------------|:------------------------------------------------------|--------:|--------:|:-----------|
|            0.7499 |    0.24591  |   0.175939 |          1 | xgboost      |              9.29669 |             37488 |            9384 |           86 | 2025-11-29_06-12_xgboost_scenario1_lr0p03_max_depth6 | {'params.max_depth': 6, 'params.learning_rate': 0.03} |       2 |       4 | True       |
|            0.7754 |    0.248032 |   0.176758 |          1 | xgboost      |              5.12427 |             37488 |            9384 |           86 | 2025-11-29_06-12_xgboost_scenario1_lr0p05_max_depth6 | {'params.max_depth': 6, 'params.learning_rate': 0.05} |       3 |       4 | True       |
|            0.785  |    0.249318 |   0.178433 |          1 | xgboost      |              6.64087 |             37488 |            9384 |           86 | 2025-11-29_06-12_xgboost_scenario1_lr0p05_max_depth4 | {'params.max_depth': 4, 'params.learning_rate': 0.05} |       1 |       4 | True       |
|            0.7873 |    0.250124 |   0.180013 |          1 | xgboost      |              4.45777 |             37488 |            9384 |           86 | 2025-11-29_06-12_xgboost_scenario1_lr0p03_max_depth4 | {'params.max_depth': 4, 'params.learning_rate': 0.03} |       0 |       4 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-12_xgboost_scenario1_lr0p03_max_depth6
- **Official Metric**: 0.7499
- **RMSE**: 0.2459
