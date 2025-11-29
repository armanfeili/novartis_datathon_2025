# Sweep Results: 2025-11-29_06-13_xgboost_scenario2

- **Scenario**: 2
- **Model**: xgboost
- **Total Runs**: 4
- **Sweep Axes**: ['params.max_depth', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                               | axes                                                  |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:-----------------------------------------------------|:------------------------------------------------------|--------:|--------:|:-----------|
|            0.2659 |    0.206504 |   0.127689 |          2 | xgboost      |              2.84497 |             28116 |            7038 |          101 | 2025-11-29_06-13_xgboost_scenario2_lr0p05_max_depth4 | {'params.max_depth': 4, 'params.learning_rate': 0.05} |       1 |       4 | True       |
|            0.2716 |    0.207042 |   0.127566 |          2 | xgboost      |              4.92764 |             28116 |            7038 |          101 | 2025-11-29_06-13_xgboost_scenario2_lr0p03_max_depth4 | {'params.max_depth': 4, 'params.learning_rate': 0.03} |       0 |       4 | True       |
|            0.2914 |    0.209812 |   0.12893  |          2 | xgboost      |              6.49258 |             28116 |            7038 |          101 | 2025-11-29_06-13_xgboost_scenario2_lr0p03_max_depth6 | {'params.max_depth': 6, 'params.learning_rate': 0.03} |       2 |       4 | True       |
|            0.2993 |    0.209955 |   0.12935  |          2 | xgboost      |              4.99857 |             28116 |            7038 |          101 | 2025-11-29_06-13_xgboost_scenario2_lr0p05_max_depth6 | {'params.max_depth': 6, 'params.learning_rate': 0.05} |       3 |       4 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-13_xgboost_scenario2_lr0p05_max_depth4
- **Official Metric**: 0.2659
- **RMSE**: 0.2065
