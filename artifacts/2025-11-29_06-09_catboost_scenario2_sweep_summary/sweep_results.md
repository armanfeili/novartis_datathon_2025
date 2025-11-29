# Sweep Results: 2025-11-29_06-09_catboost_scenario2

- **Scenario**: 2
- **Model**: catboost
- **Total Runs**: 6
- **Sweep Axes**: ['params.depth', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                            | axes                                              |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:--------------------------------------------------|:--------------------------------------------------|--------:|--------:|:-----------|
|            0.2762 |    0.205616 |   0.12666  |          2 | catboost     |              8.88644 |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth6_lr0p03 | {'params.depth': 6, 'params.learning_rate': 0.03} |       3 |       6 | True       |
|            0.2791 |    0.207143 |   0.128493 |          2 | catboost     |              5.60767 |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth4_lr0p03 | {'params.depth': 4, 'params.learning_rate': 0.03} |       1 |       6 | True       |
|            0.2851 |    0.204591 |   0.126326 |          2 | catboost     |             17.086   |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth8_lr0p03 | {'params.depth': 8, 'params.learning_rate': 0.03} |       5 |       6 | True       |
|            0.2959 |    0.207966 |   0.1298   |          2 | catboost     |             10.0058  |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth6_lr0p01 | {'params.depth': 6, 'params.learning_rate': 0.01} |       2 |       6 | True       |
|            0.2973 |    0.206765 |   0.128518 |          2 | catboost     |             21.6745  |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth8_lr0p01 | {'params.depth': 8, 'params.learning_rate': 0.01} |       4 |       6 | True       |
|            0.3059 |    0.210309 |   0.132436 |          2 | catboost     |              5.31572 |             28116 |            7038 |          101 | 2025-11-29_06-09_catboost_scenario2_depth4_lr0p01 | {'params.depth': 4, 'params.learning_rate': 0.01} |       0 |       6 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-09_catboost_scenario2_depth6_lr0p03
- **Official Metric**: 0.2762
- **RMSE**: 0.2056
