# Sweep Results: 2025-11-29_06-07_catboost_scenario1

- **Scenario**: 1
- **Model**: catboost
- **Total Runs**: 6
- **Sweep Axes**: ['params.depth', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                            | axes                                              |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:--------------------------------------------------|:--------------------------------------------------|--------:|--------:|:-----------|
|            0.7778 |    0.24934  |   0.180305 |          1 | catboost     |             10.3583  |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth6_lr0p03 | {'params.depth': 6, 'params.learning_rate': 0.03} |       3 |       6 | True       |
|            0.78   |    0.249056 |   0.180024 |          1 | catboost     |             13.2496  |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth8_lr0p03 | {'params.depth': 8, 'params.learning_rate': 0.03} |       5 |       6 | True       |
|            0.7961 |    0.250597 |   0.182119 |          1 | catboost     |             23.1975  |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth8_lr0p01 | {'params.depth': 8, 'params.learning_rate': 0.01} |       4 |       6 | True       |
|            0.8034 |    0.250719 |   0.182201 |          1 | catboost     |              8.60514 |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth4_lr0p03 | {'params.depth': 4, 'params.learning_rate': 0.03} |       1 |       6 | True       |
|            0.8055 |    0.252242 |   0.184409 |          1 | catboost     |             10.1777  |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth6_lr0p01 | {'params.depth': 6, 'params.learning_rate': 0.01} |       2 |       6 | True       |
|            0.8351 |    0.255379 |   0.188241 |          1 | catboost     |              6.38911 |             37488 |            9384 |           86 | 2025-11-29_06-07_catboost_scenario1_depth4_lr0p01 | {'params.depth': 4, 'params.learning_rate': 0.01} |       0 |       6 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-07_catboost_scenario1_depth6_lr0p03
- **Official Metric**: 0.7778
- **RMSE**: 0.2493
