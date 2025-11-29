# Sweep Results: 2025-11-29_06-11_lightgbm_scenario2

- **Scenario**: 2
- **Model**: lightgbm
- **Total Runs**: 4
- **Sweep Axes**: ['params.num_leaves', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                          | axes                                                    |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:------------------------------------------------|:--------------------------------------------------------|--------:|--------:|:-----------|
|            0.289  |    0.209433 |   0.129056 |          2 | lightgbm     |              3.50542 |             28116 |            7038 |          101 | 2025-11-29_06-11_lightgbm_scenario2_lr0p03_nl31 | {'params.num_leaves': 31, 'params.learning_rate': 0.03} |       0 |       4 | True       |
|            0.2941 |    0.209533 |   0.129323 |          2 | lightgbm     |              2.79316 |             28116 |            7038 |          101 | 2025-11-29_06-11_lightgbm_scenario2_lr0p05_nl31 | {'params.num_leaves': 31, 'params.learning_rate': 0.05} |       1 |       4 | True       |
|            0.2969 |    0.211092 |   0.129293 |          2 | lightgbm     |              4.52116 |             28116 |            7038 |          101 | 2025-11-29_06-11_lightgbm_scenario2_lr0p03_nl63 | {'params.num_leaves': 63, 'params.learning_rate': 0.03} |       2 |       4 | True       |
|            0.3019 |    0.211126 |   0.128618 |          2 | lightgbm     |              4.36796 |             28116 |            7038 |          101 | 2025-11-29_06-11_lightgbm_scenario2_lr0p05_nl63 | {'params.num_leaves': 63, 'params.learning_rate': 0.05} |       3 |       4 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-11_lightgbm_scenario2_lr0p03_nl31
- **Official Metric**: 0.2890
- **RMSE**: 0.2094
