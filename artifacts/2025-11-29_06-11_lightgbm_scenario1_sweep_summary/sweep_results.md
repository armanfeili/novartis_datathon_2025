# Sweep Results: 2025-11-29_06-11_lightgbm_scenario1

- **Scenario**: 1
- **Model**: lightgbm
- **Total Runs**: 4
- **Sweep Axes**: ['params.num_leaves', 'params.learning_rate']

## Results (sorted by official_metric)

|   official_metric |   rmse_norm |   mae_norm |   scenario | model_type   |   train_time_seconds |   n_train_samples |   n_val_samples |   n_features | run_id                                          | axes                                                    |   index |   total | is_sweep   |
|------------------:|------------:|-----------:|-----------:|:-------------|---------------------:|------------------:|----------------:|-------------:|:------------------------------------------------|:--------------------------------------------------------|--------:|--------:|:-----------|
|            0.7526 |    0.248937 |   0.177576 |          1 | lightgbm     |              3.49103 |             37488 |            9384 |           86 | 2025-11-29_06-11_lightgbm_scenario1_lr0p05_nl31 | {'params.num_leaves': 31, 'params.learning_rate': 0.05} |       1 |       4 | True       |
|            0.7597 |    0.246639 |   0.177275 |          1 | lightgbm     |              4.00199 |             37488 |            9384 |           86 | 2025-11-29_06-11_lightgbm_scenario1_lr0p03_nl31 | {'params.num_leaves': 31, 'params.learning_rate': 0.03} |       0 |       4 | True       |
|            0.7611 |    0.245938 |   0.175321 |          1 | lightgbm     |              6.72015 |             37488 |            9384 |           86 | 2025-11-29_06-11_lightgbm_scenario1_lr0p03_nl63 | {'params.num_leaves': 63, 'params.learning_rate': 0.03} |       2 |       4 | True       |
|            0.7698 |    0.248005 |   0.17707  |          1 | lightgbm     |              4.3113  |             37488 |            9384 |           86 | 2025-11-29_06-11_lightgbm_scenario1_lr0p05_nl63 | {'params.num_leaves': 63, 'params.learning_rate': 0.05} |       3 |       4 | True       |

## Best Run

- **Run ID**: 2025-11-29_06-11_lightgbm_scenario1_lr0p05_nl31
- **Official Metric**: 0.7526
- **RMSE**: 0.2489
