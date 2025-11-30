# CV Sweep Results: 2025-11-29_10-39_xgboost_scenario1_cv3

- **Scenario**: 1
- **Model**: xgboost
- **Configurations**: 9
- **CV Folds**: 3
- **Sweep Axes**: ['params.max_depth', 'params.learning_rate']

## Results (sorted by mean official_metric)

|   config_idx |   mean_official_metric |   std_official_metric |   ci_lower |   ci_upper |   mean_rmse |   mean_mae |   max_depth |   learning_rate |
|-------------:|-----------------------:|----------------------:|-----------:|-----------:|------------:|-----------:|------------:|----------------:|
|            5 |               0.741    |             0         |   0.741    |   0.741    |    0.233141 |   0.176591 |           5 |            0.03 |
|            3 |               0.785    |             0.0526334 |   0.654251 |   0.915749 |    0.240553 |   0.1815   |           4 |            0.05 |
|            4 |               0.785967 |             0.0439878 |   0.676695 |   0.895238 |    0.239876 |   0.181167 |           5 |            0.02 |
|            1 |               0.7881   |             0.0412564 |   0.685613 |   0.890587 |    0.240813 |   0.182682 |           4 |            0.02 |
|            2 |               0.7898   |             0.0454436 |   0.676912 |   0.902688 |    0.24084  |   0.182316 |           4 |            0.03 |
|            6 |             nan        |           nan         | nan        | nan        |  nan        | nan        |           5 |            0.05 |
|            7 |             nan        |           nan         | nan        | nan        |  nan        | nan        |           6 |            0.02 |
|            8 |             nan        |           nan         | nan        | nan        |  nan        | nan        |           6 |            0.03 |
|            9 |             nan        |           nan         | nan        | nan        |  nan        | nan        |           6 |            0.05 |

## Best Configuration

- **Config**: {'params.max_depth': 5, 'params.learning_rate': 0.03}
- **Mean Official Metric**: 0.7410
- **Std Official Metric**: 0.0000
