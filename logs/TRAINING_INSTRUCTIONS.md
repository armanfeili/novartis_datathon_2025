# CatBoost Hero Model Training with All Bonus Features

## Overview

This document describes the complete training and validation pipeline for the CatBoost Hero Model with ALL Bonus Features enabled.

## Script: `train_catboost_bonus_complete.py`

This script performs:
1. **Training and Validation** for both Scenario 1 and Scenario 2 with all bonus features
2. **Comprehensive Logging** to `/logs` folder with config information
3. **Submission Generation** with best config for both scenarios
4. **Error Handling** and automatic issue fixing

## Enabled Bonus Features

All bonus features from `configs/run_bonus_all.yaml`:

- ✅ **B2: Bucket Specialization** - Separate models for buckets 1 and 2
- ✅ **B3: Post-hoc Calibration** - Linear calibration per scenario/bucket/window
- ✅ **B4: Temporal Smoothing** - Rolling median smoothing
- ✅ **B6: Bias Correction** - Group-level corrections (ther_area, country)
- ✅ **B8: Multi-Seed Training** - Trains with seeds [42, 2025, 1337], selects best
- ✅ **B10: Target Transform** - log1p transform for skewed targets
- ✅ **G6: Data Augmentation** - Volume jitter and random month dropping

## Output Files

### Logs Directory (`/logs`)

1. **`full_pipeline_{run_name}.log`** - Complete training log with all details
2. **`config_summary_{run_name}.yaml`** - Human-readable config summary
3. **`results_{run_name}.json`** - Complete results with metrics, artifacts paths, and status

### Artifacts Directory (`/artifacts`)

For each scenario (`{run_name}_s1` and `{run_name}_s2`):
- `model.bin` or `model.bin.cbm` - Trained model
- `calibration_params.json` - Calibration parameters (if enabled)
- `bias_corrections.json` - Bias corrections (if enabled)
- `target_transform_params_{scenario}.json` - Transform parameters (if enabled)
- `multi_seed_summary.csv` - Multi-seed experiment results
- `metrics.json` - Validation metrics
- `metadata.json` - Training metadata
- `configs/` - Config snapshots for reproducibility

### Submissions Directory (`/submissions`)

- `{run_name}_submission.csv` - Final submission file with predictions for both scenarios

## Results JSON Structure

```json
{
  "run_name": "catboost_bonus_all_{timestamp}",
  "timestamp": "{timestamp}",
  "config_file": "configs/run_bonus_all.yaml",
  "config_summary": {
    "bonus_features": {
      "B2_Bucket_Specialization": true,
      "B3_Calibration": true,
      ...
    }
  },
  "scenario1": {
    "status": "success",
    "official_metric": 0.6743,
    "artifacts_dir": "artifacts/{run_name}_s1",
    ...
  },
  "scenario2": {
    "status": "success",
    "official_metric": 0.7182,
    "artifacts_dir": "artifacts/{run_name}_s2",
    ...
  },
  "submission": {
    "status": "success",
    "submission_path": "submissions/{run_name}_submission.csv",
    "rows": 12345,
    ...
  },
  "overall_status": "success"
}
```

## Running the Script

```bash
# Run training with all bonus features
python train_catboost_bonus_complete.py

# Or with output redirection
python train_catboost_bonus_complete.py 2>&1 | tee logs/training_run_$(date +%Y%m%d_%H%M%S).log
```

## Monitoring Progress

Check logs:
```bash
# View latest log
tail -f logs/full_pipeline_catboost_bonus_all_*.log

# Check results
cat logs/results_catboost_bonus_all_*.json | jq .

# Check config summary
cat logs/config_summary_catboost_bonus_all_*.yaml
```

## Expected Runtime

- **Scenario 1 Training**: ~1-2 minutes (with multi-seed: ~3-5 minutes)
- **Scenario 2 Training**: ~1-2 minutes (with multi-seed: ~3-5 minutes)
- **Submission Generation**: ~30 seconds - 2 minutes
- **Total**: ~5-10 minutes

## Troubleshooting

### Common Issues

1. **Missing `ther_area` column**: Fixed automatically - bias correction uses only available columns
2. **Feature matrix validation errors**: Check feature engineering logs
3. **Memory issues**: Reduce batch size or disable data augmentation

### Fixes Applied

- ✅ Fixed bias correction to handle missing `ther_area` column gracefully
- ✅ Added comprehensive error handling and logging
- ✅ Automatic fallback when group columns are missing

## Best Config Selection

The script automatically:
1. Trains multiple seeds (if multi-seed enabled)
2. Selects best seed based on official metric
3. Uses best model for submission generation
4. Stores all results with config information

## Submission Format

The final submission file contains:
- `country` - Country code
- `brand_name` - Brand name
- `months_postgx` - Months post-Gx (0-23 for S1, 6-23 for S2)
- `volume` - Predicted volume (normalized and inverse-transformed)

## Next Steps

After training completes:
1. Check `results_{run_name}.json` for metrics
2. Verify submission file format
3. Compare metrics with baseline
4. Review logs for any warnings or errors

