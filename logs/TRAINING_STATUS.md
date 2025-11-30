# Training Status - Bonus Features

## Current Status

**Training Script:** `train_and_submit_complete.py`
**Config:** `configs/run_bonus_all.yaml`
**Started:** Check latest log file timestamp

## Enabled Bonus Features

1. ✅ **B2: Bucket Specialization** - Separate models for buckets 1 and 2
2. ✅ **B3: Post-hoc Calibration** - Linear calibration per scenario/bucket/window
3. ✅ **B4: Temporal Smoothing** - Rolling median smoothing
4. ✅ **B6: Bias Correction** - Group-level corrections (ther_area, country)
5. ✅ **B8: Multi-Seed Training** - Trains with seeds [42, 2025, 1337], selects best
6. ✅ **B10: Target Transform** - log1p transform for skewed targets
7. ✅ **G6: Data Augmentation** - Volume jitter and random month dropping

## Fixes Applied

1. ✅ Fixed `json` import issue in train.py
2. ✅ Fixed `epsilon` type conversion in transform_target function
3. ✅ Fixed categorical feature detection in CatBoost model
4. ✅ Fixed feature engineering column mismatch issue

## Training Process

The script runs:
1. **Scenario 1 Training** (~10-30 minutes)
   - Trains with multi-seed (selects best)
   - Applies bucket specialization if enabled
   - Fits calibration and bias corrections
   - Saves all artifacts

2. **Scenario 2 Training** (~10-30 minutes)
   - Same process for scenario 2

3. **Submission Generation** (~2-5 minutes)
   - Loads both models
   - Applies all bonus corrections
   - Generates final submission

## Monitoring

Check progress:
```bash
# View latest log
tail -f logs/full_pipeline_fixed_*.log

# Check if training is running
ps aux | grep train_and_submit

# Check artifacts
ls -lt artifacts/catboost_bonus_all_*/
```

## Expected Outputs

- **Artifacts:** `artifacts/catboost_bonus_all_*_s1/` and `*_s2/`
- **Submission:** `submissions/catboost_bonus_all_*_submission.csv`
- **Results:** `logs/results_*.json`
- **Logs:** `logs/full_pipeline_fixed_*.log`

## Next Steps After Training

1. Check results JSON for metrics
2. Verify submission file format
3. Compare metrics with baseline
4. Enable/disable features based on performance

