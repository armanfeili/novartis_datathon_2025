# Submissions Directory

This directory stores submission files for the Novartis Datathon 2025.

## File Naming Convention

Submissions should follow the pattern:
```
submission_YYYYMMDD_HHMMSS_<model>_<scenario>.csv
```

Example:
```
submission_20250115_143022_catboost_combined.csv
```

## Submission Format

Each submission file must have exactly 4 columns:

| Column | Type | Description |
|--------|------|-------------|
| country | string | Country code (e.g., "CA", "ES") |
| brand_name | string | Brand/drug identifier |
| month | string | Month in YYYY-MM format |
| volume | float | Predicted volume (absolute, not normalized) |

## Required Rows

- **Scenario 1 series**: 228 series × 24 months = 5,472 rows
- **Scenario 2 series**: 112 series × 18 months = 2,016 rows  
- **Total**: 7,488 rows

## Validation

Before submitting, validate your file:

```python
from src.inference import validate_submission_format
import pandas as pd

submission = pd.read_csv('submissions/my_submission.csv')
is_valid, errors = validate_submission_format(submission)

if not is_valid:
    print("Errors:", errors)
```

## Auxiliary File

The auxiliary file (`submission_auxiliar.csv`) is used for bucket classification:

| Column | Type | Description |
|--------|------|-------------|
| country | string | Country code |
| brand_name | string | Brand identifier |
| mean_erosion | float | Mean normalized volume (0-23 for S1, 6-23 for S2) |
| bucket | int | 1 = high erosion (≤0.25), 2 = medium/low (>0.25) |

## Notes

- Volume values must be **non-negative**
- Volume is computed from: `volume = y_norm_pred * avg_vol_12m`
- Keep all submission files for reproducibility
- The official scoring uses `docs/guide/metric_calculation.py`
