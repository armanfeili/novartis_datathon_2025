- dataset (train)
  - df_generics_train.csv:
  - df_medicine_info_train.csv:
  - df_volume_train.csv:

- dataset (test)
  - df_generics_train.csv:
  - df_medicine_info_train.csv:
  - df_volume_train.csv:

- auxiliar_metric_computation_example.csv
- metric_calculation.py
- submission_template.csv
- submission_example.csv

---

## 1. Big picture: how all these files fit together

At a high level:

* The **train datasets** (`df_volume_train`, `df_generics_train`, `df_medicine_info_train`) give you everything you need to:

  * understand pre- and post-generic-entry sales behavior,
  * build and validate forecasting models,
  * compute the auxiliary quantities used in the official metrics (`avg_vol`, `bucket`).

* The **test datasets** have the same structure, but:

  * contain only **known data** (pre-entry volumes and, depending on scenario, some post-entry actuals),
  * do **not** contain the “future” volumes you must predict (months_postgx 0–23 or 6–23).

* `auxiliar_metric_computation_example.csv` shows the structure of the **auxiliary table** you must compute yourself: one row per `(country, brand_name)` with:

  * `avg_vol` (average monthly volume in the 12 months before generic entry),
  * `bucket` (1 = high erosion, 2 = medium/low erosion).

* `submission_template.csv` is the **skeleton** of the file you must submit: it lists all `(country, brand_name, months_postgx)` combinations you must predict, with an empty `volume`.

* `submission_example.csv` is an example where the template is filled with dummy predictions (all zeros), just to show the correct structure.

* `metric_calculation.py` is the **official local metric script**:

  * It defines `compute_metric1` (Phase 1A: Scenario 1, “0 actuals”) and `compute_metric2` (Phase 1B: Scenario 2, “6 actuals”).
  * It explains how to merge actuals, predictions, and auxiliary data to get the final scores, with Bucket 1 double-weighted.

Now, let us go through each file in detail.

---

## 2. Training datasets

### 2.1 `df_volume_train.csv`

**Example rows you provided:**

```text
country,brand_name,month,months_postgx,volume
COUNTRY_B6AE,BRAND_1C1E,Jul,-24,272594.39214363444
COUNTRY_B6AE,BRAND_1C1E,Aug,-23,351859.31030592526
COUNTRY_B6AE,BRAND_1C1E,Sep,-22,447953.4812752315
COUNTRY_B6AE,BRAND_1C1E,Oct,-21,411543.2923598504
COUNTRY_B6AE,BRAND_1C1E,Nov,-20,774594.4542361738
COUNTRY_B6AE,BRAND_1C1E,Dec,-19,442279.1780362012
COUNTRY_B6AE,BRAND_1C1E,Jan,-18,485069.4858336451
COUNTRY_B6AE,BRAND_1C1E,Feb,-17,549902.6994992384
COUNTRY_B6AE,BRAND_1C1E,Mar,-16,751044.237822728
COUNTRY_B6AE,BRAND_1C1E,Apr,-15,181852.13103682228
```

**Columns:**

* `country`

  * Categorical ID of the **market** (e.g., `COUNTRY_B6AE`, `COUNTRY_0024`, etc.).
  * Used as a grouping key and a feature (if you encode it).

* `brand_name`

  * Categorical ID of the **branded drug** (e.g., `BRAND_1C1E`, `BRAND_DB48`).
  * Together with `country`, identifies one **time series**.

* `month`

  * Calendar month name (`Jul`, `Aug`, `Sep`, etc.) in this snippet.
  * In full data it may be a more precise date (e.g., with year), but for modeling you will typically use `months_postgx` as time index; `month` can be used for **seasonality** (month-of-year) if you reconstruct dates.

* `months_postgx`

  * Integer index of **time relative to generic entry (gx)** for that country–brand.
  * `0` = month when the **first generic** enters.
  * Negative values (e.g., `-24`, `-23`, …) = months **before** generic entry.
  * Positive values (e.g., `1`, `2`, …) = months **after** generic entry.
  * In train, for each `(country, brand_name)` you can have up to:

    * 24 months **pre-entry** (`-24` to `-1`),
    * 24 months **post-entry** (`0` to `23`).

* `volume`

  * Numeric: **number of units sold** (e.g., packs) in that month, in that country, for that brand.
  * This is the **target** you want to forecast for the 24 months after generic entry (with different observed horizons in Scenario 1 vs Scenario 2).

**Role in the Datathon:**

* This is your **main fact table** in train:

  * You will use pre-entry volumes (months_postgx < 0) to:

    * compute `avg_vol` (12 months before entry),
    * characterize pre-entry growth, maturity, volatility, etc.
  * You will use post-entry volumes (months_postgx ≥ 0) as **labels** to:

    * train models to predict monthly volume,
    * compute mean normalized erosion and bucket labels.

* You will also use it to build **train/validation splits**:

  * For Scenario 1: use pre-entry months as features, months 0–23 as labels.
  * For Scenario 2: use pre-entry + months 0–5 as features, months 6–23 as labels.

---

### 2.2 `df_generics_train.csv`

**Example rows:**

```text
country,brand_name,months_postgx,n_gxs
COUNTRY_B6AE,BRAND_1C1E,0,4.0
COUNTRY_B6AE,BRAND_1C1E,1,4.0
COUNTRY_B6AE,BRAND_1C1E,2,4.0
COUNTRY_B6AE,BRAND_1C1E,3,4.0
COUNTRY_B6AE,BRAND_1C1E,4,4.0
COUNTRY_B6AE,BRAND_1C1E,5,4.0
COUNTRY_B6AE,BRAND_1C1E,6,4.0
COUNTRY_B6AE,BRAND_1C1E,7,4.0
COUNTRY_B6AE,BRAND_1C1E,8,4.0
COUNTRY_B6AE,BRAND_1C1E,9,4.0
```

**Columns:**

* `country`, `brand_name`

  * Same meaning and keys as in `df_volume_train`: identify a **brand** in a **specific country**.

* `months_postgx`

  * Same relative time index as in `df_volume_train`.
  * Here you see `0, 1, 2, …` (post-entry months).
  * In general, this table focuses on the **post-entry period**.

* `n_gxs`

  * Numeric: **number of generics** present in the market for this `(country, brand_name)` at that `months_postgx`.
  * For example, in the snippet:

    * At month 0 (the entry month), there are already `4.0` generics.
    * It stays constant at 4.0 from month 0 to 9 in this example.
  * Over a longer horizon, you might see:

    * `0` at month 0 or early months,
    * then increasing values as more generics enter,
    * possibly decreasing if some generics exit the market (less common, but theoretically possible).

**Role in the Datathon:**

* This is your **competition landscape feature**:

  * It captures **how many generic competitors** exist at each point in time.
  * It is highly predictive of erosion: more generics → stronger price pressure → faster volume drop (typically).

* For **Scenario 1 (just after entry)**:

  * At prediction time (month 0), you may only know `n_gxs` up to month 0 (depending on rules: you must not use future `n_gxs` for months > 0 when predicting months 1–23).
  * In training, you can use the **historical trajectories** of `n_gxs` from previous brands to learn typical patterns.

* For **Scenario 2 (6 months after entry)**:

  * You can use `n_gxs` up to month 5 (or 6, depending on scenario definition), but must not leak beyond the current forecasting horizon.

* You join `df_generics_train` with `df_volume_train` on `(country, brand_name, months_postgx)` to have **volume + competitive intensity** per time step.

---

### 2.3 `df_medicine_info_train.csv`

**Example rows:**

```text
country,brand_name,ther_area,hospital_rate,main_package,biological,small_molecule
COUNTRY_0024,BRAND_1143,Sensory_organs,0.0885496976322991,EYE DROP,False,True
COUNTRY_0024,BRAND_1865,Muscoskeletal_Rheumatology_and_Osteology,92.35982629641548,INJECTION,False,False
COUNTRY_0024,BRAND_240F,Antineoplastic_and_immunology,36.94400678705606,PILL,False,True
COUNTRY_0024,BRAND_2F6C,Antineoplastic_and_immunology,0.0063544122168699,INJECTION,True,False
COUNTRY_0024,BRAND_3A67,Nervous_system,,PILL,False,False
COUNTRY_0024,BRAND_3CB9,Antineoplastic_and_immunology,1.421185523626305,PILL,False,True
COUNTRY_0024,BRAND_3E0C,Antineoplastic_and_immunology,47.06338742730865,INJECTION,True,False
COUNTRY_0024,BRAND_41B7,Nervous_system,0.0217042658208487,PILL,False,True
COUNTRY_0024,BRAND_467B,Antineoplastic_and_immunology,0.0061993306198961,INJECTION,True,False
COUNTRY_0024,BRAND_4920,Endocrinology_and_Metabolic_Disease,0.8956771122275429,Others,True,False
```

**Columns:**

* `country`, `brand_name`

  * Keys identifying the same entities as in the other tables.
  * **One row per (country, brand_name)**: these are **static features** (do not change with month).

* `ther_area` (`therapeutic_area`)

  * Categorical: the **therapeutic area** of the drug:

    * Examples: `Sensory_organs`, `Muscoskeletal_Rheumatology_and_Osteology`, `Antineoplastic_and_immunology`, `Nervous_system`, `Endocrinology_and_Metabolic_Disease`, `Anti-infectives`, etc.
  * These categories capture the disease area and often correlate with:

    * typical volume levels,
    * typical erosion patterns,
    * channel (hospital vs retail).

* `hospital_rate`

  * Numeric: **percentage of units delivered via hospitals** (0–100, although here it seems like a percentage as a numeric value).
  * Example: `92.3598` → ~92% of units delivered in the hospital channel.
  * Missing values are possible (see `BRAND_3A67` where it is empty).
  * This is important because hospital products can behave differently:

    * more central procurement,
    * different price/volume dynamics.

* `main_package`

  * Categorical: **main dosage form / package**:

    * Examples: `EYE DROP`, `INJECTION`, `PILL`, `Others`.
  * Package type can influence:

    * usage patterns,
    * pricing,
    * substitution by generics.

* `biological`

  * Boolean (stored as string `True`/`False` in CSV, or boolean when parsed):

    * `True` if the drug is a **biological** product (e.g., monoclonal antibodies, complex proteins).
    * Biologicals often have different generic dynamics (biosimilars) and may experience different erosion patterns.

* `small_molecule`

  * Boolean: `True` if the drug is a **small molecule** (classical chemically synthesized drug).
  * Typically, `biological` and `small_molecule` are complementary:

    * biological = True → small_molecule = False, and vice versa.

**Role in the Datathon:**

* This table gives **contextual/product features**:

  * `ther_area`, `biological`, `small_molecule` are good predictors of **typical erosion curves**, because different drug classes face different competitive and regulatory environments.

  * `hospital_rate` and `main_package` describe **channel and formulation**, which also relate to erosion speed.

* These features **do not change over time**, so they are safe to use in both Scenarios without time-leak issues.

* You join `df_medicine_info_train` with the time-series tables on `(country, brand_name)`.

---

## 3. Test datasets

You wrote:

* dataset (test)

  * `df_generics_train.csv`: (but this is clearly the **test** generics file; consider it `df_generics_test.csv`)
  * `df_medicine_info_train.csv`: (again, this is logically `df_medicine_info_test.csv`)
  * `df_volume_train.csv`: (this is logically `df_volume_test.csv`)

The **names in your snippet** still say `*_train.csv`, but semantically these are the **test** versions. Their structure is identical, but they are used differently.

### 3.1 `df_generics_test.csv` (named `df_generics_train.csv` in snippet)

**Example rows:**

```text
country,brand_name,months_postgx,n_gxs
COUNTRY_B6AE,BRAND_DF2E,0,0.0
COUNTRY_B6AE,BRAND_DF2E,1,0.0
COUNTRY_B6AE,BRAND_DF2E,2,1.0
COUNTRY_B6AE,BRAND_DF2E,3,2.0
COUNTRY_B6AE,BRAND_DF2E,4,2.0
COUNTRY_B6AE,BRAND_DF2E,5,2.0
COUNTRY_B6AE,BRAND_DF2E,6,2.0
COUNTRY_B6AE,BRAND_DF2E,7,2.0
COUNTRY_B6AE,BRAND_DF2E,8,2.0
COUNTRY_B6AE,BRAND_DF2E,9,2.0
```

**Same columns and meaning** as the train version (`country`, `brand_name`, `months_postgx`, `n_gxs`).

Differences:

* These rows refer to **test series** for which you must forecast volumes.
* At evaluation time, the organizer has **actual volumes** for these (held out); you only see:

  * known `n_gxs` values up to certain months (depending on scenario),
  * known volumes for pre-entry and, in Scenario 2, months 0–5.

In this example:

* At month 0: `n_gxs = 0.0` (no generics exactly at entry).
* At month 2: first generic appears (`n_gxs = 1.0`).
* At month 3: `n_gxs = 2.0`, and remains 2.0 until month 9, meaning stable competition.

You must **not** use future `n_gxs` beyond your forecasting horizon in a way that would cause leakage in your internal validation. For test, since you do not know the future volumes, you simply treat these as **features** at inference time.

---

### 3.2 `df_medicine_info_test.csv` (named `df_medicine_info_train.csv` in snippet)

**Example rows:**

```text
country,brand_name,ther_area,hospital_rate,main_package,biological,small_molecule
COUNTRY_0024,BRAND_31BE,Nervous_system,0.3126175786975748,PILL,False,True
COUNTRY_0024,BRAND_5165,Anti-infectives,4.338363148161993,PILL,False,True
COUNTRY_0024,BRAND_79B0,Nervous_system,0.7088264990781284,PILL,False,True
COUNTRY_0024,BRAND_8164,Cardiovascular_Metabolic,1.279747390758746,PILL,False,True
COUNTRY_01A1,BRAND_03DE,Muscoskeletal_Rheumatology_and_Osteology,1.4641586963119757,PILL,False,True
COUNTRY_01A1,BRAND_0D48,Anti-infectives,20.262343689570603,PILL,False,True
COUNTRY_01A1,BRAND_240F,Antineoplastic_and_immunology,99.88363636363572,PILL,False,True
COUNTRY_01A1,BRAND_7FD0,Nervous_system,71.89475133770271,PILL,False,True
COUNTRY_01A1,BRAND_9EFF,Antineoplastic_and_immunology,0.2403323215894833,INJECTION,False,True
COUNTRY_0309,BRAND_0721,Anti-infectives,18.48715679632601,PILL,False,True
```

Same columns and meaning as training `df_medicine_info_train.csv`:

* `ther_area`: therapeutic area.
* `hospital_rate`, `main_package`.
* `biological`, `small_molecule`.

This table provides static **product features** for the **test country–brand pairs**.

---

### 3.3 `df_volume_test.csv` (named `df_volume_train.csv` in snippet)

**Example rows:**

```text
country,brand_name,month,months_postgx,volume
COUNTRY_9891,BRAND_DB48,Oct,-24,231503.7589298541
COUNTRY_9891,BRAND_DB48,Nov,-23,203754.67542232448
COUNTRY_9891,BRAND_DB48,Dec,-22,224900.01912060223
COUNTRY_9891,BRAND_DB48,Jan,-21,252599.92705163336
COUNTRY_9891,BRAND_DB48,Feb,-20,229358.62272819405
COUNTRY_9891,BRAND_DB48,Mar,-19,230519.1046466163
COUNTRY_9891,BRAND_DB48,Apr,-18,243683.61396675056
COUNTRY_9891,BRAND_DB48,May,-17,215499.4637909811
COUNTRY_9891,BRAND_DB48,Jun,-16,209084.97205601243
COUNTRY_9891,BRAND_DB48,Jul,-15,246396.1003371468
```

Same columns as `df_volume_train.csv`. Differences:

* For the test series, you will typically see:

  * **Pre-entry volumes** (months_postgx < 0), like the snippet.
  * Possibly some **post-entry actuals** for **Scenario 2** (months_postgx 0–5), depending on how the test files are split per scenario.

* You will **not** see volumes for the horizon you must predict:

  * Scenario 1: months 0–23 are unknown and must be predicted.
  * Scenario 2: months 6–23 are unknown and must be predicted; months 0–5 are available to you.

---

## 4. `auxiliar_metric_computation_example.csv`

The snippet you gave at the end:

```text
country,brand_name,avg_vol,bucket
FAKE_COUNTRY1,FAKE_BRAND1,1000,1
FAKE_COUNTRY2,FAKE_BRAND2,1500,2
FAKE_COUNTRY3,FAKE_BRAND3,1200,1
FAKE_COUNTRY4,FAKE_BRAND4,900,2
FAKE_COUNTRY5,FAKE_BRAND5,2000,1
```

This is the **auxiliary file** the metric code expects as `df_aux`:

* One row per **(country, brand_name)** present in train/test.

* Columns:

  * `country`, `brand_name`

    * Keys to join with `df_actual` and `df_pred`.

  * `avg_vol`

    * The **average monthly volume** in the 12 months before generic entry for that country–brand.
    * Computation (according to the Datathon description):

      * Take volumes for months_postgx = -12, -11, …, -1.
      * Compute the arithmetic mean.
    * Used in metrics to **normalize errors**, so different brands with very different scales are comparable.

  * `bucket`

    * Integer (1 or 2):

      * `1` = **Bucket 1** = high-erosion drugs (mean normalized erosion between 0 and 0.25).
      * `2` = **Bucket 2** = medium/low erosion (mean > 0.25).
    * Bucket is used to:

      * Weight errors: Bucket 1 gets **double weight** in the final metric.
      * Allow separate evaluation by erosion severity.

**Usage in metric code:**

* `metric_calculation.py` merges this file into the combined actual/prediction DataFrame and uses:

  * `avg_vol` in `_compute_pe_phase1a` and `_compute_pe_phase1b`,
  * `bucket` for the final **bucket-weighted score**.

You must **compute this file yourself** based on the training volume data (the script comments mention that).

---

## 5. Submission files

### 5.1 `submission_template.csv`

**Snippet:**

```text
country,brand_name,months_postgx,volume
COUNTRY_9891,BRAND_3C69,0,
COUNTRY_9891,BRAND_3C69,1,
COUNTRY_9891,BRAND_3C69,2,
COUNTRY_9891,BRAND_3C69,3,
COUNTRY_9891,BRAND_3C69,4,
COUNTRY_9891,BRAND_3C69,5,
COUNTRY_9891,BRAND_3C69,6,
COUNTRY_9891,BRAND_3C69,7,
COUNTRY_9891,BRAND_3C69,8,
COUNTRY_9891,BRAND_3C69,9,
```

**Columns:**

* `country`

* `brand_name`

* `months_postgx`

  * In real competition, you will see months in the full needed range:

    * Scenario 1: 0–23
    * Scenario 2: 6–23
  * This snippet shows 0–9 just as an example.

* `volume`

  * Empty values: you must fill them with your predicted **monthly volume**.

**Role:**

* This file is the **exact structure** your submission must have:

  * same column names,
  * same set of rows,
  * `volume` filled with floats (predictions).

* The submission platform and metric script expect **these exact columns** and join on:

  * `country`, `brand_name`, `months_postgx`.

---

### 5.2 `submission_example.csv`

**Snippet:**

```text
country,brand_name,months_postgx,volume
COUNTRY_9891,BRAND_3C69,0,0.0
COUNTRY_9891,BRAND_3C69,1,0.0
COUNTRY_9891,BRAND_3C69,2,0.0
COUNTRY_9891,BRAND_3C69,3,0.0
COUNTRY_9891,BRAND_3C69,4,0.0
COUNTRY_9891,BRAND_3C69,5,0.0
COUNTRY_9891,BRAND_3C69,6,0.0
COUNTRY_9891,BRAND_3C69,7,0.0
COUNTRY_9891,BRAND_3C69,8,0.0
COUNTRY_9891,BRAND_3C69,9,0.0
```

* Same columns as the template, but the `volume` is now filled with `0.0` everywhere.

* This is just a **toy example** to show the expected format.

  * If you submitted this as-is, your metric would be very bad, but structurally it would be accepted.

* It is useful for:

  * Quick dry-run: checking that the submission platform accepts a file with this shape.
  * Understanding how your own predictions should be written.

---

## 6. `metric_calculation.py` in detail

This is the helper script to **compute the Datathon metrics locally**.

### 6.1 Imports and structure

```python
from pathlib import Path

import numpy as np
import pandas as pd
```

* Uses `pandas` DataFrames as the main data structure.
* Uses `numpy` for numeric checks (`np.isnan`).
* `Path` from `pathlib` to manage file paths.

It defines:

* Metric 1 (Phase 1-a) = Scenario 1 (“0 actuals”).
* Metric 2 (Phase 1-b) = Scenario 2 (“6 actuals”).

And an example `__main__` workflow.

---

### 6.2 Metric 1 (Phase 1-a) – `compute_metric1`

#### 6.2.1 `_compute_pe_phase1a(group)`

```python
def _compute_pe_phase1a(group: pd.DataFrame) -> float:
    """Compute PE for one (country, brand, bucket) group following the corrected Metric 1 formula."""
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan
```

* This function computes the **Prediction Error (PE)** for one specific combination `(country, brand_name, bucket)` in **Scenario 1**.

* It assumes `group` contains rows for that brand:

  * `volume_actual`
  * `volume_predict`
  * `months_postgx`
  * `avg_vol` (from `df_aux`)

* If `avg_vol` is `0` or NaN, it returns `NaN` (to avoid division by zero).

It then defines two helper functions:

```python
    def sum_abs_diff(month_start: int, month_end: int) -> float:
        """Sum of absolute differences sum(|actual - pred|)."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
    
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        """Absolute difference of |sum(actuals) - sum(pred)|."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        sum_actual = subset["volume_actual"].sum()
        sum_pred = subset["volume_predict"].sum()
        return abs(sum_actual - sum_pred)
```

* `sum_abs_diff(a, b)`:

  * For the interval `months_postgx ∈ [a, b]`:

    * Computes `∑ |actual - pred|` month by month.

* `abs_sum_diff(a, b)`:

  * For the same interval:

    * Computes `|∑ actual - ∑ pred|`.
    * This is a **cumulative** error in total volume over that period.

Then it calculates four terms:

```python
    term1 = 0.2 * sum_abs_diff(0, 23) / (24 * avg_vol)
    term2 = 0.5 * abs_sum_diff(0, 5) / (6 * avg_vol)
    term3 = 0.2 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term4 = 0.1 * abs_sum_diff(12, 23) / (12 * avg_vol)

    return term1 + term2 + term3 + term4
```

* **Term 1**:

  * 0.2 × (sum of monthly absolute errors from month 0 to 23) normalized by `24 * avg_vol`.
  * Gives a global view of monthly errors across the entire 24-month horizon.

* **Term 2**:

  * 0.5 × (absolute difference between cumulative actuals and cumulative preds from month 0–5) / `(6 * avg_vol)`.
  * This is heavily weighted (0.5) and focuses on the **first 6 months** post-entry (where erosion is strongest and most important).

* **Term 3**:

  * 0.2 × cumulative error over months 6–11.

* **Term 4**:

  * 0.1 × cumulative error over months 12–23.

So:

* The metric emphasizes **early months** (term2) and still considers mid and full horizon (terms 3 and 1+4), all **normalized by pre-entry average volume**.

#### 6.2.2 `_metric1(df_actual, df_pred, df_aux)`

```python
def _metric1(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    """Compute Metric 1 PE value.
    ...
    """
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
```

* Merges:

  * Actual volumes `df_actual` (the ground truth for Scenario 1) and
  * Predicted volumes `df_pred` (your submission-style data),
  * On keys: `("country", "brand_name", "months_postgx")`.
* Then merges with `df_aux` to add `avg_vol` and `bucket`.

Next:

```python
    merged["start_month"] = merged.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged = merged[merged["start_month"] == 0].copy()
```

* `start_month` = minimum `months_postgx` present in this merged dataset for each `(country, brand_name)`.
* It then keeps only series where `start_month == 0`, i.e. where the earliest month present is 0.

  * This effectively selects the **Scenario 1 subset**, where for each brand the series starts at month 0 (no post-entry actuals used).

Now the grouping and PE computation:

```python
    pe_results = (
        merged.groupby(["country", "brand_name", "bucket"])
        .apply(_compute_pe_phase1a)
        .reset_index(name="PE")
    )
```

* For each `(country, brand_name, bucket)` group:

  * Calls `_compute_pe_phase1a`,
  * Stores `PE`.

Then it separates buckets and computes final weighted score:

```python
    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]

    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()
```

* `n1` = number of unique brand–country pairs in Bucket 1.
* `n2` = same for Bucket 2.
* Final score =

  * (2 / n1) × sum of PE over Bucket 1 **plus**
  * (1 / n2) × sum of PE over Bucket 2.

Interpretation:

* Each bucket is **averaged** over its brands (division by n1 or n2).
* Bucket 1 contribution is multiplied by **2**, Bucket 2 by **1**.
* So Bucket 1 is **twice as important** in the final metric.

#### 6.2.3 `compute_metric1(...)`

```python
def compute_metric1(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame) -> float:
    ...
    return round(_metric1(df_actual, df_pred, df_aux), 4)
```

* Public API: you call this with:

  * `df_actual` = actual volumes (Scenario 1 subset),
  * `df_pred` = your predictions,
  * `df_aux` = avg_vol + bucket per brand.
* Returns the final Metric 1 value, **rounded to 4 decimals**.

---

### 6.3 Metric 2 (Phase 1-b) – `compute_metric2`

#### 6.3.1 `_compute_pe_phase1b(group)`

```python
def _compute_pe_phase1b(group: pd.DataFrame) -> float:
    """Compute PE for a specific country-brand-bucket group."""
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan
```

* Same structure as `_compute_pe_phase1a`, but now for **Scenario 2**.

Same helpers:

```python
    def sum_abs_diff(month_start: int, month_end: int) -> float:
        ...
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        ...
```

But the intervals change:

```python
    term1 = 0.2 * sum_abs_diff(6, 23) / (18 * avg_vol)
    term2 = 0.5 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term3 = 0.3 * abs_sum_diff(12, 23) / (12 * avg_vol)
    
    return term1 + term2 + term3
```

Here:

* Scenario 2 assumes you already know actuals for months 0–5, and you **predict from 6–23**.

* So:

  * Term 1: 0.2 × average monthly absolute error for months 6–23.
  * Term 2: 0.5 × cumulative error for months 6–11 (emphasized early post-entry horizon).
  * Term 3: 0.3 × cumulative error for months 12–23.

* Again, everything is normalized by `avg_vol`, so the metric is scale-free.

#### 6.3.2 `_metric2(df_actual, df_pred, df_aux)`

```python
def _metric2(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    ...
    merged_data = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")
```

* Same merging logic as `_metric1`.

Scenario 2 identification:

```python
    merged_data["start_month"] = merged_data.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged_data = merged_data[merged_data["start_month"] == 6].copy()
```

* For Scenario 2, the actual dataset used here is expected to contain rows from **month 6 onwards** for each brand in this scenario.
* `start_month == 6` ensures we are operating on series where the earliest month in this evaluation slice is 6 (we assume months 0–5 are known but not evaluated here).

Then grouping and PE:

```python
    pe_results = (
        merged_data.groupby(["country", "brand_name", "bucket"])
        .apply(_compute_pe_phase1b)
        .reset_index(name="PE")
    )

    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]
    
    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()
```

* Same weighting logic: bucket 1 double-weighted.

#### 6.3.3 `compute_metric2(...)`

```python
def compute_metric2(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    df_aux: pd.DataFrame) -> float:
    ...
    return round(_metric2(df_actual, df_pred, df_aux), 4)
```

* Public API for **Scenario 2**.

---

### 6.4 `__main__` block – example workflow

```python
if __name__ == "__main__":

    # Paths (adapt as needed)
    DATA_PATH = Path("data")

    # ---- Load data ----
    # The auxiliar.metric_computation.csv contains the 'bucket', 'avg_vol', 'country' and 'brand_name' 
    # columns, and should be calculated before running this script based on the train_data.csv file
    # (take a look at the documentation for details on how to calculate 'bucket' and 'avg_vol').
    df_aux = pd.read_csv(DATA_PATH / "auxiliar_metric_computation.csv")
    train_data = pd.read_csv(DATA_PATH / "train_data.csv")
    submission_data = pd.read_csv(DATA_PATH / "submission_data.csv")
    submission = pd.read_csv(DATA_PATH / "submission_template.csv")
```

* Notes:

  * `train_data.csv` and `submission_data.csv` here are **generic placeholders**; in your context they correspond to:

    * the training and test slices prepared from `df_volume_train` (plus joins with generics and medicine info),
    * the rows that correspond to the submission (same keys as `submission_template`).
  * `df_aux` is your computed **auxiliary** file.

Custom split & training:

```python
    # ---- Custom train/validation split ----
    train, validation = None # your_train_validation_split_function(train_data)

    # ---- Model training ----
    # Train your model here
```

* You are expected to implement:

  * a custom `train_validation_split_function`,
  * the modeling logic.

Predictions and metric computation:

```python
    # ---- Predictions on validation set ----
    prediction = validation.copy()
    prediction["volume"] = None #model.predict(validation)

    # ---- Compute metrics on validation set ----
    m1 = compute_metric1(validation, prediction, df_aux)
    m2 = compute_metric2(validation, prediction, df_aux)
```

* `validation` is treated as `df_actual` (it contains `volume` as ground truth).
* `prediction` is treated as `df_pred` (it has predicted `volume`).
* You call `compute_metric1` and `compute_metric2` to evaluate your model on your **local validation split**, mimicking the official evaluation.

Submission generation:

```python
    # ---- Generate submission file ----
    # Fill in predicted 'volume' values of the submission 
    submission["volume"] = None #model.predict(submission_data)

    # ...

    # Save submission
    SAVE_PATH = Path("path/to/save/folder")
    ATTEMPT = "attempt_x"
    submission.to_csv(SAVE_PATH / f"submission_{ATTEMPT}.csv", sep=",", index=False)
```

* `submission` = `submission_template.csv` with `volume` filled with predictions from your model.
* You save the submission with a unique `attempt_x` name.
* This file is what you upload to the Datathon platform.
