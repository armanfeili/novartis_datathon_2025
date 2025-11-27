# Novartis Datathon 2025 – Generic Erosion Forecasting

---

**Repository:** `novartis_datathon_2025`  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb)

This project implements an end-to-end, **config-driven forecasting pipeline** for the **Novartis Datathon 2025** generic erosion challenge.

The core goal is to:

> **Forecast monthly sales volume after generic entry for each (country, brand) under two official scenarios, using the competition’s custom error metrics and submission format.**

It combines:

- **Code on GitHub** – versioned development with VS Code + Copilot  
- **Storage on Google Drive (optional)** – datasets, runs, models, submissions  
- **Execution in Colab or locally** – quick setup, CPU-friendly (GPU not required)  
- **Reproducibility** – config snapshots, deterministic seeds, structured artifacts  

---

## 1. Competition Problem

Novartis wants to anticipate how **branded drug volume erodes after generic (gx) entry**.  
The datathon provides de-identified commercial data and asks participants to:

- Use **pre- and post-generic-entry history** to forecast **monthly volume** (`volume`),  
- Focus on two scenarios:

  1. **Scenario 1 (0 actuals)** – forecast months **0–23** after generic entry with only **pre-entry** data.  
  2. **Scenario 2 (6 actuals)** – forecast months **6–23** after generic entry, given pre-entry and the **first 6 months post-entry**.

- Optimize a **custom prediction error (PE)** metric, computed via the official `metric_calculation.py` using:
  - `auxiliar_metric_computation.csv` (with `avg_vol` and `bucket` per series),
  - Bucket-wise aggregation and **double weighting of high-erosion brands (Bucket 1)**.

---

## 2. Data Overview

The official dataset is split into **train** and **test**, each with three core tables:

### 2.1 Train

- `df_volume_train.csv`  
  Time-series fact table with the **target**:

  - `country`
  - `brand_name`
  - `month` (calendar month)
  - `months_postgx` (relative time; 0 at generic entry, negative before, positive after)
  - `volume` (monthly units sold)

- `df_generics_train.csv`  
  Time-varying generics competition:

  - `country`
  - `brand_name`
  - `months_postgx`
  - `n_gxs` (number of generic competitors at this relative month)

- `df_medicine_info_train.csv`  
  Static drug attributes per `(country, brand_name)`:

  - `ther_area` (therapeutic area)
  - `hospital_rate` (share of volume via hospital channel)
  - `main_package` (e.g., PILL, INJECTION, EYE DROP, Others)
  - `biological` (boolean)
  - `small_molecule` (boolean)

### 2.2 Test

Same schema, but with **missing target** (`volume`) for the post-entry months to be predicted:

- `df_volume_test.csv`
- `df_generics_test.csv`
- `df_medicine_info_test.csv`

### 2.3 Metrics & Submission Helpers

- `auxiliar_metric_computation.csv`  
  Per `(country, brand_name)`:

  - `avg_vol` – pre-entry average volume, used for metric normalization  
  - `bucket` – erosion bucket (1 = high erosion, 2 = lower erosion)

- `metric_calculation.py`  
  Official implementation of:

  - **Metric 1** – Phase 1A (Scenario 1, “0 actuals”)  
  - **Metric 2** – Phase 1B (Scenario 2, “6 actuals”)  

  Both compute a **normalized PE** combining:

  - Sum of absolute monthly errors over specified windows,
  - Absolute differences between cumulative actual and cumulative predicted volume,
  - Weighted by `avg_vol` and aggregated by `bucket` with extra weight on Bucket 1.

- `submission_template.csv`  
  Template with required columns:

  - `country,brand_name,months_postgx,volume`  

  Where `volume` is empty and must be filled with our predictions.

- `submission_example.csv`  
  Example submission showing the **exact format** expected by the platform.

---

## 3. Architecture

The project separates:

- **Code** (GitHub repo)  
- **Compute** (local machine or Colab VM)  
- **Data & Artifacts** (local `data/` folder or Google Drive mount)

```text
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL / VS Code + Copilot                                      │
│  ├─ Edit src/, configs/, notebooks/                             │
│  ├─ Commit & push to GitHub                                     │
│  └─ (Optionally) keep a local data/ folder                      │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  COLAB VM (/content/)                                          │
│  ├─ Clone repo from GitHub (code only)                         │
│  ├─ (Optional) Mount Google Drive at /content/drive            │
│  └─ Run training/inference → write artifacts & submissions     │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STORAGE (local or Drive)                                      │
│  ├─ data/raw/                  ← official CSVs                 │
│  ├─ data/interim/              ← merged/cleaned panels         │
│  ├─ data/processed/            ← feature matrices               │
│  ├─ artifacts/scenario1_runs/  ← Scenario 1 runs               │
│  ├─ artifacts/scenario2_runs/  ← Scenario 2 runs               │
│  └─ submissions/               ← submission_*.csv              │
└─────────────────────────────────────────────────────────────────┘
````

**Key principle:**
The pipeline is **config-driven and scenario-aware**. You can re-run it end-to-end (from raw CSVs to submission) with a small number of commands.

---

## 4. Repository Structure

```text
novartis_datathon_2025/
├─ data/
│  ├─ raw/                    # Original files from organizers
│  ├─ interim/                # Merged, cleaned panels
│  └─ processed/              # Scenario-specific feature tables
│
├─ configs/
│  ├─ data.yaml               # Paths, filenames, column definitions
│  ├─ features.yaml           # Scenario 1/2 feature engineering cfg
│  ├─ run_scenario1.yaml      # Scenario 1 run config (validation, paths)
│  ├─ run_scenario2.yaml      # Scenario 2 run config
│  ├─ model_gbm.yaml          # Main GBM hyperparameters
│  └─ model_baseline.yaml     # Naive baseline configs
│
├─ src/
│  ├─ cfg.py                  # Simple configuration loader
│  ├─ utils.py                # Seeding, logging, timers
│  ├─ data.py                 # Data loading & panel construction
│  ├─ features.py             # Feature engineering (S1/S2)
│  ├─ metrics.py              # Thin wrapper around metric_calculation.py
│  ├─ validation.py           # Scenario-aware train/val splitting
│  ├─ train_scenario1.py      # End-to-end training for Scenario 1 (Metric 1)
│  ├─ train_scenario2.py      # End-to-end training for Scenario 2 (Metric 2)
│  ├─ inference.py            # Model loading & prediction helpers
│  ├─ submission.py           # Fill submission_template.csv with preds
│  └─ models/
│      ├─ base.py             # Base model interface
│      ├─ gbm.py              # LightGBM/XGBoost/CatBoost wrapper
│      └─ baseline.py         # Naive baselines for comparison
│
├─ notebooks/
│  ├─ 00_eda_erosion.ipynb    # EDA on erosion patterns & buckets
│  ├─ 01_scenario1_experiments.ipynb
│  ├─ 02_scenario2_experiments.ipynb
│  └─ colab/
│      └─ main.ipynb          # Colab-friendly end-to-end workflow
│
├─ artifacts/
│  ├─ scenario1_runs/         # Run dirs for Scenario 1
│  └─ scenario2_runs/         # Run dirs for Scenario 2
│
├─ submissions/               # Final CSVs for upload
├─ metric_calculation.py      # Official metric implementation (provided)
├─ requirements.txt           # Python dependencies
└─ README.md                  # This file
```

Each run under `artifacts/scenario*/` typically contains:

* `config_used.yaml` – frozen run configuration
* `metrics.json` – Metric 1 or 2 + internal metrics
* `oof_preds.csv` – out-of-fold or validation predictions
* `model_fold_*.pkl` – model weights per fold (if CV used)
* `logs.txt` – training log

---

## 5. Quick Start (Colab)

### Step 1 – Open in Colab

Click the badge at the top or open:

```text
https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/Arman/notebooks/colab/main.ipynb
```

### Step 2 – (Optional) Mount Google Drive

The Colab notebook will:

* Ask to mount Drive (if you choose to store data & artifacts there),
* Or you can keep everything under `/content/novartis_datathon_2025`.

### Step 3 – Run All Cells

The notebook will:

1. Install dependencies
2. Load configs
3. Load and merge the three train tables into a panel
4. Build features for Scenario 1 and Scenario 2
5. Train baseline and GBM models
6. Evaluate locally with **official Metric 1 / Metric 2**
7. Generate submission files from `submission_template.csv`

---

## 6. End-to-End Workflow

High-level flow for each scenario:

```text
1. Place official CSVs in data/raw/
2. Configure paths & columns in configs/data.yaml
3. (Optional) Adjust features in configs/features.yaml
4. Train:
   - Scenario 1 → python -m src.train_scenario1 ...
   - Scenario 2 → python -m src.train_scenario2 ...
5. Inspect artifacts/scenario*_runs/<run_id>/metrics.json
6. Use src/inference.py + src/submission.py to generate submission CSV
7. Upload CSV to competition platform
```

### Example Commands (Local CLI)

```bash
# Train Scenario 1
python -m src.train_scenario1 \
    --run-config configs/run_scenario1.yaml \
    --model-config configs/model_gbm.yaml

# Train Scenario 2
python -m src.train_scenario2 \
    --run-config configs/run_scenario2.yaml \
    --model-config configs/model_gbm.yaml
```

After training, generate a submission:

```bash
python -m src.submission \
    --scenario 1 \
    --run-id <best_scenario1_run_id> \
    --output submissions/submission_scenario1.csv
```

(Adapt arguments to your actual script interface.)

---

## 7. Local Development

### 1. Clone Repository

```bash
git clone https://github.com/armanfeili/novartis_datathon_2025.git
cd novartis_datathon_2025
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Data

Copy the competition CSVs into `data/raw/`:

* `df_volume_train.csv`
* `df_generics_train.csv`
* `df_medicine_info_train.csv`
* `df_volume_test.csv`
* `df_generics_test.csv`
* `df_medicine_info_test.csv`
* `auxiliar_metric_computation.csv`
* `submission_template.csv`

### 4. Develop & Iterate

* Edit `src/`, `configs/`, and `notebooks/` using VS Code + Copilot.
* Commit and push changes to GitHub.
* Re-run training in Colab or locally as needed.

---

## 8. Notes

* The project is designed around the **two official scenarios** and the **exact competition metrics**.
* Baseline models are kept intentionally simple to highlight the added value of the main GBM model.
* The code is organized to make it easy to:

  * Swap models,
  * Extend features,
  * Adjust validation strategies,
  * And reproduce results with minimal friction.

---

## Creator

Created by **[Arman Feili](https://github.com/armanfeili)** for the **Novartis Generic Erosion Datathon 2025**.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
