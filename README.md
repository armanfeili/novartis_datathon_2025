# Novartis Datathon 2025

---

novartis_datathon_2025

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/main/notebooks/colab/01_colab_setup.ipynb)

**This project combines the power of GitHub Copilot in VS Code with GPU training in Google Colab for the Novartis Datathon 2025.**

- **Code on GitHub** — version control with Copilot support
- **Storage on Google Drive** — datasets, runs, checkpoints persist across sessions
- **Run in Colab** — free T4/A100 GPUs, no local setup required
- **Reproducible** — frozen configs, deterministic seeds, structured outputs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL (VS Code + Copilot)                                      │
│  ├─ Edit src/, configs/, notebooks/                            │
│  ├─ Commit & push to GitHub                                    │
│  └─ No data/runs stored here                                   │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  COLAB VM (/content/)                                           │
│  ├─ Clone repo from GitHub (code only)                         │
│  ├─ Mount Google Drive at /content/drive                       │
│  └─ Run training → outputs to Drive                            │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOOGLE DRIVE (MyDrive/novartis-datathon-2025/)                │
│  ├─ data/raw/                  ← datasets cached here          │
│  ├─ data/processed/             ← preprocessed data            │
│  ├─ artifacts/runs/<run_id>/    ← per-run outputs              │
│  │   ├─ config_used.yaml       ← frozen config                 │
│  │   ├─ metrics.json           ← metrics                       │
│  │   ├─ checkpoints/           ← model weights                 │
│  │   ├─ preds/                 ← predictions                   │
│  │   └─ logs.txt               ← logs                          │
│  └─ submissions/               ← submission files              │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principle:** Code travels through Git. Data stays in Drive.

**Google Drive Folder:** [Link](https://drive.google.com/drive/u/2/folders/1iP8ffP5MfJ6hqig1-lQpVwzfrl9uuU03)

---

## Quick Start (3 steps)

### 1. Open in Colab
Click the badge above or go to:
```
https://colab.research.google.com/github/armanfeili/novartis_datathon_2025/blob/main/notebooks/colab/01_colab_setup.ipynb
```

### 2. Enable GPU
- **Runtime** → **Change runtime type** → **GPU** (T4 or A100) → **Save**

### 3. Run Setup
- Run `notebooks/colab/01_colab_setup.ipynb` to mount Drive and install dependencies.
- Run `notebooks/colab/02_colab_experiments.ipynb` to start training.

---

## Repository Structure

```
novartis-datathon-2025/
├─ data/
│  ├─ raw/             # Original files from organizers (read-only)
│  ├─ interim/         # Lightly cleaned/merged tables
│  └─ processed/       # Final feature tables ready for modeling
│
├─ configs/
│  ├─ data.yaml        # paths, columns, keys, date configs
│  ├─ features.yaml    # which features/lag windows to use
│  ├─ model_lgbm.yaml  # hyperparams for LightGBM
│  ├─ model_xgb.yaml   # hyperparams for XGBoost
│  ├─ model_cat.yaml   # hyperparams for CatBoost
│  ├─ model_linear.yaml# linear/tree models configs
│  ├─ model_nn.yaml    # NN architecture & training params
│  └─ run_defaults.yaml# default run-level settings
│
├─ src/
│  ├─ data.py          # Data loading & processing
│  ├─ features.py      # Feature engineering
│  ├─ validation.py    # Validation strategies
│  ├─ models/          # Model wrappers
│  ├─ train.py         # Main training loop
│  ├─ evaluate.py      # Evaluation metrics & plots
│  ├─ inference.py     # Inference script
│  └─ utils.py         # Utilities
│
├─ notebooks/
│  ├─ 00_eda.ipynb           # EDA
│  ├─ 01_feature_prototype.ipynb
│  ├─ 02_model_sanity.ipynb
│  └─ colab/                 # Colab specific notebooks
│
├─ artifacts/          # Training artifacts (Drive synced)
├─ submissions/        # Submission files (Drive synced)
├─ env/                # Environment files
└─ docs/               # Documentation
```

---

## Google Drive Structure (Storage Only)

The project is configured to use the following Google Drive folder:
`https://drive.google.com/drive/u/2/folders/1iP8ffP5MfJ6hqig1-lQpVwzfrl9uuU03`

Structure created in Drive:
```
novartis-datathon-2025/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   └── runs/
└── submissions/
```

---

## How It Works

### A. Configuration
All settings are in `configs/`. Modify `model_*.yaml` to change hyperparameters or `features.yaml` to change feature engineering.

### B. Training
Run `src/train.py` (via `notebooks/colab/02_colab_experiments.ipynb` in Colab) to start a training run.
This will:
1. Load data from Drive
2. Generate features
3. Train model using specified config
4. Save artifacts to Drive (`artifacts/runs/<run_id>`)

### C. Inference
Run `src/inference.py` to generate predictions using a trained model.

---

## Local Development

### 1. Clone Repo
```bash
git clone https://github.com/armanfeili/novartis_datathon_2025.git
cd novartis_datathon_2025
```

### 2. Install Dependencies
```bash
pip install -r env/requirements.txt
```

### 3. Edit Code
- Use **VS Code** with **GitHub Copilot**
- Modify `src/` or `configs/`
- Commit changes

### 4. Sync
Push changes to GitHub, then pull in Colab.

---

## Creator

Created by **[Arman Feili](https://github.com/armanfeili)**

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## See Also

- [notebooks/colab/01_colab_setup.ipynb](notebooks/colab/01_colab_setup.ipynb) — Colab setup
- [src/train.py](src/train.py) — Main training script