# Novartis Datathon 2024

A machine learning project for predicting pharmaceutical drug sales across different markets and clusters, developed for the [Novartis Datathon 2024](https://godatathon.com) competition powered by [Eurecat](https://eurecat.org/home/en/).

## Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Evaluation Metric](#evaluation-metric)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Features & Feature Engineering](#features--feature-engineering)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Team](#team)
- [License](#license)

## About the Project

This repository contains the solution developed by our team for the Novartis Datathon 2024 - a data science competition focused on pharmaceutical sales forecasting. The challenge involves predicting future drug sales using historical data, handling cold-start problems for new drug launches, and optimizing for a custom metric (CYME - Combined Yearly and Monthly Error).

## Problem Statement

The goal is to predict monthly pharmaceutical drug sales (`target`) given historical sales data and various features related to drugs, markets, and economic indicators. Key challenges include:

- **Cold Start Problem**: Predicting sales for drugs/clusters that have no historical data in the training set
- **Time Series Forecasting**: Handling temporal dependencies in sales data
- **Multi-dimensional Data**: Managing multiple drugs across multiple countries and clusters
- **Outlier Handling**: Dealing with negative values and extreme outliers in the data

## Dataset

The dataset consists of two main files:

| File | Description |
|------|-------------|
| `train_data.csv` | Historical training data with target sales values |
| `submission_data.csv` | Test data for which predictions need to be submitted |

### Key Features

| Feature | Description |
|---------|-------------|
| `cluster_nl` | Unique identifier for drug-market cluster |
| `drug_id` | Drug identifier |
| `brand` | Brand name |
| `corporation` | Pharmaceutical corporation |
| `country` | Country code |
| `indication` | Medical indication(s) for the drug |
| `therapeutic_area` | Therapeutic area classification |
| `launch_date` | Drug launch date in the market |
| `ind_launch_date` | Indication launch date |
| `date` | Time period for the sales record |
| `price_unit` | Unit price |
| `price_month` | Monthly price |
| `che_pc_usd` | Current health expenditure per capita (USD) |
| `che_perc_gdp` | Health expenditure as percentage of GDP |
| `public_perc_che` | Public expenditure as percentage of health expenditure |
| `insurance_perc_che` | Insurance percentage of health expenditure |
| `prev_perc` | Prevalence percentage |
| `target` | Monthly sales value (target variable) |

## Evaluation Metric

The competition uses a custom **CYME (Combined Yearly and Monthly Error)** metric:

$$CYME = \frac{1}{2} \times (\text{median}(\text{yearly\_error}) + \text{median}(\text{monthly\_error}))$$

The final score is a weighted combination of:

- **Recent launches**: Products that exist in both training and test sets
- **Future launches (Zero Actuals)**: Products that only appear in the test set (cold start)

$$\text{Score} = w_{recent} \times CYME_{recent} + w_{future} \times \min(1, CYME_{future})$$

## Solution Approach

### Data Preprocessing

1. **Missing Value Handling**: Replace `-1` values with column means for economic indicators
2. **Outlier Removal**: Z-score based outlier removal for specific columns
3. **Date Feature Engineering**: Extract year, month, sale_month (months since launch)
4. **Future Launch Identification**: Identify cold-start products for proper evaluation

### Feature Engineering

- **Temporal Features**: Year, month, sale_month (months since drug launch), ind_sale_month
- **LTM (Last Twelve Months) KPIs**: Rolling 12-month aggregations with various fill strategies
- **Sum of First Year Targets**: Aggregate sales from the first year after launch

### Model Selection & Training

Multiple models were experimented with, using time-series cross-validation for robust evaluation.

## Project Structure

```text
novartis_datathon_2024/
├── data/
│   └── input/
│       ├── train_data.csv          # Training dataset
│       └── submission_data.csv     # Test dataset for predictions
├── src/
│   ├── 0_EDA.ipynb                 # Exploratory Data Analysis notebook
│   ├── 10_Naive_Model.ipynb        # Baseline naive model
│   ├── automl.py                   # AutoML with LazyPredict
│   ├── helper.py                   # CYME metric computation
│   ├── utils.py                    # Utility functions for data loading/processing
│   ├── model_catboost.py           # CatBoost implementation
│   ├── model_catboost_best_score_point_045.py  # Optimized CatBoost
│   ├── model_xgboost.py            # XGBoost implementation
│   ├── model_xgboost_best_score.py # Optimized XGBoost with CV
│   ├── model_hgbr.py               # HistGradientBoostingRegressor
│   ├── model_etr.py                # ExtraTreesRegressor
│   └── models/
│       ├── __init__.py
│       └── models.py               # Model abstractions (Naive, CatBoost)
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_utils.py           # Unit tests
├── pyproject.toml                  # Poetry dependencies & project config
├── poetry.toml                     # Poetry settings
├── settings.yaml                   # Model parameters configuration
├── profiling_report.html           # YData profiling report
└── README.md
```

## Models Implemented

| Model | Description | Best CYME Score |
|-------|-------------|-----------------|
| **Naive** | Baseline model predicting constant value | ~1.0 |
| **XGBoost** | Gradient boosting with categorical support | ~0.055 |
| **CatBoost** | Gradient boosting optimized for categorical features | **~0.045** |
| **HistGradientBoostingRegressor** | Sklearn's histogram-based gradient boosting | ~0.077 |
| **ExtraTreesRegressor** | Extremely randomized trees | ~0.033 |
| **LazyPredict AutoML** | Automated model comparison | Various |

### Model Configurations

**CatBoost (Best Performer)**:

- Depth: 8-12
- Categorical features handled natively
- Custom CYME evaluation metric

**XGBoost**:

- Tree method: histogram
- Max depth: 5-30
- Max categorical threshold: 1000
- Time-series cross-validation with 5 splits

## Features & Feature Engineering

### Categorical Features

- `brand`, `cluster_nl`, `corporation`, `country`, `drug_id`, `indication`, `therapeutic_area`

### Numerical Features

- Economic indicators: `che_pc_usd`, `che_perc_gdp`, `public_perc_che`, `insurance_perc_che`
- Price metrics: `price_unit`, `price_month`
- Prevalence: `prev_perc`

### Engineered Features

- `year`, `month` - Extracted from date
- `sale_month` - Months since drug launch
- `ind_sale_month` - Months since indication launch
- `ltm_target` - Last twelve months rolling sum
- `sum_of_first_year_targets` - Aggregate first year sales

## Setup & Installation

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/) (Python dependency manager)

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/JSchoeck/novartis_datathon_2024.git
   cd novartis_datathon_2024
   ```

2. **Install dependencies with Poetry**

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**

   ```bash
   poetry shell
   ```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML utilities and models |
| `xgboost` | XGBoost regressor |
| `catboost` | CatBoost regressor |
| `lightgbm` | LightGBM regressor |
| `darts` | Time series forecasting |
| `neuralforecast` | Neural network forecasting |
| `statsforecast` | Statistical forecasting |
| `mlforecast` | ML-based forecasting |
| `lazypredict` | AutoML model comparison |
| `matplotlib` / `plotly` | Visualization |
| `ydata-profiling` | Automated EDA |
| `pytest` | Testing framework |
| `ruff` | Linting and formatting |

## Usage

### 1. Exploratory Data Analysis

Run the EDA notebook to understand the data:

```bash
jupyter notebook src/0_EDA.ipynb
```

### 2. Train a Model

Run any of the model scripts:

```bash
# CatBoost (best performing)
python src/model_catboost_best_score_point_045.py

# XGBoost with cross-validation
python src/model_xgboost_best_score.py

# AutoML comparison
python src/automl.py
```

### 3. Generate Predictions

Each model script automatically:

1. Loads and preprocesses data
2. Trains the model
3. Evaluates using CYME metric
4. Generates submission file in `data/output/`

### 4. Configuration

Modify `settings.yaml` to adjust parameters:

```yaml
params:
  user: "your_name"
  test_size: 0.3

naive:
  value: 1.01
  
xgboost:
  param1: 0
```

### 5. Run Tests

```bash
pytest src/tests/
```

## Key Utilities

### `utils.py` Functions

| Function | Description |
|----------|-------------|
| `load_data(kind)` | Load train/predict datasets |
| `add_date_features(df)` | Extract date-based features |
| `identify_future_launches(df_train, df_test)` | Mark cold-start products |
| `remove_outlier_data(df, column, threshold_z)` | Z-score outlier removal |
| `replace_minus_one_with_mean(df, columns)` | Handle missing values |
| `add_ltm_kpis(df, columns)` | Calculate rolling 12-month KPIs |
| `save_submission_file(df, attempt, model, user)` | Save predictions |

### `helper.py` Functions

| Function | Description |
|----------|-------------|
| `compute_metric(submission)` | Calculate CYME score |
| `compute_metric_terms(submission)` | Get separate recent/future scores |
| `cyme_scorer()` | Sklearn-compatible scorer |

## Team

- **Simon Walker** - [simonnwalker297@gmail.com](mailto:simonnwalker297@gmail.com)
- **Johannes Schöck** - [johannes@schoeck.org](mailto:johannes@schoeck.org)

## License

This project is released under the [Unlicense](LICENSE) - see the LICENSE file for details.

---

Developed for the Novartis Datathon 2024
