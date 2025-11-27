# Project Structure and File Responsibilities

This document provides a detailed overview of the project's folder structure and the specific responsibilities of each file. This structure is designed to separate configuration, source code, notebooks, and artifacts, facilitating a clean workflow between local development (VS Code) and remote training (Google Colab).

## Root Directory

- **`README.md`**: The main entry point for the project. It contains instructions on how to set up the project, the architecture overview, and quick start guides for both local and Colab environments.
- **`LICENSE`**: The license file for the project.
- **`CONTRIBUTING.md`**: Guidelines for contributing to the project.

## `configs/`

This directory contains all YAML configuration files. These files allow for changing experiment parameters without modifying the code.

- **`data.yaml`**: Defines paths to data directories (local and Drive), file names, column types, and data validation rules.
- **`features.yaml`**: Controls feature engineering steps, including which feature groups to generate (lags, rolling stats, etc.) and their specific parameters.
- **`model_lgbm.yaml`**: Hyperparameters and training settings specific to LightGBM models.
- **`model_xgb.yaml`**: Hyperparameters and training settings specific to XGBoost models.
- **`model_cat.yaml`**: Hyperparameters and training settings specific to CatBoost models.
- **`model_linear.yaml`**: Configuration for linear models (Ridge, Lasso, ElasticNet) and their preprocessing steps.
- **`model_nn.yaml`**: Architecture definitions and training parameters for Neural Network models (PyTorch).
- **`run_defaults.yaml`**: Global settings for experiments, including random seeds, cross-validation strategies, logging, and hardware configuration.

## `src/`

The source code directory containing the core logic of the pipeline.

- **`data.py`**: Contains the `DataManager` class responsible for loading raw data, performing initial cleaning/merging (`make_interim`), and saving processed data.
- **`features.py`**: Contains the `FeatureEngineer` class which implements feature generation logic based on `features.yaml`.
- **`validation.py`**: Contains the `Validator` class for creating cross-validation splits (K-Fold, TimeSeriesSplit) and adversarial validation.
- **`train.py`**: The main orchestration script. It loads configs, prepares data, runs the training loop across folds, saves models, and logs metrics.
- **`evaluate.py`**: Contains the `Evaluator` class for calculating performance metrics (RMSE, MAE, etc.) and generating evaluation plots.
- **`inference.py`**: Script for loading trained models and generating predictions on the test set.
- **`utils.py`**: General utility functions for logging, timing execution, setting random seeds, and handling paths.

### `src/models/`

Wrappers for different machine learning libraries to ensure a consistent interface.

- **`base.py`**: Defines the abstract `BaseModel` class that all model wrappers must inherit from. Enforces `fit`, `predict`, `save`, and `load` methods.
- **`lgbm_model.py`**: Wrapper for LightGBM.
- **`xgb_model.py`**: Wrapper for XGBoost.
- **`cat_model.py`**: Wrapper for CatBoost.
- **`linear.py`**: Wrapper for Scikit-Learn linear models.
- **`nn.py`**: Wrapper for PyTorch neural networks, handling data loaders and training loops.

## `notebooks/`

Jupyter notebooks for exploration and prototyping.

- **`00_eda.ipynb`**: Exploratory Data Analysis to understand data distributions and relationships.
- **`01_feature_prototype.ipynb`**: A playground for testing new feature engineering ideas before moving them to `src/features.py`.
- **`02_model_sanity.ipynb`**: Used to verify that models can be initialized and trained on dummy or small data subsets.

### `notebooks/colab/`

Notebooks specifically designed for the Google Colab environment.

- **`main.ipynb`**: The single entry point for the Colab workflow. It handles environment setup (Drive mount, dependency installation) and runs training experiments by calling `src/train.py`.

## `data/`

Local storage for data. **Note:** In the Colab workflow, the actual data resides in Google Drive.

- **`raw/`**: Place for original, immutable data files.
- **`interim/`**: Intermediate data that has been cleaned or merged.
- **`processed/`**: Final datasets ready for modeling.

## `artifacts/`

Stores outputs generated during training runs.

- **`runs/`**: Contains subdirectories for each experiment run (named by timestamp and model), storing:
  - Config snapshots
  - Trained model binaries/checkpoints
  - Logs and metrics
  - Prediction files (OOF and test)

## `submissions/`

Directory for storing final CSV files generated for submission to the datathon platform.

## `env/`

Environment definition files.

- **`requirements.txt`**: Python dependencies for local development (pip).
- **`environment.yml`**: Conda environment definition.
- **`colab_requirements.txt`**: A minimal set of dependencies required to run the project in Google Colab.

## `docs/`

Project documentation.

- **`problem.md`**: Description of the business problem and objectives.
- **`data_schema.md`**: Documentation of the data tables and columns.
- **`validation.md`**: Explanation of the validation strategy used.
- **`experiments_log.md`**: A manual or automated log of experiments and their results.
- **`project_structure.md`**: This file.
