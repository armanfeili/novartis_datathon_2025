* Novartis-style financial time-series / panel data
* 4-day online datathon structure
* A team of **Arman (engineering + DS)** and **Saeed (statistics + DS)**

I’ll structure it as:

1. Phase 1 – Understand & Design
2. Phase 2 – Core Implementation & “Guaranteed Good” Solution
3. Phase 3 – Aggressive Improvement & Winning Edge

For each phase: **goals, concrete steps, advanced approaches, and generalization tactics**.

---

## Phase 1 – Understanding & Design (Task, Data, Research)

**Goal:** In a few hours, build a *deep mental model* of the problem and dataset, define a **robust validation strategy**, and design a pipeline that will generalize beyond this particular test split.

Think of Phase 1 as producing a **“design doc” + scaffolded repo**, not models.

### 1.1. Problem framing & evaluation (Saeed lead, Arman supporting)

1. Write a **1–2 sentence problem statement** in a Markdown file (`docs/problem.md`):

   * What are we predicting (exact target)?
   * At which granularity (country/brand/portfolio/time)?
   * For what decision (budget allocation, risk assessment, forecasting)?

2. Clarify **evaluation** precisely:

   * Metric (RMSE, MAE, MAPE, custom) and level (overall vs per entity, weighted by revenue or not).
   * Public vs private leaderboard splitting (if any).
   * Role of jury: how much weight for metric vs business story vs robustness.

3. Define **success criteria**:

   * Quantitative: “We want at least X% improvement over naive baseline and official baseline.”
   * Qualitative: “We can explain our model in 60 seconds to a CFO and show 2–3 compelling allocation scenarios.”

4. Decide **risk posture**:

   * If metric is volatile and jury matters a lot → prioritize **robustness + narrative** over exotic models.
   * If metric is everything → accept more modeling complexity, but still with solid validation.

### 1.2. Data & domain understanding (joint)

Once dataset is released:

1. Inventory all files:

   * Fact table(s): time-stamped financial data.
   * Dimension tables: products, countries, portfolios, regions.
   * Sample submission / test set.

2. Build a **schema map** (even rough) in `docs/data_schema.md`:

   * For each table: primary keys, foreign keys, main fields.
   * Diagram: fact at center, dims around (country, brand, portfolio).

3. Domain notes:

   * Identify likely **life-cycle indicators** (years since first sale, product type).
   * Any explicit investment / marketing spend variables (these are critical for allocation story).
   * Fields that scream “reporting artifact” (indices, codes) that are not domain.

This doc becomes your single source of truth. You’ll revisit it constantly.

### 1.3. Validation design – the core of generalization (Saeed lead)

This is the most critical part for **generalization to larger data** and hidden splits.

1. Identify the **time horizon**:

   * Are we predicting the next month(s), year, or multiple horizons?
   * Are train and test separated by time (usual in this type of competition)?

2. Use **time-based validation** as default:

   * For each entity (e.g., country-brand), split by time: train on early periods, validate on later ones.
   * Option A: Single hold-out (e.g., last 6–12 months as validation).
   * Option B (better): **rolling/expanding window** (TimeSeriesSplit or custom).

3. Consider **hierarchical aspects**:

   * If you suspect train/test differ in regions or product types, design **stratified time split**: keep the same proportion of regions in val as in train.

4. Add **adversarial validation early** (advanced but powerful):

   * Train a classifier to distinguish train rows from test rows using features (without the target).
   * If AUC is high → distribution shift.
   * Use this to:

     * Identify features that drive shift.
     * Possibly **reweight** or be more conservative with those features.

5. Explicitly document validation in `docs/validation.md`:

   * How you split.
   * Why it matches the real decision scenario.
   * Any known limitations.

This is a huge advantage in juries: you can *prove* you thought about generalization, not just fitting the test.

### 1.4. Project structure & tooling (Arman lead)

Set up a code base that allows fast iteration but stays clean:

1. **Repo structure** (you already have a version, but for datathon):

   ```text
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
   │  ├─ model_nn.yaml    # NN architecture & training params (epochs, lr, etc.)
   │  └─ run_defaults.yaml# default run-level settings (seeds, cv scheme, etc.)
   │
   ├─ src/
   │  ├─ data.py          # load_raw_data(), make_interim(), make_processed()
   │  ├─ features.py      # build_features(df), add_lags(), rolling stats, etc.
   │  ├─ validation.py    # time-based splits, rolling CV, adversarial validation
   │  ├─ models/
   │  │  ├─ base.py       # BaseModel interface (fit, predict, save, load)
   │  │  ├─ lgbm_model.py # LightGBM wrapper
   │  │  ├─ xgb_model.py  # XGBoost wrapper
   │  │  ├─ cat_model.py  # CatBoost wrapper
   │  │  ├─ linear.py     # Ridge/Lasso/etc.
   │  │  └─ nn.py         # PyTorch/Keras tabular NN with checkpointing
   │  ├─ train.py         # main train loop; orchestrates data→features→model
   │  ├─ evaluate.py      # metrics, plotting, error analysis
   │  ├─ inference.py     # load best model + generate predictions on test
   │  └─ utils.py         # logging, timers, seed setting, path helpers
   │
   ├─ notebooks/
   │  ├─ 00_eda.ipynb           # EDA & understanding
   │  ├─ 01_feature_prototype.ipynb
   │  ├─ 02_model_sanity.ipynb
   │  └─ colab/
   │     ├─ 01_colab_setup.ipynb    # mount Drive, pip install, set paths
   │     └─ 02_colab_experiments.ipynb
   │
   ├─ artifacts/
   │  └─ runs/
   │     ├─ 2025-11-27_lgbm_v1/          # RUN_ID (timestamp + short name)
   │     │  ├─ config_used.yaml          # merged config snapshot for this run
   │     │  ├─ model/                    # final frozen model(s)
   │     │  │  └─ best_model.bin
   │     │  ├─ checkpoints/              # periodic checkpoints (NN and others)
   │     │  │  ├─ epoch_001.pt
   │     │  │  ├─ epoch_010.pt
   │     │  │  └─ best.pt
   │     │  ├─ metrics.json              # train/val metric history, final scores
   │     │  ├─ logs.txt                  # training logs (per epoch, warnings, etc.)
   │     │  ├─ preds/                    # predictions for val/test
   │     │  │  ├─ oof_train.csv          # out-of-fold predictions
   │     │  │  └─ test_predictions.csv   # raw test predictions
   │     │  └─ figures/                  # plots for this run (errors, importances)
   │     │
   │     ├─ 2025-11-27_catboost_v2/
   │     └─ 2025-11-28_nn_ensemble_v1/
   │
   ├─ submissions/
   │  ├─ baseline_naive.csv
   │  ├─ 2025-11-29_lgbm_v1.csv
   │  └─ 2025-11-30_ensemble_final.csv
   │
   ├─ env/
   │  ├─ requirements.txt        # for local (pip)
   │  ├─ environment.yml         # optional, for conda
   │  └─ colab_requirements.txt  # minimal pip install list for Colab
   │
   ├─ docs/
   │  ├─ problem.md              # problem statement & business framing
   │  ├─ data_schema.md          # tables, keys, column descriptions
   │  ├─ validation.md           # split strategy, rationale
   │  └─ experiments_log.md      # short log of runs (ID, model, metric)
   │
   └─ README.md                  # how to run on local & Colab
   ```

2. **Config**:

   * Put paths, feature lists, model hyperparameters, seeds in `config.py` or YAML.
   * This makes it easy to rerun everything with one change.

3. **Experiment logging** (lightweight):

   * In `docs/experiments_log.md`, each row is:

     * Date/time, model name, features, validation scheme, metric.
   * This avoids confusion on Day 3 when many experiments exist.

### 1.5. EDA plan – not random exploration

You will not have time for full exhaustive EDA. Design it upfront:

1. **Overview EDA** (quick):

   * `info()`, `describe()`, missingness, correlations.
   * Size, time span, number of entities.

2. **Target EDA**:

   * Plot target over time for:

     * A few large markets / brands.
     * A small market.
   * Check:

     * Seasonality, trends, volatility.
     * Sudden jumps or structural breaks.

3. **Key candidates for feature engineering**:

   * Investment vs revenue scatter plots.
   * Portfolio / region breakdown.
   * Basic growth metrics.

Define in advance which plots matter for modeling, to avoid wasting hours.

---

## Phase 2 – Data Processing & Core Implementation (Solid “Guaranteed Good” Solution)

**Goal:** Build a **reliable, well-validated, end-to-end solution** that:

* Produces valid submissions
* Achieves clearly better than baseline performance
* Is robust, interpretable enough, and easy to extend in Phase 3

Treat Phase 2 as building the “production skeleton” that you can then optimize.

### 2.1. Data cleaning & preprocessing

1. Implement `data_loading.py`:

   * Standardized reading functions.
   * Joining of fact + dims.
   * Any simple unit conversions.

2. Handle **missingness** with rules:

   * Time series: forward/backward fill where appropriate; else explicit NA indicators.
   * Categorical: special “missing” category.
   * Document each imputation decision (you may need it for Q&A).

3. Create **train/test objects**:

   * Train with target available.
   * Test with same columns but no target.
   * Ensure consistent encoding (same dtypes/categories).

### 2.2. Feature engineering – v1 (general but not overcomplicated)

Objective: create a **strong generic feature set** that works for most panel/time-series finance problems.

In `features.py`, implement:

1. **Time features**:

   * `month`, `quarter`, `year`.
   * If dataset is long enough, a time index (e.g., months since start).

2. **Lag features**:

   * For each entity (country-brand / relevant key):

     * `target_t-1`, `target_t-2`, `target_t-3`
     * Rolling mean & std over last 3–6 periods.

3. **Growth & momentum**:

   * `(target_t-1 - target_t-2) / (|target_t-2| + ε)`
   * 3-month cumulative growth.

4. **Hierarchy aggregates**:

   * For each time step, compute:

     * Total revenue per country, per portfolio, per region.
     * Entity’s share of those totals.
   * These features generalize well to larger datasets.

5. **Investment-related features** (if available):

   * Current and lagged investments.
   * Ratio `revenue / investment` as ROI proxy.
   * Bucketed investment levels (low/medium/high).

Generalization angle: these engineered features don’t depend on a specific test split; they exploit the **structure of the process**, not the quirks of a given dataset.

### 2.3. Baseline models & “default hero model”

**Target for Phase 2:** a single, robust “hero model” you can rely on.

1. **Baselines**:

   * Time naive: last value / moving average.
   * Simple linear regression on lags and time features.
   * Evaluate with your chosen time split.
   * Save results; you will need them in the final story.

2. **Hero model choice** (realistic for this datathon):

   * **Gradient boosting**: LightGBM / XGBoost / CatBoost.
   * Use all the features from v1 plus categorical encodings.

3. **First hero model training**:

   * Use a **single, clear validation** setup from Phase 1 (e.g., last 12 months).
   * Conservative hyperparameters:

     * Moderate depth (max_depth 6–8).
     * Enough estimators with early stopping.
     * Regularization (L2, subsampling).

4. **Check performance & sanity**:

   * Compare to baselines.
   * Plot residuals vs time, vs regions, vs entities.
   * Check for obvious leakage patterns (e.g., suspiciously low error).

If this model is clean and clearly better than baselines, you already have a **top-tier candidate** in many competitions. Phase 3 is about pushing from “top-10” to “top-3”.

### 2.4. End-to-end submission pipeline

In `submission.py` implement:

1. A `train_and_predict()` function that:

   * Loads raw data.
   * Runs preprocessing and feature engineering.
   * Trains final hero model on full training data.
   * Generates predictions for test.
   * Outputs a file that matches the **sample submission format exactly**.

2. Ensure it is **fully reproducible**:

   * Fixed random seeds.
   * Deterministic data operations (sorted by ID and time before lagging).
   * No manual notebook editing steps needed.

3. Test this pipeline **at least twice** before Phase 3:

   * Once early in Day 2.
   * Once before going to sleep Day 3.

This ensures that whatever crazy improvements you try in Phase 3, you always have a **safe fallback** ready.

### 2.5. Early business story draft

Before Phase 2 ends, start a **1-page narrative** (Saeed focuses on interpretation, Arman on visuals):

1. Describe:

   * What drives the target according to feature importance.
   * A couple of **example entities** (markets/brands) and how the model predicts them.

2. Create **2–3 key plots** you know you will use in slides:

   * Overall error distribution.
   * Error by region/portfolio.
   * Example forecast vs actual for a representative entity.

These will evolve in Phase 3 but having a draft structure is crucial.

---

## Phase 3 – Improvement, Edge, and Polish (Push to 1st Place)

**Goal:** Starting from a solid hero model, aggressively improve **performance, robustness, and business value**, while building a **killer presentation**.

Think of Phase 3 as three parallel tracks:

1. Modeling & feature improvements
2. Robustness & generalization checks
3. Storytelling & presentation

### 3.1. Advanced modeling & features

Here we list **many** options; you will choose what fits time constraints and data.

#### 3.1.1. Stronger feature engineering

1. **Life-cycle stage features**:

   * For each entity: time from first non-zero observation.
   * Categorize into “launch / growth / maturity / decline”.
   * Add as categorical feature.

2. **Stability & volatility**:

   * Rolling standard deviation of revenue/investment.
   * Coefficient of variation (std / mean) over past N periods.
   * Entities with high volatility may need different treatment.

3. **Segment-specific models** (if enough data):

   * Train separate models per region or portfolio type.
   * Or at least add interaction features (region × time).

4. **Quantile / distributional features**:

   * Train models for quantiles (e.g., 25%, 50%, 75%) to approximate uncertainty.
   * Use predicted spread as risk indicator for allocation logic.

5. **Adversarially stable features**:

   * From your adversarial validation, identify features that differ strongly between train and test.
   * Consider dropping or down-weighting them to improve **out-of-sample generalization**.

#### 3.1.2. Model improvements

1. **Hyperparameter tuning** (smart, not exhaustive):

   * Random search or small grid around the already good configuration.
   * Always with **time-based validation** + early stopping.
   * Focus on narrowing learning_rate, num_leaves / max_depth, min_data_in_leaf, regularization.

2. **Model families**:

   * Main hero: gradient boosting (GBM).
   * Backup: regularized linear model (Ridge/Lasso) on a curated feature set; sometimes ensembles of linear + GBM improve robustness.
   * Optional: very simple neural network for global time series (if dataset is large and you still have time).

3. **Ensembles**:

   * Different GBM variants (changed seeds / feature sets) + simple average of predictions.
   * GBM + linear model blend.
   * Keep ensemble **simple** (2–3 models max) to explain easily.

4. **Hierarchical forecasting flavor** (without full-blown reconciliation):

   * Train separate models at entity level and aggregate to region/portfolio.
   * or train a global model and ensure you also **analyze errors** at aggregated levels; mention this analysis as a proxy for hierarchical consistency.

### 3.2. Robustness & generalization (Saeed lead)

To generalize to “greater datasets” and hidden conditions, systematically test robustness:

1. **Alternative validation splits**:

   * Shift your validation window (e.g., last 6 months vs last 12).
   * Ensure performance is stable, not dependent on one lucky split.

2. **Subgroup performance**:

   * Evaluate metric per region, portfolio, entity size.
   * Identify where model underperforms and decide if:

     * You accept it but mention as limitation, or
     * You add targeted features/models.

3. **Sensitivity to data perturbation**:

   * Introduce light noise or randomly remove a small fraction of rows and retrain quickly.
   * Large metric jumps indicate overfitting or fragile pipeline.

4. **Adversarial validation final check**:

   * Re-run train/test classifier.
   * If still high AUC, acknowledge in final story:

     * “We detected structural shift; we tuned our features/models to prioritize stability rather than chasing minimal metric gains.”

5. **Error analysis on worst cases**:

   * Sort entities by validation error.
   * Inspect the top 10 worst; check patterns:

     * Short history?
     * Extreme volatility?
     * Data quality issues?

This level of rigor is rare and **impresses juries** a lot.

### 3.3. Resource allocation logic & decision support

This is your “business edge” – what turns good predictions into a **finance decision tool**.

1. **Define an allocation score** for each entity:

   Example structure:
   [
   \text{score}_i = w_1 \cdot \text{predicted_growth}_i +
   w_2 \cdot \text{predicted_margin}_i -
   w_3 \cdot \text{risk}_i
   ]

   Where:

   * predicted_growth: forecasted % change.
   * predicted_margin or revenue: predicted level.
   * risk: volatility or uncertainty (quantile range).

2. **Rank entities** by this score and show:

   * Top 10 candidates to **increase investment**.
   * Bottom 10 where investment might be **reallocated**.

3. **Scenario simulation** (simple but impressive):

   * Assume: “We can reallocate X% of budget from bottom decile to top decile of score.”
   * Compute a rough **expected impact on portfolio metrics** (e.g., revenue).
   * Present this as: “Under these simplifying assumptions, our allocation policy could yield Y% improvement.”

4. **Portfolio-level view**:

   * For each region or portfolio, show:

     * Total predicted revenue.
     * Suggested investment tilt (more towards high-score segments).

You are not doing full optimization; you are demonstrating **how the model feeds decisions**, which is what Novartis wants.

### 3.4. Explainability & communication

To win, you must explain **what the model learned** in human terms.

1. **Feature importance**:

   * Use model’s built-in importance (gain/cover).
   * Optionally compute SHAP values for a sample of points (if time permits).

2. Translate importances into business language:

   * “Recent growth and recent investment are key predictors of future revenue.”
   * “Regional effects are strong; e.g., portfolio X behaves differently in region Y.”

3. **Case studies**:

   * Pick 2–3 entities (markets/brands) and show:

     * Past vs predicted time series.
     * Which features drove their predictions.
     * How the allocation score ranks them.

4. Be explicit on **limits**:

   * “We do not model macro shocks or regulatory changes.”
   * “For very new brands, limited history reduces accuracy; we advise manual review.”

### 3.5. Final presentation structure (slides)

Your final slides should follow a **tight narrative**:

1. **Title & team** (Team 8 – Arman & Saeed).
2. **Business problem** – in 2–3 bullets.
3. **Data & challenges** – time-series, hierarchy, shift.
4. **Modeling approach** – features, hero model, validation strategy.
5. **Performance** – baseline vs our best model; stability across segments.
6. **Allocation logic** – the scoring formula, ranking, and scenario.
7. **Case studies** – 1 or 2 examples with plots.
8. **Key insights** – actionable patterns for finance.
9. **Limitations & future work** – honest but strategic.
10. **Takeaway** – “We transform raw financial data into a robust, interpretable decision tool for resource allocation.”

Make every chart **simple, labeled, and directly tied** to your story.

---

## How this leads to 1st place & generalizes to greater datasets

* **Generalization** is built in from Phase 1:

  * Time-based validation & adversarial validation.
  * Features driven by process structure (lags, aggregates, life-cycle) – not by test quirks.

* **Phase 2** guarantees a **solid, robust baseline** that would already be competitive in many datathons.

* **Phase 3** adds:

  * Performance gains (better features, tuned ensembles).
  * Robustness checks that prevent overfitting to the competition’s particular split.
  * A **decision-support layer** (allocation logic) that juries love.

Together, this is exactly the pattern you see in **winning teams**:
good metric + strong methodology + clear business story.
