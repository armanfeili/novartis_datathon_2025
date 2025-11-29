Here’s a single, merged, “A–Z” TODO file that supports **all three papers**:

* KG-GCN-LSTM knowledge-graph model
* Supply-chain visibility & advanced demand forecasting
* CNN-LSTM deep temporal model


---

# Novartis Datathon 2025 – Research-Inspired TODO (3 Papers)

> Papers:
>
> 1. **KG-GCN-LSTM** paper (knowledge-graph GCN + LSTM for pharma forecasting).
> 2. **Ghannem et al. (2023)** – “Enhancing Pharmaceutical Supply Chain Resilience: Visibility & Demand Forecasting”.
> 3. **Li et al. (2024)** – “A Deep Learning Algorithm Based on CNN-LSTM for Predicting Cancer Drug Sales Volume”.

Goal: integrate the ideas and models from all three into a coherent pipeline that still respects the current architecture (`BaseModel`, config system, datathon constraints).

---

## A. Global Pre-Flight Checks (repo scan, architecture alignment)

* [ ] Scan the repo for existing implementations of:

  * Graphs / GNNs / KGs: `"gcn"`, `"graph"`, `"adjacency"`, `"knowledge graph"`, `"gnn"`.
  * Temporal / deep models: `"LSTM"`, `"RNN"`, `"Conv1d"`, `"seq2seq"`, `"temporal"`, `"torch"`, `"pytorch"`.
  * External/context features: `"external_features"`, `"events"`, `"holidays"`, `"macroeconomic"`, `"covid"`, `"shock"`, `"scenario"`.
* [ ] Identify:

  * Existing `BaseModel` / `TorchModelWrapper` or similar abstractions.
  * Existing **sequence builder** / panel→sequence utilities.
  * Existing **feature registry** (where we list feature groups).
  * Existing experiment runner / config system.
* [ ] Decide:

  * Where new modules should live (`src/models/`, `src/features/`, `src/graph_utils.py`, `src/external_data.py`, `src/scenario_analysis.py`, etc.).
  * How to hook new models into current CLI / scripts (`train.py`, `inference.py`, etc.).
* [ ] Create or update **one central config entry-point** (Python + YAML) that:

  * Defines model families: `catboost`, `xgboost`, `kg_gcn_lstm`, `cnn_lstm`, any others.
  * Defines feature families: `core_features`, `visibility_features`, `collaboration_features`, `graph_embeddings`, `deep_sequence_features`.
  * Defines evaluation settings (folds, metrics, ablations, scenario tests).

---

## B. Supply-Chain Visibility & Demand-Context Layer (Ghannem et al.)

> Bring in “visibility” & “advanced demand forecast” ideas: more context features, resilience metrics, and scenario analysis.

### B.1 Config: visibility & context as first-class citizens

* [ ] In `src/config.py` or main YAML, add a **visibility/context section**:

  ```python
  VISIBILITY_CONFIG = {
      "enable_external_context": True,
      "sources": {
          "holidays": True,
          "epidemics": False,
          "macro": False,
          "promotions": False,
          "stockout_flags": True,
      },
      "default_lag_window": 24,
      "country_level_calendar": True,
  }
  ```

* [ ] Add feature flags:

  ```yaml
  features:
    use_visibility_features: true
    use_collaboration_features: true
    max_event_lag: 24
  ```

* [ ] Ensure training/inference pipeline reads:

  * `features.use_visibility_features`
  * `features.visibility_sources`
  * `features.max_event_lag`
  * and passes them to `features.py` / `external_data.py`.

---

### B.2 External & contextual data hooks (IoT, epidemics, macro, promotions)

* [ ] Create `src/external_data.py` (if not present) with generic loaders:

  ```python
  def load_holiday_calendar(country, config) -> pd.DataFrame: ...
  def load_epidemic_events(config) -> pd.DataFrame: ...
  def load_macro_indicators(config) -> pd.DataFrame: ...
  def load_promo_or_policy_events(config) -> pd.DataFrame: ...
  ```

  * Each function:

    * Reads data from `data/external/...` if present.
    * Returns an **empty DataFrame** with correct schema if file not found (fails gracefully).

* [ ] Define canonical schemas:

  * Holidays: `["country", "date", "holiday_name", "is_public_holiday"]`.
  * Epidemics: `["country", "date", "event_type", "severity_score"]`.
  * Macro: `["country", "date", "indicator_name", "value"]`.
  * Promotions/Policies: `["country", "date", "event_type", "intensity"]`.

* [ ] Implement a **generic joiner**:

  ```python
  def join_external_context(panel_df, external_tables, config) -> pd.DataFrame:
      """
      For each external table, align by (country, date or months_postgx)
      and add feature columns:
        - holiday flags
        - epidemic severity (current + lags)
        - macro indicators (current + lags)
        - promo/policy flags
      """
  ```

* [ ] Define a single function converting `months_postgx` to calendar `date` if needed:

  * `loi_date + months_postgx` or `loe_date + months_postgx`.
  * Keep it in **one place** so all external joins use the same mapping.

---

### B.3 Visibility-style features in `features.py`

> Encode the paper’s end-to-end visibility ideas as features.

* [ ] In `src/features.py`, add a `make_visibility_features(panel_df, context_df, config)`:

  * Calendar/seasonality:

    * `month_of_year`, `quarter`, `is_flu_season`, `is_summer`, etc.
  * Event proximity:

    * Time since last epidemic or major event.
    * Time until next scheduled/recurring event (if known).
    * Rolling average of `epidemic_severity` over last K months.
  * Shortage/disruption proxies:

    * `vis_zero_volume_flag` (volume suddenly 0 after being >0).
    * `vis_big_drop_flag` (e.g. `vol_t / vol_{t-1} < threshold`).
    * `vis_spike_flag` for sharp increases.
  * Data-quality / visibility indicators:

    * `vis_data_gap_flag` (missing or irregular intervals).
    * `vis_reporting_variance` per brand/country over time.

* [ ] All new columns should be prefixed (e.g. `vis_`) to avoid name collisions.

* [ ] Wire this into the main feature pipeline:

  * If `config.features.use_visibility_features` is true:

    * Load external context tables.
    * Join them with panel.
    * Call `make_visibility_features` and append columns.

---

### B.4 Stakeholder collaboration & “upstream/downstream” aggregation features

> Approximate collaboration between pharmacies, hospitals, manufacturers with hierarchical aggregates.

* [ ] If hierarchical info exists (region, manufacturer, therapeutic area, etc.), add:

  ```python
  def make_collaboration_features(panel_df, config) -> pd.DataFrame:
      """
      Create proxies for information sharing & collaboration:
        - cross-country aggregates per brand
        - therapeutic-class aggregates
        - co-movement indicators
      """
  ```

* [ ] For each `(country, brand, time)` add features like:

  * `collab_peer_avg_vol_t` (avg volume of same brand across other countries).
  * `collab_peer_avg_vol_rolling_K` (rolling window).
  * `collab_country_minus_peer_gap` (country volume – peer average).
  * `collab_ther_class_share` (brand’s share of its therapeutic class volume).

* [ ] Make these features optional via:

  ```yaml
  features:
    use_collaboration_features: true
  ```

* [ ] Ensure integration into:

  * Tabular models (include `collab_*` in feature set).
  * Deep models (include as additional covariates per time step).

---

### B.5 Demand forecasting layer – using visibility & context

> Feed these features into all models (tabular + deep).

* [ ] Update feature selection logic used by:

  * CatBoost, XGBoost, LightGBM.
  * Deep models (CNN-LSTM, KG-GCN-LSTM, any LSTM/RNN).

* [ ] Make sure:

  * `vis_*` and `collab_*` columns are included when `use_visibility_features` / `use_collaboration_features` are true.
  * They can be **explicitly turned off** for ablation experiments.

* [ ] For **deep sequence models**:

  * Extend sequence builder so each time step `x_t` includes:

    * target history (volume),
    * static features,
    * visibility features,
    * collaboration features,
    * scenario flags.

---

### B.6 Resilience & shortage-risk metrics (not only RMSE)

> Align evaluation with supply-chain resilience & shortage risk.

* [ ] In `src/evaluate.py`, add:

  ```python
  def compute_resilience_metrics(y_true, y_pred, panel_df, config) -> dict:
      """
      Key metrics:
        - under-forecast bias on high-volume segments
        - frequency of severe under-forecast
        - series-level stockout-risk proxies
      """
  ```

* [ ] Implement metrics:

  * Under-forecast ratio on top X% volume periods (e.g. `y_pred/y_true` on high-demand).
  * Severe under-forecast frequency (e.g. `y_pred < 0.7 * y_true` over the horizon).
  * For each `(country, brand)`:

    * `res_mean_underforecast_ratio`,
    * `res_underforecast_freq`,
    * a composite `resilience_score`.

* [ ] Allow grouping & reporting by:

  * Scenario (1 vs 2).
  * Country.
  * Brand.
  * Therapeutic area (if available).

* [ ] Add a small report helper:

  ```python
  def summarize_resilience_by_series(...)->pd.DataFrame
  ```

---

### B.7 Scenario analysis & “what-if” simulations

> Handle epidemics, shocks, policy changes as in the review.

* [ ] Create `src/scenario_analysis.py` with:

  ```python
  def apply_demand_shock(panel_df, shock_spec) -> pd.DataFrame:
      """
      shock_spec:
        type: "multiplicative" | "additive"
        target: {"country": ..., "brand": ..., "ther_area": ...}
        start_month: int
        end_month: int
        factor or delta: float
      """
  ```

* [ ] Implement standard scenarios:

  * `epidemic_wave_high`:

    * +X% demand for specific therapeutic areas or countries.
  * `economic_downturn`:

    * −Y% demand for some segments.
  * `supply_disruption`:

    * set volume to 0 (or strongly reduced) for selected `(country, brand)` over a window.

* [ ] Provide a script/CLI `scripts/run_scenarios.py` to:

  * Load baseline panel.
  * Apply defined scenarios.
  * Run forecasts with selected models.
  * Compute both:

    * official datathon metrics,
    * resilience metrics from B.6.
  * Output comparison tables (per scenario, per model).

---

### B.8 Architectural hooks for IoT / blockchain / smart contracts

> Only design the interfaces now; full implementation is out of scope.

* [ ] In `src/visibility_sources.py` define a **generic visibility source interface**:

  ```python
  class VisibilitySource(Protocol):
      def load(self, config) -> pd.DataFrame: ...
      def to_feature_frame(self, panel_df, config) -> pd.DataFrame: ...
  ```

* [ ] Implement a simple example:

  * `CsvVisibilitySource`:

    * loads CSV with simulated IoT readings: `["country", "date", "sensor_type", "value"]`.
    * converts into aggregated features (e.g. avg temperature excursions per month).

* [ ] In config:

  ```yaml
  visibility:
    sources:
      - type: "csv"
        path: "data/external/iot_signals.csv"
      # future: "blockchain", "api", "stream"
  ```

* [ ] Make sure:

  * If no visibility sources are configured, pipeline still runs with existing core features only.

---

### B.9 Experiments & ablations inspired by Ghannem et al.

> Show benefit of visibility & collaboration features.

* [ ] Define experiment configs:

  1. **Baseline**: core features only.
  2. **+Visibility**: baseline + `vis_*`.
  3. **+Visibility+Collab**: add `collab_*`.
  4. **+Full Context**: add external epidemic/macro/promo features if available.

* [ ] For each config:

  * Run standard cross-validation / validation.
  * Compute:

    * official competition metrics,
    * RMSE/MAE,
    * resilience metrics.

* [ ] Save results in `artifacts/visibility_experiments.json` including:

  * Feature flags used.
  * Metrics per model.
  * Basic comments (e.g. `visibility_features_helped_high_erosion_segments`).

---

### B.10 Documentation for visibility & demand context

* [ ] Update `FUNCTIONALITY.md` or `docs/` with a section:

  * “Supply-Chain Visibility & Contextual Demand Forecasting Layer”

    * What data it uses (optional external files).
    * Which features it creates (`vis_*`, `collab_*`).
    * How these features plug into:

      * Tabular models.
      * CNN-LSTM and KG-GCN-LSTM.
    * How resilience metrics and scenario analysis connect to business story (shortages, resilience).

* [ ] Add a short “future work” note:

  * Real IoT ingestion.
  * Blockchain traceability for real supply chains.
  * Smart contracts for automated inventory/ordering decisions.

---

## C. Knowledge-Graph GCN-LSTM Model (KG-GCN-LSTM)

> Build a knowledge-graph-aware GCN-LSTM model and graph-based features.

### C.1 Graph design & node definitions

* [ ] Decide the **graph node definition** for this competition:

  * Option A (simple): node = `brand_name` (global).
  * Option B (richer): node = `(country, brand_name)`.

* [ ] Decide initial **edge construction** inspired by the KG paper:

  * Connect nodes that share:

    * same `ther_area` or ATC class,
    * same `main_package` / dosage form,
    * similar `hospital_rate` bucket,
    * high correlation of pre-LOE volume curves,
    * other domain links (if available).

* [ ] Decide which graph is used where:

  * For KG-GCN-LSTM model (dynamic use during training).
  * For **graph embeddings** reused in tree-based models.

---

### C.2 Graph utilities & node features

* [ ] Create `src/graph_utils.py` with:

  ```python
  def build_drug_graph(panel_df, medicine_info_df, config):
      """
      Returns:
        node_index_df: mapping node_id -> identifiers (brand, country, etc.)
        edge_index: array of (src, dst) edges
        node_features: np.ndarray [n_nodes, n_node_features]
      """
  ```

* [ ] Node features should include (at least):

  * Encoded `ther_area` / ATC or therapeutic class.
  * Encoded `main_package` / formulation.
  * Binned `hospital_rate` (e.g. low/med/high).
  * Simple pre-LOE aggregates:

    * `avg_vol_pre_loe`,
    * growth rate (pre-LOE),
    * volatility (std dev of volume),
    * share of class volume, etc.

* [ ] Provide graph configurability via config:

  ```yaml
  graph:
    node_definition: "brand" | "country_brand"
    similarity_edges:
      use_ther_area: true
      use_package: true
      use_hospital_rate: true
      use_corr_threshold: true
      corr_threshold: 0.7
  ```

* [ ] Add a function to compute **graph embeddings** (for re-use):

  ```python
  def compute_graph_embeddings(node_index_df, edge_index, node_features, config):
      """
      Runs a small GCN (unsupervised or supervised) and returns embedding matrix.
      Saves embeddings to artifacts/graph_embeddings.parquet.
      """
  ```

* [ ] Extend `src/features.py` to join `gcn_emb_*` back to panel:

  * Join on `brand_name` or `(country, brand_name)` consistent with graph design.
  * Prefix columns `gcn_emb_0`, `gcn_emb_1`, …

---

### C.3 GCN implementation (plain PyTorch)

* [ ] Add `src/models/gcn_layers.py` with:

  ```python
  class GCNLayer(nn.Module):
      def __init__(self, in_dim, out_dim, bias=True): ...
      def forward(self, x, edge_index_or_adj): ...
  ```

  * Implement:

    * self-loops: `A_tilde = A + I`.
    * degrees: `D_tilde`.
    * normalized adjacency: `D^(-1/2) A_tilde D^(-1/2)`.
    * propagation: `H_{l+1} = ReLU( A_norm @ H_l @ W_l )`.

* [ ] Add a simple `GCNEncoder(nn.Module)`:

  ```python
  class GCNEncoder(nn.Module):
      def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2): ...
      def forward(self, x, edge_index_or_adj): ...
  ```

* [ ] Support:

  * Dense adjacency or `edge_index` (Copilot can pick whichever is simpler).
  * Configurable `num_layers`, `hidden_dim`, `out_dim`, activation.

---

### C.4 KG-GCN-LSTM model class (PyTorch + BaseModel)

* [ ] Add `src/models/kg_gcn_lstm_model.py`:

  ```python
  class KGGCNLSTMModel(BaseModel):
      def __init__(self, config): ...
      def fit(self, train_df, valid_df=None, sample_weight=None): ...
      def predict(self, test_df): ...
      def save(self, path): ...
      def load(self, path): ...
  ```

* [ ] Internally define `KGGCNLSTMNet(nn.Module)`:

  * Components:

    * `GCNEncoder` to get static node embeddings `Z`.
    * LSTM (or stacked LSTM) to model temporal sequences per node.
    * Dense head to output multi-step horizon predictions.

* [ ] Input / output design:

  * For each node (brand or country-brand):

    * `x_node`: static node features (from `medicine_info_df` + pre-LOE stats + maybe `gcn_emb_*`).
    * `seq`: time series of target and covariates for input window.
  * GCN:

    * Takes all nodes at once: `node_features`, `edge_index`.
    * Produces `Z[node]` for each node.
  * LSTM:

    * For each node sequence, at each time step, input = `[time_step_features, Z[node]]`.
    * Train for scenario 1 & 2:

      * Scenario 1:

        * Input: months -24..-1.
        * Output: months 0..23.
      * Scenario 2:

        * Input: months -24..5.
        * Output: months 6..23.
  * Output: `[batch_size, forecast_horizon]` aligned with datathon target ordering.

---

### C.5 Dataset & sequence builder for KG-GCN-LSTM

* [ ] Extend `src/data.py` / `src/features.py` with:

  ```python
  def build_sequences_for_gcn_lstm(panel_df, scenario, config):
      """
      Returns:
        seq_data: per node sequences (inputs)
        target_data: per node targets (horizon)
        meta: mapping to (country, brand, scenario)
      """
  ```

* [ ] Ensure:

  * Proper sorting by time.
  * Fixed input window lengths (truncate or pad as needed; define policy).
  * Same normalization (scaling/log) as used in other models, with stored scaler params.

* [ ] Implement PyTorch Dataset/DataLoader:

  ```python
  class NodeSequenceDataset(Dataset):
      def __init__(..., sequences, targets, node_ids, extra_covariates): ...
      def __len__(self): ...
      def __getitem__(self, idx): ...
  ```

  * Yields:

    * node index,
    * sequence tensor: `[input_window, num_features]`,
    * target tensor: `[forecast_horizon]`,
    * maybe scenario flag, time covariates.

---

### C.6 Training & inference for KG-GCN-LSTM

* [ ] In `KGGCNLSTMModel.fit(...)`:

  * Build graph (if not precomputed).
  * Build node features and sequences.
  * Initialize `KGGCNLSTMNet` with config.
  * Use optimizer (Adam), LR, weight decay from config.
  * Loss: MSE (optionally with sample weights).
  * Early stopping on validation RMSE / official metric.
  * Gradient clipping (configurable).
  * Save best checkpoint.

* [ ] In `predict(...)`:

  * Rebuild or load graph & node features (ensure consistency).
  * Build sequences from test/inference panel.
  * Run model to obtain predictions.
  * Map predictions back to a DataFrame with:

    * `country`, `brand_name`, `months_postgx`, `y_hat` (or whatever target column is called).

* [ ] Implement `save(...)`/`load(...)`:

  * Save:

    * model state dict,
    * config,
    * node_index mapping,
    * scalers/normalizers,
    * any graph metadata necessary.
  * Load:

    * reconstruct net with config,
    * load weights,
    * restore mappings & scalers.

---

### C.7 Config file for KG-GCN-LSTM

* [ ] Add `configs/model_kg_gcn_lstm.yaml`:

  ```yaml
  model:
    name: "kg_gcn_lstm"
    task: "regression"
    priority: 3

  graph:
    node_definition: "country_brand"
    hidden_dim: 64
    out_dim: 64
    num_layers: 2

  lstm:
    hidden_size: 64
    num_layers: 1
    dropout: 0.2
    bidirectional: false

  training:
    epochs: 100
    batch_size: 64
    learning_rate: 0.001
    weight_decay: 0.0001
    use_early_stopping: true
    eval_metric: "rmse"
    use_sample_weights: true

  gpu:
    enabled: true
    device_id: 0
  ```

* [ ] Include sweep configuration:

  ```yaml
  sweep:
    enabled: false
    selection_metric: "official_metric"
    n_folds: 3
    sweep_configs:
      - id: "kg_default"
        params:
          graph.hidden_dim: 64
          lstm.hidden_size: 64
      - id: "kg_small"
        params:
          graph.hidden_dim: 32
          lstm.hidden_size: 32
  ```

---

### C.8 Model factory integration & ablations (from KG paper)

* [ ] In `src/models/__init__.py` (or equivalent):

  * Register `"kg_gcn_lstm"` → `KGGCNLSTMModel`.
  * Add aliases such as `"gcn_lstm"`, `"kg_gcn"` if helpful.

* [ ] Update training/inference entrypoints:

  * `train.py` should accept `model_type="kg_gcn_lstm"`.
  * `inference.py` should be able to load and run KG-GCN-LSTM, producing submissions.

* [ ] Implement **ablations**:

  1. LSTM-only (same sequences, no graph):

     * Either a separate `LSTMModel`, or `KGGCNLSTMNet` with GCN disabled.
  2. GCN-MLP (node embeddings + shallow MLP on aggregated history).
  3. GCN-LSTM with **only self-loop graph** (no real KG edges).

* [ ] Add an evaluation helper to run these variants and store results in:

  * `artifacts/ablation_kg_gcn_lstm.json` with:

    * model name,
    * hyperparameters,
    * official metric1/2,
    * RMSE / MAE.

---

### C.9 Ensembles with KG-GCN-LSTM

* [ ] Ensure ensemble components (if any) can include `"kg_gcn_lstm"`:

  ```yaml
  ensemble:
    include: true
    members:
      - "catboost"
      - "xgboost"
      - "kg_gcn_lstm"
    weight_optimization: true
  ```

* [ ] Make sure predictions from KG-GCN-LSTM align in shape & index with other models for blending.

---

### C.10 Documentation for KG-GCN-LSTM

* [ ] Add to `docs/models.md`:

  * “KG-GCN-LSTM (Knowledge-Graph GCN-LSTM)”

    * Motivation.
    * Architecture (GCN over graph of drugs → node embeddings → LSTM on sequences → horizon).
    * Where code lives.
    * How to enable/configure it.
    * How to run ablations.

---

## D. CNN-LSTM Deep Temporal Model (Li et al.)

> Implement and integrate a hybrid CNN-LSTM model for time-series forecasting.

### D.1 Repo scan & reuse

* [ ] Confirm whether there’s an existing PyTorch temporal model:

  * If yes, reuse its base class/training loop.
  * If there is already a `TorchModelWrapper`, plug into that.

---

### D.2 CNN-LSTM config

* [ ] Add a `CNN_LSTM_CONFIG` section in `src/config.py` or a YAML:

  ```python
  CNN_LSTM_CONFIG = {
      "enabled": False,
      "input_window": 24,
      "forecast_horizon": 24,   # adjust to datathon horizon
      "use_multivariate_features": True,
      "cnn": {
          "num_blocks": 2,
          "filters": [64, 128],
          "kernel_sizes": [3, 5],
          "use_batchnorm": True,
          "activation": "relu",
          "pool_size": 2,
      },
      "lstm": {
          "num_layers": 2,
          "hidden_size": 128,
          "dropout": 0.3,
          "stateful": False,
      },
      "optimizer": {
          "type": "adam",
          "lr": 1e-3,
          "weight_decay": 0.0,
      },
      "training": {
          "batch_size": 64,
          "max_epochs": 100,
          "early_stopping_patience": 10,
          "gradient_clip": 1.0,
      },
  }
  ```

* [ ] Add `MODELS["cnn_lstm"]["enabled"]` toggle and integrate into model selection.

---

### D.3 Sequence builder for CNN-LSTM

* [ ] Add or extend `src/sequence_builder.py`:

  ```python
  def build_cnn_lstm_sequences(
      panel_df,
      config,
      id_cols=("country", "brand_name"),
      target_col="normalized_volume",
      feature_cols=None,
  ):
      """
      Returns:
        X: [num_samples, input_window, num_features]
        y: [num_samples, forecast_horizon]
        meta: metadata for mapping predictions back
      """
  ```

* [ ] For each `(country, brand)`:

  * Sort by time index (`months_postgx` or date).
  * Create sliding windows:

    * Input window: last `input_window` months.
    * Target: next `forecast_horizon` months (consistent with scenario/horizon).
  * Use `feature_cols` including:

    * Target history.
    * Price.
    * Generics count/share.
    * Hospital vs retail share.
    * LOE features.
    * Scenario flags.
    * Visibility & collaboration features (if enabled).

* [ ] Handle gaps:

  * Decide policy (drop incomplete windows, or pad with NaNs then impute).
  * Log when dropping series due to insufficient history.

---

### D.4 CNN-LSTM model implementation

* [ ] Create `src/models/cnn_lstm.py`:

  ```python
  class CNNLSTMNet(nn.Module):
      def __init__(self, config, num_features): ...
      def forward(self, x): ...
  ```

  * Input `x` shape for PyTorch Conv1d: `[batch_size, num_features, seq_len]`.
  * CNN blocks:

    * Conv1d(kernel_size=3, out_channels=64) → BatchNorm → ReLU → MaxPool1d(2).
    * Conv1d(kernel_size=5, out_channels=128) → BatchNorm → ReLU → MaxPool1d(2).
  * After last pooling, transpose to `[batch_size, seq_len', channels']` for LSTM.

* [ ] LSTM:

  * `num_layers=2`, `hidden_size=128`, `batch_first=True`, dropout=0.3.
  * Optionally stateful if needed later.

* [ ] Output:

  * Use last time step output (or aggregated) → Dense layer → `[batch_size, forecast_horizon]`.

* [ ] Wrap into `CNNLSTMModel(BaseModel)`:

  ```python
  class CNNLSTMModel(BaseModel):
      def fit(...): ...
      def predict(...): ...
      def save(...): ...
      def load(...): ...
  ```

---

### D.5 Training wrapper & integration

* [ ] Add a training function, e.g. in `src/train.py` or `src/models/training_utils.py`:

  ```python
  def train_cnn_lstm_model(train_sequences, val_sequences, config, experiment_name): ...
  ```

  * Instantiate `CNNLSTMNet`.
  * Optimizer from config.
  * Loss = MSE (optionally MAE as secondary).
  * Early stopping on validation MSE/RMSE.
  * Gradient clipping.
  * Checkpoint best model.

* [ ] Modify main training entrypoint:

  * If `MODELS["cnn_lstm"]["enabled"]`:

    * Build sequences.
    * Call `train_cnn_lstm_model`.

* [ ] Save:

  * model state dict,
  * config,
  * scalers/encoders,
  * mapping meta (id_cols & time indices).

---

### D.6 Inference & submission mapping for CNN-LSTM

* [ ] In `src/inference.py`, add:

  ```python
  def run_cnn_lstm_inference(panel_df, model, config) -> pd.DataFrame:
      """
      Build sequences for inference horizon, run model, map back to submission format.
      """
  ```

* [ ] Requirements:

  * Use same feature_cols, scaling, window length as training.
  * For each prediction, map back to:

    * `(country, brand_name, months_postgx, scenario_id)` + predicted target.
  * Integrate with `make_submission.py` or equivalent.

---

### D.7 Metrics & model comparison (Li et al. style)

* [ ] In `src/evaluate.py`:

  * Ensure we have:

    ```python
    def compute_mse(y_true, y_pred): ...
    def compute_rmse(y_true, y_pred): ...
    ```

* [ ] For CNN-LSTM:

  * Log train/val MSE & RMSE per epoch.
  * After training, compute final MSE/RMSE on validation/test.

* [ ] Add a small comparison utility:

  * Compare CNN-LSTM with:

    * LSTM only,
    * CNN only,
    * Plain RNN (if implemented),
    * Tabular baselines (CatBoost).
  * Save to `artifacts/model_comparison_cnn_lstm.json`.

---

### D.8 Baseline LSTM / CNN models (ablations)

* [ ] If not present, implement:

  * `LSTMModel(BaseModel)`:

    * Similar architecture but without CNN.
  * `CNNOnlyModel(BaseModel)`:

    * Conv1d + pooling + flatten + Dense for regression.

* [ ] Use same sequence builder (`build_cnn_lstm_sequences`) to ensure apples-to-apples comparison.

* [ ] Include these in experiments and comparisons (Li et al. table style).

---

### D.9 Feature engineering for multivariate CNN-LSTM

* [ ] Decide the feature set for deep models:

  * Core features:

    * `normalized_volume`,
    * price/net price,
    * generics count/share,
    * LOE features (`months_postgx`, `post_loe_flag` etc.),
    * scenario flag,
    * brand/country encodings if needed.
  * Contextual:

    * `vis_*` visibility features,
    * `collab_*` collaboration features.
  * Graph:

    * `gcn_emb_*` graph embeddings (optional).

* [ ] Ensure all features are numeric:

  * Use same encoders/scalers as for tabular models where sensible.
  * For deep models, typically apply per-feature scaling (`StandardScaler` or `MinMaxScaler`) and keep scaler parameters saved.

---

### D.10 CNN-LSTM experiments

* [ ] Define small config grid:

  * `input_window`: {12, 18, 24}.
  * `lstm.hidden_size`: {64, 128}.
  * `use_visibility_features`: {true, false}.
  * `use_gcn_embeddings`: {false, true (later)}.

* [ ] For each experiment:

  * Run standard validation.
  * Compute:

    * official metrics,
    * RMSE/MSE,
    * resilience metrics if available.

* [ ] Summarize into `artifacts/cnn_lstm_experiments.json` with comments about:

  * Overfitting,
  * Benefit on volatile post-LOE periods,
  * Comparison with CatBoost.

---

### D.11 Documentation for CNN-LSTM

* [ ] Update docs:

  * Section “CNN-LSTM Deep Temporal Model (Li et al.)”:

    * Architecture (CNN for local patterns, LSTM for long-term trends).
    * Which features it uses.
    * How to enable it.
    * How to interpret evaluation outputs (MSE, RMSE, official metrics).

---

## E. Cross-Model & Cross-Paper Integration

* [ ] Ensure **visibility features** and **graph embeddings** are available to:

  * Tabular models.
  * CNN-LSTM.
  * KG-GCN-LSTM (as node features or covariates).

* [ ] Ensure **scenario analysis** module can:

  * Run any model (or ensemble) under multiple scenarios.
  * Compare resilience metrics across:

    * CatBoost baseline,
    * CNN-LSTM,
    * KG-GCN-LSTM,
    * Ensembles that combine them.

* [ ] Add a high-level design note (`docs/research_integration.md`) explaining:

  * How the three papers map to:

    * Graph structure & KG-GCN-LSTM,
    * Visibility & resilience features,
    * CNN-LSTM sequence modeling.
  * How all of this ties back to the **generic erosion forecasting** business story.

---
