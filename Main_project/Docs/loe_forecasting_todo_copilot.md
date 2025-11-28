# ✅ LOE Forecasting – To-Do List for Copilot Agent

Goal: Implement a Situation-Room-style, ROI-aware, erosion-focused forecasting system for the Datathon project.

---

## 1. Build the “LOE Situation Room” Concept

- [ ] Create a notebook or simple app: `loe_situation_room.ipynb`.
- [ ] Add UI/plots for a selected brand:
  - [ ] Plot historical volume (pre- and post-generic entry).
  - [ ] Plot `n_gxs` (number of generics) over time.
  - [ ] Plot model forecast vs actuals as more months become available.
- [ ] Add simple scenario controls (sliders, dropdowns) for:
  - [ ] Defense strength (e.g. “strong defense”, “weak defense”).
  - [ ] Different assumed erosion shapes (steep vs gradual).
- [ ] Document this as the “LOE Situation Room” for the report/slides.

---

## 2. Implement the Continuous Forecasting Loop (“Forecasting Flywheel”)

- [ ] Implement a rolling simulation for each brand:
  - [ ] At month 0 (LoE): train only on pre-entry data and forecast 0–23 (Scenario 1).
  - [ ] At month 6: update model with months 0–5 and forecast 6–23 (Scenario 2).
  - [ ] (Optional) Repeat at months 12, 18 for extended experiments.
- [ ] Track metrics at each stage:
  - [ ] Store Prediction Error at LoE, month 6, etc.
  - [ ] Compare how error shrinks as more real data is available.
- [ ] Visualize the loop:
  - [ ] Timeline plot: data available vs forecast quality.
  - [ ] Add text/diagram mapping: Data → Model → Strategy → Execution → Outcome → Data.

---

## 3. Make `n_gxs` (Number of Competitors) Central in the Model

- [ ] Check datasets for `n_gxs` and related generics information.
- [ ] Engineer features based on `n_gxs`:
  - [ ] Current number of generics (`n_gxs_t`).
  - [ ] Maximum generics seen so far (`max_n_gxs_post`).
  - [ ] Month-on-month change (`delta_n_gxs`).
  - [ ] Flags for “generic flood” events (e.g. jump of ≥2 in one month).
- [ ] Train models **with** and **without** `n_gxs` features:
  - [ ] Compare Prediction Error and bucket-wise performance.
  - [ ] Generate plots showing improvement when `n_gxs` is included.
- [ ] Optionally segment models by competition level:
  - [ ] Low competition: `n_gxs` ≤ 2.
  - [ ] High competition: `n_gxs` ≥ 3.
- [ ] Add a short analysis cell describing `n_gxs` as the “master variable”.

---

## 4. Add an ROI-Oriented Evaluation Layer

### 4.1 Revenue-Error Approximation

- [ ] Assume a simple price per unit for each brand (e.g. constant or proxy).
- [ ] Compute true revenue over 24 months:
  - [ ] `R_true = sum(volume_t * price_t)`.
- [ ] Compute forecasted revenue using predicted volumes:
  - [ ] `R_pred = sum(pred_volume_t * price_t)`.
- [ ] Define revenue error metrics:
  - [ ] Absolute revenue error |R_true − R_pred|.
  - [ ] Percentage revenue error.
- [ ] Compare:
  - [ ] Baseline model revenue error vs advanced model revenue error.
  - [ ] Summaries per bucket (Bucket 1 vs 2).

### 4.2 Simple Lifecycle Management (LCM) ROI Example

- [ ] Select one representative brand.
- [ ] Compute baseline 5-year revenue from the forecast (no LCM change).
- [ ] Define a hypothetical LCM tactic:
  - [ ] E.g. flatter erosion curve (increase predicted volumes by a factor in months 6–23).
- [ ] Recompute projected revenue under the “flatter” scenario.
- [ ] Define an investment amount `Y` (e.g. “cost of new formulation”).
- [ ] Compute:
  - [ ] `X = baseline revenue`.
  - [ ] `Z = LCM scenario revenue`.
  - [ ] Check if `Z − X > Y`.
- [ ] Add a markdown explanation: how accurate forecasts enable rational LCM ROI decisions.

---

## 5. Design the “Integrated Capability” Architecture

- [ ] Add a section (markdown or slide-ready) describing 4 layers:

  ### 5.1 Data Layer
  - [ ] List internal data sources: volumes, generics, product attributes.
  - [ ] Mention potential external sources (IQVIA, DrugPatentWatch, claims data).

  ### 5.2 Model Layer
  - [ ] List chosen models (e.g. gradient boosting, ARHOW, LSTM, etc.).
  - [ ] Explain separate pipelines for Scenario 1 and Scenario 2.
  - [ ] Mention erosion buckets (Bucket 1 emphasis) explicitly.

  ### 5.3 Application Layer – “Situation Room”
  - [ ] Describe how business users (Brand, Finance, Market Access) would use:
    - [ ] Situation Room dashboard.
    - [ ] Scenario analysis (different `n_gxs`, different tactics).
  - [ ] Show examples of alerts/outputs:
    - [ ] Earlier-than-expected extra generic entrant.
    - [ ] Faster-than-expected erosion in a high-value brand.

  ### 5.4 Governance & Process
  - [ ] Specify a notional forecast review cadence (e.g. monthly).
  - [ ] Document which roles would participate (Brand, Market Access, Analytics, Supply, Legal).
  - [ ] Include how updated data automatically retriggers model updates.

---

## 6. Create a “Forecasting Flywheel” Slide/Section

- [ ] Add a markdown diagram or slide content titled **“Our LOE Forecasting Flywheel”**.
- [ ] Represent the loop:
  - [ ] Data → Model → Strategy → Execution → Outcome → Data.
- [ ] Under each node, map actual project components:
  - [ ] Data: 3 core datasets + features.
  - [ ] Model: Scenario 1/2 models + erosion buckets.
  - [ ] Strategy: what-if scenarios, segmentation by `n_gxs`.
  - [ ] Execution: choice of forecast strategy, error thresholds, tactics.
  - [ ] Outcome: realized future months, new metrics, backtest results.
- [ ] Link this slide/section in final presentation/report structure.

---

## 7. Documentation & Presentation Integration

- [ ] Summarize the Situation Room, Flywheel, and ROI logic in `reports/final_report.md`.
- [ ] Add a “Strategic Value of Accurate LOE Forecasting” section:
  - [ ] High cost of being wrong (tie to your revenue error).
  - [ ] Support for LCM decisions.
  - [ ] Support for portfolio and M&A planning (conceptually).
- [ ] Ensure final slides:
  - [ ] Include the flywheel diagram.
  - [ ] Show at least one ROI example.
  - [ ] Highlight `n_gxs` as the key explanatory driver.

---
