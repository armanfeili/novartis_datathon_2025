# Novartis Datathon 2025: Complete Slide Presentation Documentation

## Comprehensive Technical Reference Guide

---

## Table of Contents

2. [Drug Lifecycle and Market Dynamics](#2-drug-lifecycle-and-market-dynamics)
   - 2.1 [Complete Drug Lifecycle Phases](#21-complete-drug-lifecycle-phases)
   - 2.2 [Key Commercial Milestones](#22-key-commercial-milestones)
   - 2.3 [The Datathon Focus Window](#23-the-datathon-focus-window)
3. [Generic Erosion: Concept and Quantification](#3-generic-erosion-concept-and-quantification)
   - 3.1 [Mean Generic Erosion Definition](#31-mean-generic-erosion-definition)
   - 3.2 [Mathematical Formulation](#32-mathematical-formulation)
   - 3.3 [Erosion Visualization Profiles](#33-erosion-visualization-profiles)
   - 3.4 [Erosion Classification and Buckets](#34-erosion-classification-and-buckets)
4. [The Datathon Challenge](#4-the-datathon-challenge)
   - 4.1 [Data Science Challenge](#41-data-science-challenge)
   - 4.2 [Business Challenge](#42-business-challenge)
   - 4.3 [Winner Selection Process](#43-winner-selection-process)
5. [Data Description](#5-data-description)
   - 5.1 [Dataset Overview and Structure](#51-dataset-overview-and-structure)
   - 5.2 [Volume Dataset (df_volume.csv)](#52-volume-dataset-df_volumecsv)
   - 5.3 [Generics Dataset (df_generics.csv)](#53-generics-dataset-df_genericscsv)
   - 5.4 [Medicine Information Dataset (df_medicine_info.csv)](#54-medicine-information-dataset-df_medicine_infocsv)
   - 5.5 [Erosion Buckets Distribution](#55-erosion-buckets-distribution)
6. [Evaluation Metrics](#6-evaluation-metrics)
   - 6.1 [Phase 1-a: Scenario 1 Prediction Error](#61-phase-1-a-scenario-1-prediction-error)
   - 6.2 [Phase 1-b: Scenario 2 Prediction Error](#62-phase-1-b-scenario-2-prediction-error)
   - 6.3 [Final Score Aggregation](#63-final-score-aggregation)
   - 6.4 [Metric Interpretation Guide](#64-metric-interpretation-guide)
7. [Summary and Key Takeaways](#7-summary-and-key-takeaways)

---

## 2. Drug Lifecycle and Market Dynamics

### 2.1 Complete Drug Lifecycle Phases

A pharmaceutical product progresses through distinct phases from market introduction to eventual sales decline:

```
Drug Launch → Growth → Maturity → Competition → Loss of Exclusivity → Generic Erosion
```

| Phase | Characteristics | Sales Behavior |
|-------|-----------------|----------------|
| **Drug Launch** | Market introduction; low awareness and adoption | Sales low but rising |
| **Growth** | Increased prescriber awareness; market acceptance | Rapid sales increase |
| **New Indication Entry** | Additional disease/patient group approvals | Accelerated growth; upward curve |
| **Maturity** | Market saturated; established reimbursement | Sales stabilize at peak |
| **New Competitor Entry** | Other branded products enter same therapeutic area | Growth slows; curve flattens |
| **Loss of Exclusivity (LoE)** | Patent protection expires | Peak reached; decline begins |
| **Generics Entry (Gx)** | Generic competitors launch | Sharp, rapid volume decline |
| **Generic Erosion** | Post-generic stabilization | Low residual sales |

### 2.2 Key Commercial Milestones

#### Loss of Exclusivity (LoE)
- **Definition:** Expiration of legal protections (patents, data exclusivity) preventing generic copying
- **Timing:** Occurs near or at the peak of the sales curve
- **Impact:** Allows generic manufacturers to launch cheaper copies

#### Generics Entry (Gx)
- **Definition:** The moment generic drugs actually enter the market
- **Timing:** Typically very close to, or just after, LoE
- **Impact:** Triggers rapid decline in originator brand's sales volume

### 2.3 The Datathon Focus Window

The Datathon specifically targets the **post-LoE period**:

- Time window: From generic entry date through the subsequent 24 months
- Focus areas:
  - Forecasting the volume decline
  - Understanding price and volume erosion
  - Optimizing strategies during and after generics entry
  
**The challenge is NOT about the entire drug lifecycle, but specifically about the critical phase when generics enter and sales collapse.**

---

## 3. Generic Erosion: Concept and Quantification

### 3.1 Mean Generic Erosion Definition

**Generic erosion** quantifies how much a branded drug's sales fall after generic competitors enter the market.

**Conceptual Definition:**
> Mean Generic Erosion is defined as the mean of the normalized volumes after generic entry, considering a 24-month horizon. Volumes after generic entry are normalized by the average monthly volume of the last 12 months before generic entry.

### 3.2 Mathematical Formulation

#### Formula 1: Mean Generic Erosion

$$\text{Mean Generic Erosion} = \frac{\sum_{i=0}^{23} \text{Vol}_{\text{norm},i}}{24}$$

**Where:**
- $i$ = month index (0 = entry month, 23 = 24th month after entry)
- $\text{Vol}_{\text{norm},i}$ = normalized volume in month $i$

#### Formula 2: Normalized Volume

$$\text{Vol}_{\text{norm},i} = \frac{\text{Vol}_i}{\text{Avg}_j}$$

**Where:**
- $\text{Vol}_i$ = actual sales volume in month $i$ after generic entry
- $\text{Avg}_j$ = reference average volume for product/market $j$

**Interpretation:**
| $\text{Vol}_{\text{norm},i}$ Value | Meaning |
|-----------------------------------|---------|
| 1.0 | Current month equals pre-generic average |
| 0.5 | Volume is 50% of pre-generic average |
| 2.0 | Volume is double pre-generic average (rare) |

#### Formula 3: Pre-Generic Reference Average

$$\text{Avg}_j = \frac{\sum_{i=-12}^{-1} Y^{\text{act}}_{j,i}}{12}$$

**Where:**
- $j$ = drug, country, or market index
- $Y^{\text{act}}_{j,i}$ = actual observed volume for drug/market $j$ in month $i$
- $i = -12$ to $-1$ = the 12 months before generic entry

**Interpretation of Mean Generic Erosion Values:**

| Value | Interpretation |
|-------|----------------|
| ≈ 1 | Little erosion; post-generic volume similar to pre-generic |
| < 1 | Sales dropped; e.g., 0.4 means 40% of pre-generic level |
| ≈ 0 | Near-total sales collapse |
| > 1 | Volume increased (unusual) |

### 3.3 Erosion Visualization Profiles

Three typical volume erosion patterns exist after generic entry:

#### High Erosion (Red Curve)
- **Pre-entry:** Volume increases modestly
- **Post-entry:** Volume collapses almost vertically
- **Long-term:** Very low residual level near zero
- **Characteristics:** Steepest and deepest decline

#### Medium Erosion (Yellow Curve)
- **Pre-entry:** Highest and stable volume
- **Post-entry:** Sharp fall but not to zero
- **Long-term:** Continues gradual decline over time
- **Characteristics:** Significant but not complete erosion

#### Low Erosion (Blue Curve)
- **Pre-entry:** Moderate, slightly decreasing volume
- **Post-entry:** Slow, smooth decline
- **Long-term:** Remains highest among the three profiles
- **Characteristics:** Brand retains substantial market share

### 3.4 Erosion Classification and Buckets

#### Conceptual Categories

| Category | Mean Erosion Range | Description |
|----------|-------------------|-------------|
| **Low Erosion** | Close to 1 | Volume remains relatively stable |
| **Medium Erosion** | Between 0 and 1 | Moderate decline in sales |
| **High Erosion** | Close to 0 | Sharp drop in volume |

#### Datathon Bucket System

For the Datathon, drugs are classified into **two operational buckets**:

| Bucket | Mean Erosion Range | Description | Focus Level |
|--------|-------------------|-------------|-------------|
| **Bucket 1 (B1)** | [0, 0.25] | High erosion—loses ≥75% of pre-generic sales | **Primary Datathon Focus** |
| **Bucket 2 (B2)** | (0.25, 1] | Medium/low erosion—retains >25% of pre-generic sales | Secondary |

**Interval Notation Explanation:**
- `[0, 0.25]`: Both 0 and 0.25 are **included** (square brackets)
- `(0.25, 1]`: 0.25 is **excluded** (round bracket), 1 is **included** (square bracket)

**Datathon Focus:**
> The Datathon is specifically about the most severe erosion scenarios (Bucket 1), while Bucket 2 provides background context.

---

## 4. The Datathon Challenge

### 4.1 Data Science Challenge

#### Core Objective
Forecast the volume erosion following generic entry over a **24-month horizon** from the generic entry date.

#### Two Forecasting Scenarios

| Scenario | Timing | Available Data | Forecast Horizon | Description |
|----------|--------|----------------|------------------|-------------|
| **Scenario 1** | Right after generic entry | Pre-generic history only; no post-entry actuals | Month 0 to Month 23 (24 months) | Pure ex-ante prediction |
| **Scenario 2** | 6 months after generic entry | Pre-generic history + 6 months post-entry actuals | Month 6 to Month 23 (18 months) | Mid-course forecast update |

#### Scenario 1: Zero-Knowledge Forecast
- **Decision point:** Immediately at generic entry (month 0)
- **Available information:** Only historical pre-generic data and static features (country, molecule, competition, etc.)
- **Challenge:** Hardest forecasting setting—predict entire erosion trajectory without seeing any post-entry behavior
- **Business context:** Planning at LoE before seeing market reaction

#### Scenario 2: Six-Month Update
- **Decision point:** Six months after generic entry
- **Available information:** Pre-generic history plus 6 realized post-generic months (months 0-5)
- **Challenge:** Use early erosion pattern to refine predictions for remaining 18 months
- **Business context:** Mid-course update with observed market data

### 4.2 Business Challenge

Beyond technical modeling, participants must demonstrate business understanding:

#### Required Deliverables

| Requirement | Description |
|-------------|-------------|
| **Deep Exploratory Analysis** | Thoroughly explore the dataset: distributions, correlations, trends, differences across countries/molecules/therapy areas |
| **Preprocessing Documentation** | Explain and justify data cleaning and transformation choices: missing values handling, outlier treatment, normalization, feature engineering |
| **High-Erosion Focus** | Analyze characteristics of Bucket 1 markets: specific countries, therapeutic areas, payer types, competitive situations |
| **Visualization Tools** | Present time-series plots, comparative charts, feature importance, geographic/categorical breakdowns |

#### Evaluation Criteria
- **Not just how** you build the model (algorithms, features, metrics)
- **But also why** you choose approaches in terms of:
  - Business value
  - Interpretability
  - Practicality

### 4.3 Winner Selection Process

#### Phase 1: Model Evaluation (Quantitative)

| Step | Scope | Teams Evaluated | Advancement Criteria |
|------|-------|-----------------|---------------------|
| **Phase 1-a** | Scenario 1 predictions | All teams | Top 10 teams with lowest prediction error |
| **Phase 1-b** | Scenario 2 predictions | Top 10 from Phase 1-a | Top 5 teams with lowest prediction error |

**Key Points:**
- All teams must submit predictions for the **entire test dataset** (both scenarios)
- Phase 1-a filters based on Scenario 1 accuracy
- Phase 1-b filters based on Scenario 2 accuracy
- Only teams excelling in **both scenarios** advance to finals

#### Phase 2: Jury Evaluation (Qualitative)

**Participants:** Top 5 teams from Phase 1

**Presentation Requirements:**
- Methodology (data preprocessing, feature engineering, modeling choices, validation strategy)
- Insights (learnings from data, especially about high-erosion markets)
- Conclusions (business implications, recommendations, limitations)

**Jury Composition:**
- Technical experts (data scientists, statisticians)
- Business experts (finance, commercial, market access professionals)

**Outcome:** Jury selects **top 3 winning teams** based on:
- Technical robustness
- Interpretability
- Business relevance

---

## 5. Data Description

### 5.1 Dataset Overview and Structure

#### Target Variable
- **Monthly volume** for **2,293 country–brand combinations** that experienced generic entry
- Each row represents a specific **country + brand + month** combination

#### Train-Test Split

| Dataset | Observations | Description |
|---------|--------------|-------------|
| **Training Set** | 1,953 | Up to 24 months before and after generic entry |
| **Test Set** | 340 | Split between Scenario 1 and Scenario 2 |

#### Test Set Distribution

| Scenario | Observations | Percentage | Forecast Requirement |
|----------|--------------|------------|---------------------|
| **Scenario 1** | 228 | ~67% | Months 0–23 (full 24 months) |
| **Scenario 2** | 112 | ~33% | Months 6–23 (18 months) |

### 5.2 Volume Dataset (df_volume.csv)

The core time-series dataset containing monthly sales volumes around generic entry.

#### Schema

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market identifier (anonymized, e.g., `COUNTRY_B6AE`) |
| `brand_name` | String | Brand identifier (anonymized, e.g., `BRAND_1C1E`) |
| `month` | String | Calendar month name (e.g., `Jul`, `Aug`) |
| `months_postgx` | Integer | Months relative to generic entry |
| `volume` | Float | **Target variable** — Number of units sold |

#### `months_postgx` Interpretation

| Value | Meaning |
|-------|---------|
| 0 | Month of generic entry |
| Negative (e.g., -3) | Months before generic entry |
| Positive (e.g., 6) | Months after generic entry |

#### Example Data

| country | brand_name | month | months_postgx | volume |
|---------|------------|-------|---------------|--------|
| COUNTRY_B6AE | BRAND_1C1E | Jul | -24 | 272594.39 |
| COUNTRY_B6AE | BRAND_1C1E | Aug | -23 | 351859.31 |
| COUNTRY_B6AE | BRAND_1C1E | Sep | -22 | 447953.48 |

**Key Uses:**
- Reconstruct volume trajectory for each country-brand pair
- Identify pre-generic and post-generic periods
- Feed time-series and panel models for forecasting

### 5.3 Generics Dataset (df_generics.csv)

Contains **time-varying information on generic competitors** for each brand and country.

#### Schema

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market identifier |
| `brand_name` | String | Brand identifier |
| `months_postgx` | Integer | Months after generic entry (starts at 0) |
| `n_gxs` | Integer/Float | Number of generic competitors at that time |

**Note:** `n_gxs` varies over time as generics enter or exit the market.

#### Example Data

| country | brand_name | months_postgx | n_gxs |
|---------|------------|---------------|-------|
| COUNTRY_B6AE | BRAND_DF2E | 0 | 0.0 |
| COUNTRY_B6AE | BRAND_DF2E | 1 | 0.0 |
| COUNTRY_B6AE | BRAND_DF2E | 2 | 1.0 |
| COUNTRY_B6AE | BRAND_DF2E | 3 | 2.0 |

**Example Interpretation:**
- Months 0-1: No active generic competitors (regulatory/commercial delays)
- Month 2: First generic appears
- Month 3+: Second generic joins, competition intensifies

#### Modeling Applications
- Join with `df_volume.csv` on `(country, brand_name, months_postgx)`
- Use `n_gxs` directly as feature
- Derive additional variables:
  - Binary indicator: "any generics present" (`n_gxs > 0`)
  - Cumulative months since first generic
  - Month-over-month change in `n_gxs`

### 5.4 Medicine Information Dataset (df_medicine_info.csv)

Contains **static product-level attributes** for each country-brand combination.

#### Schema

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market identifier |
| `brand_name` | String | Brand identifier |
| `ther_area` | String | Therapeutic area (e.g., `Sensory_organs`, `Nervous_system`) |
| `hospital_rate` | Float | Percentage of drug delivered in hospitals (0-100) |
| `main_package` | String | Most common dispensing format (e.g., `PILL`, `INJECTION`, `EYE DROP`) |
| `biological` | Boolean | Whether drug is derived from living organism |
| `small_molecule` | Boolean | Whether drug is a low molecular weight compound |

#### Column Details

**`ther_area` (Therapeutic Area):**
- Indicates clinical indication category
- Examples: `Sensory_organs`, `Muscoskeletal_Rheumatology`, `Antineoplastic`, `Nervous_system`
- Different areas show different erosion patterns due to clinical need, prescribing habits, and reimbursement rules

**`hospital_rate`:**
- Range: 0 to 100
- High values (e.g., 92%) indicate predominantly hospital-based distribution
- Low values (e.g., 0.09%) indicate mostly retail distribution
- High hospital share may indicate tender-driven procurement and stepwise erosion patterns

**`main_package`:**
- Categorical description of dosage/dispensing format
- Examples: `PILL`, `INJECTION`, `EYE DROP`
- Affects patient convenience, generic competition intensity, and price differentials

**`biological`:**
- `True`: Drug derived from living organism (proteins, antibodies, nucleic acids)
- `False`: Not a biologic
- Biologics face biosimilar competition with typically slower/different erosion patterns

**`small_molecule`:**
- `True`: Low molecular weight, chemically synthesized compound
- `False`: Not a small molecule (often when `biological` is `True`)
- Small molecules typically face many inexpensive generics with faster, deeper erosion

#### Example Data

| country | brand_name | ther_area | hospital_rate | main_package | biological | small_molecule |
|---------|------------|-----------|---------------|--------------|------------|----------------|
| COUNTRY_0024 | BRAND_1143 | Sensory_organs | 0.09 | EYE DROP | False | True |
| COUNTRY_0024 | BRAND_1865 | Muscoskeletal_Rheu... | 92.36 | INJECTION | False | False |
| COUNTRY_0024 | BRAND_2F6C | Antineoplastic_and... | 0.01 | INJECTION | True | False |

**Note:** Missing values (`nan`) exist in some columns and must be handled during preprocessing.

#### Modeling Applications
- **Stratified analysis:** Compare erosion patterns by therapeutic area, hospital_rate, or biologic vs small molecule
- **Feature engineering:**
  - One-hot encode `ther_area` and `main_package`
  - Include `hospital_rate` and interactions (e.g., `hospital_rate × n_gxs`)
  - Binary indicators for `biological` and `small_molecule`
- **Better generalization:** Leverage drug characteristics for unseen country-brand pairs

### 5.5 Erosion Buckets Distribution

The 340 test observations are distributed across **both scenarios** and **both erosion buckets**:

#### Bucket Definitions (Recap)

| Bucket | Mean Erosion Range | Description |
|--------|-------------------|-------------|
| **Bucket 1 (B1)** | [0, 0.25] | High erosion |
| **Bucket 2 (B2)** | (0.25, 1] | Low/medium erosion |

#### Distribution Characteristics
- Both Scenario 1 (228 observations) and Scenario 2 (112 observations) contain **both B1 and B2 cases**
- The proportion/structure is consistent between both scenarios
- Models must handle **both kinds of erosion dynamics** within each scenario

#### Visual Summary

| Dataset | Observations | Contains |
|---------|--------------|----------|
| Training | 1,953 | All erosion profiles |
| Test - Scenario 1 | 228 | Mix of B1 (high) + B2 (low) erosion |
| Test - Scenario 2 | 112 | Mix of B1 (high) + B2 (low) erosion |

---

## 6. Evaluation Metrics

### 6.1 Phase 1-a: Scenario 1 Prediction Error

#### Context
- Participants predict months 0–23 **without any post-generic actuals**
- Error computed by comparing predictions to true values

#### Four Error Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Absolute monthly error (months 0–23) | 20% | Month-by-month accuracy across full horizon |
| Absolute accumulated error (months 0–5) | **50%** | Total volume accuracy in critical first 6 months |
| Absolute accumulated error (months 6–11) | 20% | Total volume accuracy in months 6–11 |
| Absolute accumulated error (months 12–23) | 10% | Total volume accuracy in second year |

**Total: 20% + 50% + 20% + 10% = 100%**

#### Complete Formula

$$PE_j = 0.2 \cdot T_1 + 0.5 \cdot T_2 + 0.2 \cdot T_3 + 0.1 \cdot T_4$$

**Where:**

**Term $T_1$ — Monthly Error (20%):**
$$T_1 = \frac{\sum_{i=0}^{23} \left| Y^{act}_{j,i} - Y^{pred}_{j,i} \right|}{24 \cdot Avg_j}$$

- Sum absolute errors for each of 24 months
- Divide by 24 (average) and $Avg_j$ (brand scale)
- Rewards models matching detailed month-by-month shape

**Term $T_2$ — Accumulated Error Months 0–5 (50%):**
$$T_2 = \frac{\left| \sum_{i=0}^{5} Y^{act}_{j,i} - \sum_{i=0}^{5} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j}$$

- Compare total actual vs total predicted volume (months 0–5)
- Absolute difference of sums (not sum of absolute differences)
- **Highest weight (50%)** — critical early months

**Term $T_3$ — Accumulated Error Months 6–11 (20%):**
$$T_3 = \frac{\left| \sum_{i=6}^{11} Y^{act}_{j,i} - \sum_{i=6}^{11} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j}$$

- Same structure for mid-term period
- Measures cumulative accuracy in second half of year 1

**Term $T_4$ — Accumulated Error Months 12–23 (10%):**
$$T_4 = \frac{\left| \sum_{i=12}^{23} Y^{act}_{j,i} - \sum_{i=12}^{23} Y^{pred}_{j,i} \right|}{12 \cdot Avg_j}$$

- Second year cumulative accuracy
- Lowest weight — long-term errors less critical

#### Normalization Factor

$$Avg_j = \frac{\sum_{i=-12}^{-1} Y^{act}_{j,i}}{12}$$

- Average monthly volume in 12 months before generic entry
- Makes errors **relative to brand size**
- Enables fair comparison across large and small brands

### 6.2 Phase 1-b: Scenario 2 Prediction Error

#### Context
- Participants have 6 actual post-entry months (0–5) available
- Predictions required for months 6–23 only

#### Three Error Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Absolute monthly error (months 6–23) | 20% | Month-by-month accuracy across 18-month horizon |
| Absolute accumulated error (months 6–11) | **50%** | Total volume accuracy in first 6 forecast months |
| Absolute accumulated error (months 12–23) | 30% | Total volume accuracy in second year |

**Total: 20% + 50% + 30% = 100%**

#### Complete Formula

$$PE_j = 0.2 \cdot T_1 + 0.5 \cdot T_2 + 0.3 \cdot T_3$$

**Where:**

**Term $T_1$ — Monthly Error (20%):**
$$T_1 = \frac{\sum_{i=6}^{23} \left| Y^{act}_{j,i} - Y^{pred}_{j,i} \right|}{18 \cdot Avg_j}$$

- Sum absolute errors for 18 forecast months (6–23)
- Normalized by 18 months and $Avg_j$

**Term $T_2$ — Accumulated Error Months 6–11 (50%):**
$$T_2 = \frac{\left| \sum_{i=6}^{11} Y^{act}_{j,i} - \sum_{i=6}^{11} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j}$$

- **Highest weight (50%)** — first 6 months of prediction window
- Most important since observed months 0–5 inform this period

**Term $T_3$ — Accumulated Error Months 12–23 (30%):**
$$T_3 = \frac{\left| \sum_{i=12}^{23} Y^{act}_{j,i} - \sum_{i=12}^{23} Y^{pred}_{j,i} \right|}{12 \cdot Avg_j}$$

- Second year cumulative accuracy
- Higher weight (30%) than in Scenario 1 (10%) since early months are observed

### 6.3 Final Score Aggregation

#### Bucket-Weighted Final Score

After computing $PE_j$ for each country-brand, the final competition score aggregates across buckets:

$$PE = \frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1} + \frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}$$

**Where:**
- $n_{B1}$ = number of test observations in Bucket 1 (high erosion)
- $n_{B2}$ = number of test observations in Bucket 2 (low/medium erosion)
- $PE_{j,B1}$ = prediction error for brand $j$ in Bucket 1
- $PE_{j,B2}$ = prediction error for brand $j$ in Bucket 2

#### Interpretation

| Term | Meaning |
|------|---------|
| $\frac{1}{n_{B1}} \sum PE_{j,B1}$ | Average error across high-erosion brands |
| $\frac{1}{n_{B2}} \sum PE_{j,B2}$ | Average error across low-erosion brands |
| Factor **2** on B1 | **Bucket 1 counts twice as much as Bucket 2** |

**Rationale:** High-erosion cases (Bucket 1) are more business-critical, so they receive double weight in the final score.

### 6.4 Metric Interpretation Guide

#### Per-Brand Error ($PE_j$) Interpretation

| $PE_j$ Value | Interpretation |
|--------------|----------------|
| 0 | Perfect prediction |
| Close to 0 | Excellent predictions |
| Close to 1 | Average error ≈ baseline monthly volume |
| > 1 | Poor predictions; errors exceed typical pre-LOE volumes |

#### Final Score (PE) Ranges

| Final PE | Interpretation |
|----------|----------------|
| 0 | All predictions perfect |
| 3 | All $PE_j = 1$ (since 2×1 + 1×1 = 3) |
| > 3 | Some individual errors exceed 1 |

**Goal:** Minimize final PE score. **Lower is better.**

#### Metric Design Rationale

The metric captures three key business dimensions:

| Dimension | How Captured | Why Important |
|-----------|--------------|---------------|
| **Erosion Severity** | Bucket weighting (B1 × 2, B2 × 1) | High-erosion cases are most critical for business |
| **Time Sensitivity** | Period-specific weights (early months weighted highest) | Early erosion dynamics drive immediate business decisions |
| **Pattern Accuracy** | Monthly error terms | Captures month-to-month shape, not just cumulative totals |

#### Component Weight Comparison

| Component Type | Scenario 1 | Scenario 2 |
|----------------|-----------|-----------|
| Monthly error (all forecast months) | 20% | 20% |
| Early period cumulative | 50% (Months 0–5) | 50% (Months 6–11) |
| Mid period cumulative | 20% (Months 6–11) | — |
| Late period cumulative | 10% (Months 12–23) | 30% (Months 12–23) |

---

## 7. Summary and Key Takeaways

### Competition Overview

| Aspect | Detail |
|--------|--------|
| **Host** | Novartis Digital Finance Hub (Barcelona) |
| **Objective** | Forecast post-generic volume erosion over 24 months |
| **Primary Focus** | High-erosion cases (Bucket 1: Mean Erosion ∈ [0, 0.25]) |
| **Two Scenarios** | Scenario 1 (months 0–23, no actuals) and Scenario 2 (months 6–23, 6 actuals) |
| **Selection Process** | Phase 1 (quantitative) → Phase 2 (jury presentation) → Top 3 winners |

### Data Summary

| Dataset | Purpose | Key Fields |
|---------|---------|------------|
| `df_volume.csv` | Time series of monthly sales | `country`, `brand_name`, `months_postgx`, `volume` |
| `df_generics.csv` | Generic competition over time | `country`, `brand_name`, `months_postgx`, `n_gxs` |
| `df_medicine_info.csv` | Static product attributes | `ther_area`, `hospital_rate`, `main_package`, `biological`, `small_molecule` |

### Metric Summary

| Scenario | Forecast Horizon | Highest-Weighted Component |
|----------|------------------|---------------------------|
| **Scenario 1** | Months 0–23 | Accumulated error months 0–5 (50%) |
| **Scenario 2** | Months 6–23 | Accumulated error months 6–11 (50%) |

### Critical Success Factors

1. **Model Performance:**
   - Accurate predictions in both scenarios
   - Especially strong performance on Bucket 1 (high-erosion) cases
   - Good month-by-month trajectory AND cumulative accuracy

2. **Business Understanding:**
   - Deep exploratory analysis
   - Focus on high-erosion case characteristics
   - Clear preprocessing documentation

3. **Presentation Quality:**
   - Strong visualizations
   - Clear methodology explanation
   - Business-relevant insights and recommendations

### Key Formulas Quick Reference

| Formula | Purpose |
|---------|---------|
| $\text{Mean Erosion} = \frac{1}{24}\sum_{i=0}^{23} \text{Vol}_{\text{norm},i}$ | Quantify erosion severity |
| $\text{Vol}_{\text{norm},i} = \frac{\text{Vol}_i}{\text{Avg}_j}$ | Normalize post-entry volumes |
| $Avg_j = \frac{1}{12}\sum_{i=-12}^{-1} Y^{act}_{j,i}$ | Pre-generic baseline |
| $PE = 2 \cdot \bar{PE}_{B1} + 1 \cdot \bar{PE}_{B2}$ | Final competition score |

---

*Document generated from the official Novartis Datathon 2025 presentation slides.*
