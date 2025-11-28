# Novartis Datathon 2025: Generic Erosion Forecasting

## Complete Technical Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: The Generics Problem](#2-background-the-generics-problem)
   - 2.1 [Patents and Loss of Exclusivity](#21-patents-and-loss-of-exclusivity)
   - 2.2 [Generic Drug Definition](#22-generic-drug-definition)
   - 2.3 [Bioequivalence Requirements](#23-bioequivalence-requirements)
   - 2.4 [Market Consequences of Generic Entry](#24-market-consequences-of-generic-entry)
   - 2.5 [Real-World Example: Diovan](#25-real-world-example-diovan)
   - 2.6 [Role of Novartis Digital Finance Hub](#26-role-of-novartis-digital-finance-hub)
3. [Drug Lifecycle and Generic Erosion](#3-drug-lifecycle-and-generic-erosion)
   - 3.1 [Drug Sales Lifecycle Phases](#31-drug-sales-lifecycle-phases)
   - 3.2 [Generic Erosion Definition](#32-generic-erosion-definition)
   - 3.3 [Mean Generic Erosion Metric](#33-mean-generic-erosion-metric)
   - 3.4 [Erosion Classification and Buckets](#34-erosion-classification-and-buckets)
4. [The Challenge](#4-the-challenge)
   - 4.1 [Forecasting Objective](#41-forecasting-objective)
   - 4.2 [Business Scenarios](#42-business-scenarios)
   - 4.3 [Business Challenge Requirements](#43-business-challenge-requirements)
5. [Evaluation Process](#5-evaluation-process)
   - 5.1 [Phase 1: Model Evaluation](#51-phase-1-model-evaluation)
   - 5.2 [Phase 2: Jury Evaluation](#52-phase-2-jury-evaluation)
6. [Data Description](#6-data-description)
   - 6.1 [Dataset Overview](#61-dataset-overview)
   - 6.2 [Train-Test Split](#62-train-test-split)
   - 6.3 [Volume Dataset (df_volume.csv)](#63-volume-dataset-df_volumecsv)
   - 6.4 [Generics Dataset (df_generics.csv)](#64-generics-dataset-df_genericscsv)
   - 6.5 [Medicine Information Dataset (df_medicine_info.csv)](#65-medicine-information-dataset-df_medicine_infocsv)
   - 6.6 [Additional Data Guidelines](#66-additional-data-guidelines)
7. [Evaluation Metrics](#7-evaluation-metrics)
   - 7.1 [Metric Design Philosophy](#71-metric-design-philosophy)
   - 7.2 [Scenario 1 Prediction Error (PE) Formula](#72-scenario-1-prediction-error-pe-formula)
   - 7.3 [Scenario 2 Prediction Error (PE) Formula](#73-scenario-2-prediction-error-pe-formula)
   - 7.4 [Final Aggregated Metric](#74-final-aggregated-metric)
   - 7.5 [Metric Interpretation](#75-metric-interpretation)
8. [Modeling Recommendations](#8-modeling-recommendations)
9. [Appendix: Mathematical Formulas Reference](#9-appendix-mathematical-formulas-reference)

---

## 1. Executive Summary

The Novartis Datathon 2025 challenges participants to **forecast pharmaceutical sales volume erosion** following the entry of generic competitors into the market. When a drug's patent expires (Loss of Exclusivity), generic manufacturers can legally produce and sell equivalent versions, typically causing significant declines in the originator's sales—a phenomenon known as **generic erosion**.

### Key Objectives

- **Primary Goal**: Predict monthly sales volumes for 24 months following generic entry
- **Focus Area**: High-erosion drugs (Bucket 1) that lose ≥75% of pre-generic sales
- **Two Forecast Scenarios**: 
  - Scenario 1: Forecast at generic entry with no post-entry data
  - Scenario 2: Forecast 6 months post-entry with partial observed data

### Competition Structure

| Phase | Description | Advancement |
|-------|-------------|-------------|
| Phase 1-a | Scenario 1 accuracy evaluation | Top 10 teams advance |
| Phase 1-b | Scenario 2 accuracy evaluation | Top 5 teams advance |
| Phase 2 | Jury presentation and evaluation | Top 3 winners selected |

### Dataset Summary

| Dataset | Observations | Purpose |
|---------|-------------|---------|
| Training Set | 1,953 country-brand pairs | Model development |
| Test Set - Scenario 1 | 228 observations | Forecast Months 0-23 |
| Test Set - Scenario 2 | 112 observations | Forecast Months 6-23 |
| **Total** | **2,293 combinations** | |

---

## 2. Background: The Generics Problem

### 2.1 Patents and Loss of Exclusivity

When a pharmaceutical company develops a new drug, it receives a **patent** granting exclusive rights to produce and commercialize the product for a limited period (typically 20 years from filing). This exclusivity allows the company to recoup substantial R&D investments.

**Loss of Exclusivity (LOE)** occurs when:
- The patent expires
- Legal protection ends
- Generic manufacturers can legally enter the market

The timeline progression:
```
Innovation → Patent Grant → Exclusivity Period → LOE → Open Competition
```

### 2.2 Generic Drug Definition

A **generic drug** must be therapeutically equivalent to the brand-name (originator) medication in:

| Attribute | Description |
|-----------|-------------|
| **Dosage Form** | Tablet, capsule, injectable, etc. |
| **Strength** | Amount of active ingredient per dose (e.g., 80 mg) |
| **Route of Administration** | Oral, intravenous, topical, etc. |
| **Quality** | Meets regulatory standards for purity and stability |
| **Performance** | Behaves similarly in the body |
| **Intended Use** | Same indications and patient population |

**Important**: Generic products may contain different **inactive ingredients** (fillers, binders, colorants, coatings), but these differences must not affect therapeutic outcomes.

### 2.3 Bioequivalence Requirements

Generic manufacturers do not repeat full clinical trials. Instead, they must demonstrate **bioequivalence** through pharmacokinetic studies comparing:

| Property | Definition |
|----------|------------|
| **Absorption** | Rate and extent of drug entering the bloodstream |
| **Distribution** | How the drug spreads through body tissues |
| **Metabolism** | How the body transforms (breaks down) the drug |
| **Elimination** | How the drug and metabolites are excreted |

**Typical Bioequivalence Study Protocol**:
1. Healthy volunteers receive both brand-name and generic products (crossover design)
2. Blood samples collected over time
3. Concentration-time curves analyzed
4. Key metrics (AUC, Cmax) compared
5. Products considered bioequivalent if ratios fall within 80-125% acceptance range

This streamlined approval process significantly reduces development costs, enabling generics to be sold at lower prices.

### 2.4 Market Consequences of Generic Entry

| Consequence | Description | Impact |
|-------------|-------------|--------|
| **Increased Competition** | Multiple manufacturers produce the same medication | Drives prices downward |
| **Improved Affordability** | Lower development costs enable lower prices | More accessible treatments |
| **Greater Access** | Reduced prices expand treatment availability | Better disease management outcomes |
| **Substitution Practices** | Pharmacists may substitute generics for branded products | Accelerates market shift |

### 2.5 Real-World Example: Diovan

**Diovan** (Novartis) illustrates the generics problem:

| Attribute | Details |
|-----------|---------|
| **Active Ingredient** | Valsartan (angiotensin II receptor blocker) |
| **Indications** | Hypertension, heart failure |
| **Patent Expiry** | 2012 |
| **Post-LOE Impact** | Multiple generic manufacturers entered, significantly increasing competition and reducing prices |

### 2.6 Role of Novartis Digital Finance Hub

The Digital Finance Hub applies advanced analytics to financial processes, with key responsibilities including:

- **Sales Forecasting**: Predicting future sales to support strategic planning
- **Financial Impact Assessment**: Evaluating effects of events like generic entry
- **Country Organization Support**: Enabling monthly and annual sales reporting
- **Company-Wide Consolidation**: Supporting financial accounting and decision-making

---

## 3. Drug Lifecycle and Generic Erosion

### 3.1 Drug Sales Lifecycle Phases

A pharmaceutical product progresses through distinct lifecycle phases:

| Phase | Sales Behavior | Key Events |
|-------|----------------|------------|
| **Launch** | Low volume; awareness developing | Market introduction |
| **Growth** | Rapid increase; market acceptance | New indication approvals |
| **Maturity** | Sales stabilize at peak | Market saturation |
| **Competition** | Growth slows or declines | New competitor entry |
| **Post-LOE** | Steep decline | Generic entry (Gx) |

```
Volume
  │
  │                    LoE
  │                     ↓
  │              ┌─────●─────┐
  │             ╱             ╲
  │            ╱               ╲
  │           ╱                 ╲
  │          ╱                   ╲────────
  │         ╱                     
  │        ╱                      
  │───────╱                       
  │                               
  └────────────────────────────────────────→ Time
     Launch   Growth  Maturity    Erosion
```

### 3.2 Generic Erosion Definition

**Generic erosion** is the steep and sudden decline in branded drug sales volume following generic entry. This is the **central topic** of the Datathon.

**Business Importance**:
- Directly affects revenue forecasts
- Impacts production planning
- Influences strategic decisions (pricing, promotion, portfolio management)
- Enables preparation for post-patent period
- Helps minimize financial losses
- Supports competitive strategy adaptation

### 3.3 Mean Generic Erosion Metric

The Mean Generic Erosion (MGE) quantifies erosion severity over 24 months post-generic entry.

#### Formula 1.1: Mean Generic Erosion

$$\text{Mean Generic Erosion} = \frac{1}{24} \sum_{i=0}^{23} \text{vol}_i^{\text{norm}}$$

Where:
- $i$ ranges from 0 to 23 (24 months post-generic entry)
- $\text{vol}_i^{\text{norm}}$ is the normalized volume in month $i$

#### Formula 1.2: Normalized Volume

$$\text{vol}_i^{\text{norm}} = \frac{\text{Vol}_i}{\text{Avg}_j}$$

Where:
- $\text{Vol}_i$ = actual sales volume in month $i$ post-entry
- $\text{Avg}_j$ = baseline volume for drug $j$

#### Formula 1.3: Baseline Volume

$$\text{Avg}_j = \frac{1}{12} \sum_{i=-12}^{-1} y_{j,i}^{\text{act}}$$

Where:
- $y_{j,i}^{\text{act}}$ = actual sales volume of drug $j$ in month $i$
- Summation covers the 12 months before generic entry ($i = -12$ to $-1$)

**Interpretation**:
- MGE ≈ 1: Sales remain close to pre-generic baseline (low erosion)
- MGE ≈ 0: Sales collapse almost entirely (high erosion)
- MGE = 0.5: Average post-generic sales are 50% of baseline

### 3.4 Erosion Classification and Buckets

#### Three Conceptual Categories

| Category | MGE Range | Description |
|----------|-----------|-------------|
| **Low Erosion** | Close to 1 | Minimal impact; volume remains stable |
| **Medium Erosion** | Between 0 and 1 | Moderate decline |
| **High Erosion** | Close to 0 | Sharp drop in volume |

#### Two Datathon Buckets

For competition purposes, drugs are classified into two buckets:

| Bucket | MGE Range | Description | Weight |
|--------|-----------|-------------|--------|
| **Bucket 1 (B1)** | [0, 0.25] | High erosion (≥75% loss) | **Primary focus** |
| **Bucket 2 (B2)** | (0.25, 1] | Medium/Low erosion (<75% loss) | Secondary |

**Bucket Derivation**: Not provided in datasets; calculate using Formula 1.1.

---

## 4. The Challenge

### 4.1 Forecasting Objective

Participants must **model and predict monthly sales volumes** for a 24-month period following generic entry. This is a **time-series forecasting problem** focusing on post-LOE behavior.

### 4.2 Business Scenarios

#### Scenario 1: Zero-Knowledge Post-Entry

| Aspect | Details |
|--------|---------|
| **Timing** | Immediately at generic entry date |
| **Available Data** | Pre-generic history only |
| **Forecast Horizon** | Month 0 to Month 23 (24 months) |
| **Business Context** | Planning at LOE before observing market reaction |
| **Test Observations** | 228 |

#### Scenario 2: Six-Month Update

| Aspect | Details |
|--------|---------|
| **Timing** | Six months after generic entry |
| **Available Data** | Pre-generic history + 6 months post-entry actuals |
| **Forecast Horizon** | Month 6 to Month 23 (18 months) |
| **Business Context** | Updated forecast incorporating early erosion behavior |
| **Test Observations** | 112 |

### 4.3 Business Challenge Requirements

The Datathon evaluates both **technical excellence** and **business acumen**:

#### Technical Requirements
- Build accurate forecasting models
- Handle data preprocessing appropriately
- Justify feature engineering decisions

#### Business Requirements
- Explain **why** certain approaches were chosen
- Provide **deep exploratory analysis** of data preprocessing
- Focus specifically on **high-erosion cases** (Bucket 1)
- Use **visualization tools** to make findings:
  - Clear
  - Interpretable
  - Business-oriented

---

## 5. Evaluation Process

### 5.1 Phase 1: Model Evaluation

All teams submit predictions for the **entire test dataset** (both scenarios).

#### Phase 1-a: Scenario 1 Accuracy

| Aspect | Details |
|--------|---------|
| **Evaluated** | All participating teams |
| **Metric** | Prediction Error (PE) per Equation 4.1 |
| **Ranking** | By prediction errors (lower is better) |
| **Advancement** | **Top 10 teams** proceed to Phase 1-b |

#### Phase 1-b: Scenario 2 Accuracy

| Aspect | Details |
|--------|---------|
| **Evaluated** | Top 10 teams from Phase 1-a |
| **Metric** | Prediction Error (PE) per Equation 4.2 |
| **Ranking** | By prediction errors (lower is better) |
| **Advancement** | **Top 5 teams** proceed to Phase 2 |

### 5.2 Phase 2: Jury Evaluation

The **five finalist teams** present to a jury composed of technical and business experts.

#### Presentation Requirements
- **Methodologies**: Models, algorithms, approaches used
- **Modeling Decisions**: Justification for choices made
- **Insights**: Patterns discovered, drivers of erosion identified
- **Conclusions**: Business implications and recommendations

#### Evaluation Criteria
- Technical soundness and rigor
- Interpretability and clarity
- Business relevance and practical applicability
- Quality of visualizations and communication

#### Outcome
Jury selects the **top 3 winning teams**.

---

## 6. Data Description

### 6.1 Dataset Overview

Data consists of historical monthly volumes for **2,293 country-brand combinations** that have experienced generic entry.

**Data Structure**: Three separate DataFrames
1. Volume Dataset (time-series metrics)
2. Generics Dataset (competition information)
3. Medicine Information Dataset (static attributes)

**Join Keys**: `country` and `brand_name`

### 6.2 Train-Test Split

| Dataset | Observations | Data Availability |
|---------|-------------|-------------------|
| **Training** | 1,953 | Up to 24 months pre-entry + up to 24 months post-entry |
| **Test - Scenario 1** | 228 | Pre-entry data only |
| **Test - Scenario 2** | 112 | Pre-entry + 6 months post-entry |

### 6.3 Volume Dataset (df_volume.csv)

The primary time-series dataset containing sales volumes.

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market of reference (e.g., COUNTRY_A67D) |
| `brand_name` | String | Product identifier (e.g., BRAND_75FD) |
| `month` | Date | Calendar month of observation (e.g., 2015-01) |
| `months_postgx` | Integer | Months relative to generic entry:<br>• 0 = entry month<br>• Negative = before entry<br>• Positive = after entry |
| `volume` | Numeric | **TARGET VARIABLE** - Units sold |

**Key Notes**:
- Each row represents one month for one country-brand pair
- `months_postgx` aligns all series around the generic entry event
- `volume` is the dependent variable to predict

### 6.4 Generics Dataset (df_generics.csv)

Tracks generic competition over time.

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market identifier |
| `brand_name` | String | Product identifier |
| `months_postgx` | Integer | Months relative to generic entry |
| `n_gxs` | Integer | Number of generic products available |

**Key Notes**:
- `n_gxs` is **time-varying**: changes as generics enter or exit
- More generics typically correlate with higher erosion
- Useful for modeling dynamic competition effects

### 6.5 Medicine Information Dataset (df_medicine_info.csv)

Static product characteristics.

| Column | Type | Description |
|--------|------|-------------|
| `country` | String | Market identifier |
| `brand_name` | String | Product identifier |
| `therapeutic_area` | Categorical | Therapeutic category (e.g., cardiovascular, oncology) |
| `hospital_rate` | Numeric | Proportion of units distributed through hospitals |
| `main_package` | Categorical | Predominant commercial format (pills, vials, etc.) |
| `biological` | Boolean | Whether the product is a biologic medicine |
| `small_molecule` | Boolean | Whether the product is a low-molecular-weight synthetic compound |

**Key Notes**:
- All variables assumed **time-invariant**
- Can be used for product segmentation and feature engineering
- Helps interpret erosion pattern differences

### 6.6 Additional Data Guidelines

| Guideline | Details |
|-----------|---------|
| **Granularity** | Monthly level, starting from brand launch or first available data |
| **Data Usage** | All provided data may be used for training (including Scenario 2 test data for Scenario 1 models) |
| **Bucket Labels** | Not provided; derive using Formula 1.1 |
| **Modeling Freedom** | Any approach/model allowed; **explainability and simplicity valued** |
| **Volume Units** | May differ by country-brand (milligrams, packs, pills, etc.); normalize per series |
| **Categorical Variables** | Assumed constant over time |
| **Missing Values** | Present in some columns; preprocessing strategy is participant's choice |

---

## 7. Evaluation Metrics

### 7.1 Metric Design Philosophy

The Prediction Error (PE) metric captures three dimensions:

| Dimension | How Captured |
|-----------|--------------|
| **Generic Erosion Severity** | Bucket-level weighting (Eq. 4.3): Bucket 1 has 2× importance |
| **Temporal Dynamics** | Differential weighting across periods (Eq. 4.1, 4.2): early months weighted more |
| **Seasonality Effects** | Monthly error terms: month-by-month deviations contribute to score |

**Normalization**: All components normalized by $\text{Avg}_j$ (pre-generic average volume) for cross-series comparability.

### 7.2 Scenario 1 Prediction Error (PE) Formula

#### Component Weights

| Component | Weight | Period |
|-----------|--------|--------|
| Absolute monthly error (all months) | 20% | Months 0-23 |
| Accumulated error (early) | **50%** | Months 0-5 |
| Accumulated error (mid) | 20% | Months 6-11 |
| Accumulated error (late) | 10% | Months 12-23 |

#### Formula 4.1

$$PE_j = 0.2 \left( \frac{\sum_{i=0}^{23} \left| Y^{act}_{j,i} - Y^{pred}_{j,i} \right|}{24 \cdot Avg_j} \right) + 0.5 \left( \frac{\left| \sum_{i=0}^{5} Y^{act}_{j,i} - \sum_{i=0}^{5} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j} \right)$$

$$+ 0.2 \left( \frac{\left| \sum_{i=6}^{11} Y^{act}_{j,i} - \sum_{i=6}^{11} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j} \right) + 0.1 \left( \frac{\left| \sum_{i=12}^{23} Y^{act}_{j,i} - \sum_{i=12}^{23} Y^{pred}_{j,i} \right|}{12 \cdot Avg_j} \right)$$

Where:
- $Y^{act}_{j,i}$ = actual volume for series $j$ in month $i$
- $Y^{pred}_{j,i}$ = predicted volume for series $j$ in month $i$
- $Avg_j$ = pre-generic average monthly volume (Formula 1.3)

#### Term-by-Term Breakdown

| Term | Formula | Interpretation |
|------|---------|----------------|
| **T₁** (20%) | $\frac{\sum \|Y^{act} - Y^{pred}\|}{24 \cdot Avg_j}$ | Average normalized monthly absolute error |
| **T₂** (50%) | $\frac{\|\sum_{0-5} Y^{act} - \sum_{0-5} Y^{pred}\|}{6 \cdot Avg_j}$ | Cumulative error in first 6 months |
| **T₃** (20%) | $\frac{\|\sum_{6-11} Y^{act} - \sum_{6-11} Y^{pred}\|}{6 \cdot Avg_j}$ | Cumulative error in months 6-11 |
| **T₄** (10%) | $\frac{\|\sum_{12-23} Y^{act} - \sum_{12-23} Y^{pred}\|}{12 \cdot Avg_j}$ | Cumulative error in year 2 |

### 7.3 Scenario 2 Prediction Error (PE) Formula

#### Component Weights

| Component | Weight | Period |
|-----------|--------|--------|
| Absolute monthly error (evaluated months) | 20% | Months 6-23 |
| Accumulated error (early evaluated) | **50%** | Months 6-11 |
| Accumulated error (late) | 30% | Months 12-23 |

#### Formula 4.2

$$PE_j = 0.2 \left( \frac{\sum_{i=6}^{23} \left| Y^{act}_{j,i} - Y^{pred}_{j,i} \right|}{18 \cdot Avg_j} \right) + 0.5 \left( \frac{\left| \sum_{i=6}^{11} Y^{act}_{j,i} - \sum_{i=6}^{11} Y^{pred}_{j,i} \right|}{6 \cdot Avg_j} \right)$$

$$+ 0.3 \left( \frac{\left| \sum_{i=12}^{23} Y^{act}_{j,i} - \sum_{i=12}^{23} Y^{pred}_{j,i} \right|}{12 \cdot Avg_j} \right)$$

**Note**: Scenario 2 evaluates 18 months (6-23), hence the denominator of 18 in Term 1.

### 7.4 Final Aggregated Metric

#### Formula 4.3: Bucket-Weighted Aggregation

$$PE = \frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1} + \frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}$$

Where:
- $n_{B1}$ = number of series in Bucket 1 (high erosion)
- $n_{B2}$ = number of series in Bucket 2
- $PE_{j,B1}$ = prediction error for series $j$ in Bucket 1
- $PE_{j,B2}$ = prediction error for series $j$ in Bucket 2

**Key Features**:
- Bucket 1 errors are weighted **2×** more than Bucket 2
- Each country-brand pair appears in only one scenario
- Final PE is computed separately for each scenario

### 7.5 Metric Interpretation

#### Individual Series PE ($PE_j$)

| Value | Interpretation |
|-------|----------------|
| **0** | Perfect predictions |
| **0.1 - 0.3** | Good accuracy |
| **0.5** | Moderate errors |
| **1** | Errors approximately equal to baseline volume |
| **> 1** | Poor predictions; errors exceed baseline |

#### Scenario-Level PE

| Value | Interpretation |
|-------|----------------|
| **0** | All predictions perfect |
| **3** | All $PE_j = 1$ (since $2 \times 1 + 1 \times 1 = 3$) |
| **> 3** | Possible if individual $PE_j > 1$ |

**Goal**: Minimize final PE.

---

## 8. Modeling Recommendations

Based on the metric design and competition structure, successful models should:

### Data Preprocessing
- Handle missing values appropriately (imputation, dropping, or model-based handling)
- Normalize volumes per series using pre-generic baseline
- Derive bucket labels using Formula 1.1 for analysis and stratification
- Account for different volume units across country-brand pairs

### Feature Engineering
- Leverage `months_postgx` for time-based alignment
- Use `n_gxs` to model competition intensity effects
- Incorporate medicine characteristics for product segmentation
- Consider seasonality patterns in historical data

### Model Design Priorities

| Priority | Rationale |
|----------|-----------|
| **Early erosion accuracy** | 50% weight on first evaluated period |
| **High-erosion cases** | Bucket 1 has 2× importance |
| **Monthly pattern matching** | 20% weight on per-month errors |
| **Cumulative accuracy** | 30-50% weight on period totals |
| **Interpretability** | Valued by jury in Phase 2 |

### Scenario-Specific Strategies

#### Scenario 1 (No post-entry data)
- Rely on pre-generic patterns and product characteristics
- Learn erosion shapes from training data
- Consider classification approach (bucket prediction) alongside regression

#### Scenario 2 (6 months post-entry data)
- Update forecasts using observed early erosion
- Calibrate based on actual erosion trajectory
- May benefit from conditional modeling given early behavior

---

## 9. Appendix: Mathematical Formulas Reference

### Core Erosion Metrics

| Formula | Equation |
|---------|----------|
| **Mean Generic Erosion (1.1)** | $\text{MGE} = \frac{1}{24} \sum_{i=0}^{23} \text{vol}_i^{\text{norm}}$ |
| **Normalized Volume (1.2)** | $\text{vol}_i^{\text{norm}} = \frac{\text{Vol}_i}{\text{Avg}_j}$ |
| **Baseline Volume (1.3)** | $\text{Avg}_j = \frac{1}{12} \sum_{i=-12}^{-1} y_{j,i}^{\text{act}}$ |

### Prediction Error Metrics

| Formula | Application |
|---------|-------------|
| **PE Scenario 1 (4.1)** | Evaluates Months 0-23 predictions |
| **PE Scenario 2 (4.2)** | Evaluates Months 6-23 predictions |
| **Final PE (4.3)** | Bucket-weighted aggregation |

### Bucket Thresholds

| Bucket | MGE Condition |
|--------|---------------|
| Bucket 1 (High Erosion) | $0 \leq \text{MGE} \leq 0.25$ |
| Bucket 2 (Other) | $0.25 < \text{MGE} \leq 1$ |

### Weight Summary

| Component | Scenario 1 | Scenario 2 |
|-----------|------------|------------|
| Monthly errors | 20% | 20% |
| Early cumulative | 50% (M0-5) | 50% (M6-11) |
| Mid cumulative | 20% (M6-11) | — |
| Late cumulative | 10% (M12-23) | 30% (M12-23) |
| **Bucket 1 weight** | **2×** | **2×** |
| **Bucket 2 weight** | 1× | 1× |

---

*Document Version: 1.0 | Novartis Datathon 2025 | Generic Erosion Forecasting Challenge*
