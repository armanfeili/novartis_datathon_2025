# Novartis BCN Digital Finance Hub Datathon

**Problem, Data & Evaluation – Full Technical Specification**

---

## 1. Context: The BCN Digital Finance Hub

### 1.1 What is the BCN Digital Finance Hub?

The **BCN Digital Finance Hub** is a Novartis structure based in Barcelona that centralizes data, analytics, and technology capabilities to support finance processes globally.

* **BCN** stands for **Barcelona**.
* It is a **Digital Finance Hub**: a cross-functional team where finance experts, data scientists, and engineers combine:

  * data,
  * analytics and machine learning,
  * and modern software engineering
    to enhance and transform financial decision-making.

The Novartis logo on the slides indicates that the hub is part of **Novartis**, a global pharmaceutical company.

---

### 1.2 Team composition

The hub’s team is explicitly quantified and segmented by role:

* **34 Data Scientists**

  * Responsibilities typically include:

    * data exploration and cleaning,
    * statistical and machine-learning modelling,
    * predictive and prescriptive analytics (e.g., cost forecasting, budget optimization, anomaly detection).
  * The size of this group signals a **strong analytics focus**.

* **8 Finance profiles**

  * Finance/business experts (e.g., financial controllers, analysts).
  * Key responsibilities:

    * translate business questions into analytical problems,
    * ensure model outputs are financially meaningful,
    * validate results against accounting, P&L, and regulatory constraints.

* **15 Visualization, ML, Software Engineers & DevOps**

  * This cluster covers:

    * **Visualization** specialists (dashboards, BI reports, data apps),
    * **ML Engineers** (industrialization, scalability, MLOps),
    * **Software Engineers** (applications and internal tools),
    * **DevOps** (infrastructure, CI/CD, monitoring, reliability).
  * This group provides the **engineering backbone** to run models and tools in production.

Overall team size (from these three groups):
**34 + 8 + 15 = 57** people.

The presence of silhouettes in the slide visually reinforces this as a **modern, professional, multidisciplinary** team.

---

### 1.3 Diversity and origin

The hub is explicitly characterized by both diversity and local anchoring:

* **“+14 Different nationalities in a diverse team”**

  * At least **14 nationalities** are represented.
  * This implies a culturally diverse environment with varied perspectives and experiences.

* **“66% Local talent”**

  * 66% of team members are local (from Barcelona or Spain).
  * The remaining 34% are international hires.
  * Using the approximate team size (57 members):

    * Local ~ 0.66 × 57 ≈ 38 people,
    * International ~ 0.34 × 57 ≈ 19 people.
  * This shows a balance between **leveraging local talent** and maintaining a **global outlook**.

---

### 1.4 Barcelona as a European tech and AI pole

The hub is located in Barcelona for strategic reasons. The slide highlights four key points:

1. **Tech cluster**

   * In the last decade, many major companies (e.g. **Amazon, Microsoft, AstraZeneca**) have located their **global AI hubs** in Barcelona.
   * This indicates a strong **tech/AI ecosystem** with:

     * abundant talent,
     * local suppliers and partners,
     * and a network of innovation actors.

2. **Forefront centers & infrastructures**

   * The city hosts advanced research infrastructure, including:

     * **Barcelona Supercomputing Center** (high-performance computing),
     * **Quantum Computer** initiatives,
     * **Synchrotron** facilities.
   * These assets reinforce Barcelona as a **scientific and technological hotspot** suitable for data-intensive work.

3. **Proficiency of local universities**

   * Local universities provide strong programs in:

     * **Data Science**,
     * **Mathematics**,
     * **Statistics**.
   * This ensures a **continuous pipeline of graduates** with quantitative and analytical skills.

4. **Attractive city for international talent to relocate**

   * Barcelona is described as an **attractive city for international talent to relocate** (spelled “reallocate” on the slide).
   * Factors include:

     * quality of life,
     * climate and culture,
     * cost of living relative to other tech hubs.
   * This facilitates **recruitment and retention** of global talent.

---

### 1.5 Evolution and growth of the hub

A bar chart summarizes the hub’s growth over time (likely headcount):

* **Years (x-axis):** 2018–2025
* **Values (number of people):**

  * 2018 → 10
  * 2019 → 19
  * 2020 → 28
  * 2021 → 34
  * 2022 → 40
  * 2023 → 50
  * 2024 → 53
  * 2025 → 55

Interpretation:

* The hub grew from **10 people (2018)** to **55 people (2025)**—more than a **five-fold increase**.
* Growth is **monotonic** (never decreases).
* Largest jumps:

  * +9 from 2018 to 2019,
  * +9 from 2019 to 2020,
  * +10 from 2022 to 2023.
* A smaller increase in the last years (53 → 55) suggests:

  * a **maturing** organization approaching its target size or
  * a shift from rapid expansion to **consolidation**.

---

### 1.6 Background and academic disciplines

The team’s educational and professional backgrounds are shown via a **stacked bar at 100%**. From bottom to top:

* **19% – Physics & Others**
* **20% – Economics**
* **17% – Engineering**
* **20% – Computer Science**
* **24% – Mathematics & Statistics**

If the hub had 100 people (for interpretation):

* 24 with **Math & Stats** backgrounds,
* 20 with **Computer Science**,
* 17 with **Engineering**,
* 20 with **Economics**,
* 19 with **Physics or other disciplines**.

This mix indicates a deliberate combination of:

* **Quantitative theory** (Math & Stats, Physics),
* **Technical and systems skills** (Computer Science, Engineering),
* **Market and financial understanding** (Economics).

Additionally, a box highlights:

* **5 PhDs**
* **3 Bioinformatics backgrounds**

This emphasizes:

* High **research and advanced analytics** capacity (PhDs).
* Cross-domain expertise linking data science to **biology and medicine** (Bioinformatics), which is highly relevant in a pharmaceutical context, even though the hub’s focus is finance.

---

## 2. Business Context: Drug Lifecycle and Generic Erosion

### 2.1 Lifecycle of a drug – market perspective

A slide titled **“Lifecycle of a drug”** presents a curve of **Volume of sales** over **Time**:

* **Y-axis:** “Volume of sales”
* **X-axis:** “Time”

Key lifecycle milestones:

1. **Drug Launch**

   * Initial market introduction.
   * Sales are low at first, then start rising as:

     * awareness grows,
     * reimbursement and pricing are set,
     * more physicians prescribe the drug.

2. **New Indication Entry**

   * The same drug obtains approval for an additional indication (new disease or patient subgroup).
   * This expands the **eligible patient population**, typically causing an increase in sales trajectory.

3. **New Competitor Entry**

   * A new branded competitor enters the same therapeutic area.
   * This typically:

     * slows growth,
     * may flatten or slightly reduce sales.

4. **Loss of Exclusivity (LoE)**

   * The point at which the drug loses patent and/or regulatory exclusivity.
   * The slide labels the peak near this event as:

     * “Loss of Exclusivity (LoE)”
   * Shortly after LoE, sales peak and then decline sharply.

5. **Generics Entry (Gx)**

   * Marked by “Generics entry (Gx)” on the x-axis.
   * Represents the arrival of **generic copies**.
   * After Gx:

     * cheaper generics capture market share,
     * branded volume typically **erodes rapidly**,
     * eventually stabilizes at a much lower residual level.

A yellow dashed rectangle labelled **“DATATHON”** highlights the **post-LoE / generics-entry period**, indicating that the Datathon focuses on **this erosion phase** rather than the entire lifecycle.

---

### 2.2 Concept of “Generic Erosion” and Mean Generic Erosion

**Generic erosion** refers to the **decline in sales volume** of a branded drug after generic competitors enter the market.

To quantify this, the slides introduce **Mean Generic Erosion**, defined as follows:

1. **Pre-generic reference average** for brand or market (j):

[
Avg_j = \frac{\sum_{i=-12}^{-1} Y^{act}_{j,i}}{12}
]

* (Y^{act}_{j,i}): actual volume for brand/country (j) in month (i).
* (i = -12,\dots,-1): 12 months before generic entry.
* (Avg_j): **average monthly volume** in the last year before generics—used as baseline scale.

2. **Normalized monthly volume after generic entry**:

[
VolNorm_i = \frac{Vol_i}{Avg_j}
]

* (Vol_i): actual volume in month (i) after generic entry.
* (VolNorm_i): volume as a ratio to the pre-generic average:

  * (VolNorm_i = 1) → equal to pre-generic average.
  * (VolNorm_i = 0.5) → 50% of pre-generic average.
  * (VolNorm_i = 2) → double the pre-generic average (rare in practice post-Gx).

3. **Mean Generic Erosion over 24 months post-entry**:

[
\textit{Mean Generic Erosion} = \frac{\sum_{i=0}^{23} Vol_{norm,i}}{24}
]

* (i = 0,\dots,23): 24 months after generic entry.
* This is the **average normalized volume** over two years post-Gx.

Interpretation:

* Mean Generic Erosion ≈ 1:

  * Little to no erosion; post-generic volumes remain close to pre-generic baseline.
* Mean Generic Erosion < 1:

  * Sales have dropped.
  * E.g. 0.4 ≈ 40% of pre-generic volume on average.
* Mean Generic Erosion near 0:

  * Extreme erosion; brand has lost almost all volume post-Gx.

This metric is used both for **describing erosion** and for defining the **buckets** used in evaluation.

---

### 2.3 Erosion profiles and bucket definitions

A family of stylised curves illustrates three **qualitative erosion profiles**:

1. **High Erosion (red curve)**

   * Pre-Gx: slight growth.
   * Post-Gx: volume drops very quickly to near zero.
   * Represents **very aggressive erosion**.

2. **Medium Erosion (yellow curve)**

   * Highest volume pre-Gx; stable.
   * Post-Gx: sharp decline, then gradual decrease.
   * Significant but not total erosion.

3. **Low Erosion (blue curve)**

   * Moderate, slightly declining volume pre-Gx.
   * Post-Gx: slow, smooth decline; retains the highest volume among the three curves.
   * Mild erosion; brand maintains a substantial market share.

These three profiles are grouped into **two numerical buckets** using Mean Generic Erosion:

* **Bucket 1 (B1) – High Erosion**

  * Interval:
    [
    \text{Mean Erosion} \in [0, 0.25]
    ]
  * Erosion is severe:

    * On average, ≤25% of pre-generic volume remains.
    * 0 indicates almost complete loss.

* **Bucket 2 (B2) – Medium/Low Erosion**

  * Interval:
    [
    \text{Mean Erosion} \in (0.25, 1]
    ]
  * Erosion is moderate to low:

    * More than 25% of pre-generic volume is retained.
    * Up to 100% (no erosion).

The slides repeatedly emphasize:

* **Bucket 1 (B1) = High Erosion** is the **primary focus** of the Datathon (explicitly labelled “Datathon Focus!!!” on one slide).
* Bucket 2 (B2) provides comparative context and is also part of evaluation, but with lower weight.

---

## 3. Datathon Challenge Overview

### 3.1 Data Science Challenge – Forecasting tasks

Participants must develop models to **forecast the volume erosion** following generic entry over a **24-month horizon** from the generic entry date.

Two forecasting **scenarios** are defined:

#### Scenario 1 – At generic entry (no post-Gx actuals)

* Time of prediction: **right after the generic entry date**.
* Available data:

  * **All pre-generic history**,
  * **No post-generic actual volumes**.
* Task:

  * Forecast **monthly volumes from month 0 to month 23** (24 months in total).
* This is the **pure ex-ante prediction** scenario.

#### Scenario 2 – Six months after generic entry

* Time of prediction: **6 months after generic entry**.
* Available data:

  * Pre-generic history,
  * **Six actual months post-generic**: months 0–5.
* Task:

  * Forecast **months 6 to 23** (18 future months).
* This scenario represents a **mid-course forecast update**, leveraging early erosion data.

In both scenarios, forecasts are at the **(country, brand)** level and will later be used to derive metrics such as **Mean Generic Erosion** and to classify markets into **B1** and **B2**.

---

### 3.2 Business Challenge – Exploratory and interpretive work

Beyond modelling, teams must address a **Business Challenge**:

> Teams presenting to the Jury must provide a **deep exploratory analysis** of the preprocessing carried out, with **focus on high-erosion cases** (Bucket 1), making extensive use of **visualization tools**.

Key expectations:

* **Deep EDA (Exploratory Data Analysis)**:

  * Structure and describe:

    * distributions, trends, and seasonality,
    * differences by country, brand, therapeutic area, etc.
  * Highlight patterns specifically for **high-erosion markets (B1)**.

* **Transparency of preprocessing**:

  * Clearly explain:

    * missing-data handling,
    * outlier treatment,
    * transformations and scaling,
    * feature engineering choices (e.g., lags, competitive features).
  * The Jury wants to understand **why** each step was taken.

* **Focus on Bucket 1 (High erosion)**:

  * Identify what characterizes markets in B1 compared to B2:

    * geography, therapy, pricing, competition, hospital vs retail, etc.
  * Derive **business insights** from these differences.

* **Visualization**:

  * Use visual tools (plots, dashboards) to make findings intuitive.

---

### 3.3 Winner selection process

The **winner selection** occurs in two phases:

#### Phase 1 – Model Evaluation (Quantitative)

All teams submit forecasts for the full test set (both Scenario 1 and 2). Evaluation occurs in two sub-phases:

1. **Phase 1-a: Scenario 1 Evaluation**

   * All teams evaluated on **Scenario 1** prediction error.
   * The **top 10 teams with lowest Scenario-1 error** progress to Phase 1-b.

2. **Phase 1-b: Scenario 2 Evaluation**

   * Only the **top 10** from Phase 1-a are evaluated on **Scenario 2** prediction error.
   * The **top 5 teams with lowest Scenario-2 error** advance to **Phase 2 (Final)**.

#### Phase 2 – Jury Evaluation (Qualitative + Quantitative)

* The **final 5 teams** present their:

  * methodology,
  * insights,
  * conclusions.
* Jury composition:

  * **Technical experts** (data science, statistics),
  * **Business experts** (finance, commercial, market access, etc.).
* Criteria:

  * model quality (from Phase 1),
  * clarity and rigor of methodology,
  * interpretability,
  * business relevance of insights.

The Jury selects the **top 3 winning teams** based on this combined assessment.

---

## 4. Data Description

### 4.1 Overall dataset structure and splits

The dataset comprises time series of **monthly volumes** for **2,293 country–brand combinations** that **experienced generic entry**.

* **Target variable**:

  * Monthly sales volume at the **(country, brand)** level.

The data is divided into:

* **Training set**:

  * **1,953 observations** (each observation = one country–brand time series).
  * For each:

    * up to **24 months before generic entry**,
    * and up to **24 months after** generic entry.

* **Test set**:

  * **340 observations**, split into:

    * **Scenario 1**: ~2/3 → **228** country–brand time series.

      * Forecast required **0–23 months** post-generic.
    * **Scenario 2**: ~1/3 → **112** country–brand time series.

      * Forecast required **6–23 months** post-generic (months 0–5 are observed).

A bar chart illustrates:

* **Train**: single grey bar labeled 1,953.
* **Test – Scenario 1**: stacked bar labeled 228, split into:

  * red (B1 – High erosion),
  * blue (B2 – Low erosion).
* **Test – Scenario 2**: stacked bar labeled 112, similarly split into B1 and B2.

The textual explanation clarifies:

* The **340 test observations** are distributed across **scenario (1 or 2)** and **erosion levels (B1, B2)** such that **both scenarios contain both buckets**, and this structure is consistent.

---

### 4.2 Volume data: `df_volume.csv`

**File:** `df_volume.csv`
**Content:** monthly sales volumes before and after generic entry.

Columns:

1. **`country`**

   * Country name (anonymized, e.g. `COUNTRY_B6AE`).

2. **`brand_name`**

   * Brand name (anonymized, e.g. `BRAND_1C1E`).

3. **`month`**

   * Calendar month name (e.g. `Jul`, `Aug`, `Sep`).
   * This is mainly descriptive; the real time index is `months_postgx`.

4. **`months_postgx`**

   * Integer index of the month **relative to generic entry**:

     * `0` = month of generic entry (Gx),
     * negative values = months **before** Gx, e.g.

       * `-3` = three months before Gx,
       * `-24` = 24 months before Gx.
     * positive values (not shown in sample) = months **after** Gx.

5. **`volume`**

   * **Number of drugs sold** in that month (e.g. `272594.39`).
   * This is the key **target** for forecasting and for computing erosion metrics.

Example snippet for one `(country, brand)` pair:

| country      | brand_name | month | months_postgx | volume    |
| ------------ | ---------- | ----- | ------------- | --------- |
| COUNTRY_B6AE | BRAND_1C1E | Jul   | -24           | 272594.39 |
| COUNTRY_B6AE | BRAND_1C1E | Aug   | -23           | 351859.31 |
| COUNTRY_B6AE | BRAND_1C1E | Sep   | -22           | 447953.48 |
| …            | …          | …     | …             | …         |

Usage:

* Construct time series per `(country, brand_name)`.
* Separate pre-generic vs post-generic segments via `months_postgx`.
* Align all series on a common relative time axis.
* Feed into forecasting models and computation of metrics (Mean Generic Erosion, prediction errors).

---

### 4.3 Generics features: `df_generics.csv`

**File:** `df_generics.csv`
**Content:** time-varying number of generic competitors for each `(country, brand)` after generic entry.

Columns:

1. **`country`**

2. **`brand_name`**

   * Same identifiers as in `df_volume.csv`, allowing joins.

3. **`months_postgx`**

   * Number of months after generic entry (0, 1, 2, …).
   * Only non-negative values (focus on post-Gx period).

4. **`n_gxs`**

   * **Number of generics** on the market for this brand in this country at that month.
   * The slide notes that **the number of generics can change over time** (new entrants, exits).

Example snippet:

| country      | brand_name | months_postgx | n_gxs |
| ------------ | ---------- | ------------- | ----- |
| COUNTRY_B6AE | BRAND_DF2E | 0             | 0.0   |
| COUNTRY_B6AE | BRAND_DF2E | 1             | 0.0   |
| COUNTRY_B6AE | BRAND_DF2E | 2             | 1.0   |
| COUNTRY_B6AE | BRAND_DF2E | 3             | 2.0   |
| …            | …          | …             | …     |

Interpretation:

* In this example, the first generic actually appears at **month 2** post-Gx (n_gxs=1).
* A second generic appears at **month 3** (n_gxs=2).
* Competition intensity is thus **time-dependent**.

Usage:

* Join with volume data on `(country, brand_name, months_postgx)`.
* Use `n_gxs` directly or derive features such as:

  * indicator `n_gxs > 0` (any generics),
  * time since first generic,
  * changes in `n_gxs` month to month.
* These features capture **competitive pressure**, a major driver of erosion.

---

### 4.4 Drug-related features: `df_medicine_info.csv`

**File:** `df_medicine_info.csv`
**Content:** static metadata about each drug (brand) in each country.

Columns:

1. **`country`**

2. **`brand_name`**

   * As before, for joining.

3. **`ther_area`**

   * Therapeutic area of the drug, e.g.:

     * `Sensory_organs`,
     * `Muscoskeletal_Rheu...` (musculoskeletal / rheumatology),
     * `Antineoplastic_and...` (oncology / antineoplastic),
     * `Nervous_system`, etc.

4. **`hospital_rate`**

   * Percentage of volume delivered **in hospitals** (0–100).
   * Represents how hospital vs retail the product is.
   * Can be `nan` (missing).

5. **`main_package`**

   * Most common dispensing format, e.g.:

     * `PILL`,
     * `INJECTION`,
     * `EYE DROP`.

6. **`biological`**

   * Boolean:

     * `True` if the drug is a **biologic** (derived from living organisms; proteins, antibodies, nucleic acids),
     * `False` otherwise.

7. **`small_molecule`**

   * Boolean:

     * `True` if the drug is a **small molecule** (low-molecular-weight, chemically synthesized),
     * `False` otherwise.

Example rows:

| country      | brand_name | ther_area           | hospital_rate | main_package | biological | small_molecule |
| ------------ | ---------- | ------------------- | ------------- | ------------ | ---------- | -------------- |
| COUNTRY_0024 | BRAND_1143 | Sensory_organs      | 0.09          | EYE DROP     | False      | True           |
| COUNTRY_0024 | BRAND_1865 | Muscoskeletal_Rheu… | 92.36         | INJECTION    | False      | False          |
| COUNTRY_0024 | BRAND_240F | Antineoplastic_and… | 36.94         | PILL         | False      | True           |
| COUNTRY_0024 | BRAND_2F6C | Antineoplastic_and… | 0.01          | INJECTION    | True       | False          |
| COUNTRY_0024 | BRAND_3A67 | Nervous_system      | nan           | PILL         | False      | False          |
| …            | …          | …                   | …             | …            | …          | …              |

Use cases:

* Feature engineering:

  * one-hot encode `ther_area` and `main_package`,
  * directly use `hospital_rate`,
  * binary indicators for `biological` and `small_molecule`.
* Analytical stratification:

  * comparing erosion patterns by:

    * therapeutic area,
    * hospital dependence,
    * biologic vs small molecule.
* These features help explain **why** certain markets end up in B1 vs B2.

---

## 5. Evaluation Metrics

### 5.1 Summary of metrics and weighting

The evaluation has two main stages, each with its own metric:

* **Phase 1-a (Scenario 1)**:

  * Forecast months 0–23.
  * **Per-brand Prediction Error (PE_j)** uses four components:

    * monthly error over 0–23 (20%),
    * cumulative error 0–5 (50%),
    * cumulative error 6–11 (20%),
    * cumulative error 12–23 (10%),
    * all normalized by (Avg_j).

* **Phase 1-b (Scenario 2)**:

  * Forecast months 6–23.
  * **Per-brand (PE_j)** uses three components:

    * monthly error 6–23 (20%),
    * cumulative error 6–11 (50%),
    * cumulative error 12–23 (30%),
    * all normalized by (Avg_j).

In both phases:

* Per-brand errors (PE_j) are aggregated across test brands, separated by **Bucket 1** and **Bucket 2**.
* The **final Prediction Error (PE)** for the phase is:

[
PE =
\frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1}
+
\frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}
]

This means:

* Average error in **Bucket 1 (high erosion)** is weighted **twice** as strongly as average error in Bucket 2.
* Bucket 1 is the **primary evaluation focus**.

---

### 5.2 Pre-generic normalization: (Avg_j)

For any brand–country combination (j), define:

[
Avg_j = \frac{\sum_{i=-12}^{-1} Y^{act}_{j,i}}{12}
]

* (Y^{act}_{j,i}): actual monthly volume for (j) in pre-Gx month (i).
* (i = -12,\dots,-1): 12 months before generic entry.

All error components are normalized by (Avg_j) to ensure:

* Errors are **scale-free** and comparable across brands of different sizes.
* A 10,000-unit error is penalized differently for a small brand vs a blockbuster.

---

### 5.3 Phase 1-a (Scenario 1) – Per-brand Prediction Error

**Scenario 1 context:**
Forecast months 0–23 with **no post-Gx actuals** available at prediction time.

The per-brand error (PE_j) is:

[
\begin{aligned}
PE_j = &;0.2\left(
\frac{\sum_{i=0}^{23} \left|Y^{act}*{j,i} - Y^{pred}*{j,i}\right|}
{24 \cdot Avg_j}
\right) \
&+ 0.5\left(
\frac{\left|\sum_{i=0}^{5} Y^{act}*{j,i} - \sum*{i=0}^{5} Y^{pred}*{j,i}\right|}
{6 \cdot Avg_j}
\right) \
&+ 0.2\left(
\frac{\left|\sum*{i=6}^{11} Y^{act}*{j,i} - \sum*{i=6}^{11} Y^{pred}*{j,i}\right|}
{6 \cdot Avg_j}
\right) \
&+ 0.1\left(
\frac{\left|\sum*{i=12}^{23} Y^{act}*{j,i} - \sum*{i=12}^{23} Y^{pred}_{j,i}\right|}
{12 \cdot Avg_j}
\right)
\end{aligned}
]

Components:

1. **Monthly error (0–23), weight 0.2**

   * Average absolute error per month over the full 24 months,
   * normalized by (Avg_j).

2. **Cumulative error (0–5), weight 0.5**

   * Absolute difference between **total actual** and **total predicted** volumes in months 0–5,
   * normalized by (6 \cdot Avg_j).
   * Dominant term (50%): early erosion is most critical.

3. **Cumulative error (6–11), weight 0.2**

   * Same structure for months 6–11.

4. **Cumulative error (12–23), weight 0.1**

   * Same structure for months 12–23 (second year).

The combined error balances:

* **Shape fidelity** (monthly term),
* **Business-critical blocks** (0–5, 6–11, 12–23) with decreasing importance over time.

---

### 5.4 Phase 1-a – Aggregation across buckets

After computing (PE_j) for all test brands in Scenario 1:

* Partition into:

  * (n_{B1}) brands in **Bucket 1 (high erosion)**,
  * (n_{B2}) brands in **Bucket 2 (low/medium erosion)**.

* The final **Scenario-1 Prediction Error** is:

[
PE =
\frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1}
+
\frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}
]

Where:

* (PE_{j,B1}) is the per-brand error for brand (j) in B1,
* (PE_{j,B2}) is the per-brand error for brand (j) in B2.

The term for B1 is multiplied by 2, explicitly **up-weighting high-erosion markets**.

---

### 5.5 Phase 1-b (Scenario 2) – Per-brand Prediction Error

**Scenario 2 context:**
At evaluation time, months 0–5 post-Gx are observed; the model must forecast **months 6–23**.

The per-brand error (PE_j) is:

[
\begin{aligned}
PE_j = &;0.2\left(
\frac{\sum_{i=6}^{23} \left|Y^{act}*{j,i} - Y^{pred}*{j,i}\right|}
{18 \cdot Avg_j}
\right) \
&+ 0.5\left(
\frac{\left|\sum_{i=6}^{11} Y^{act}*{j,i} - \sum*{i=6}^{11} Y^{pred}*{j,i}\right|}
{6 \cdot Avg_j}
\right) \
&+ 0.3\left(
\frac{\left|\sum*{i=12}^{23} Y^{act}*{j,i} - \sum*{i=12}^{23} Y^{pred}_{j,i}\right|}
{12 \cdot Avg_j}
\right)
\end{aligned}
]

Components:

1. **Monthly error (6–23), weight 0.2**

   * Average absolute error per month over the 18-month prediction window.

2. **Cumulative error (6–11), weight 0.5**

   * Absolute difference between total actual and total predicted volumes for months 6–11.

3. **Cumulative error (12–23), weight 0.3**

   * Same idea for months 12–23, with higher weight than in Scenario 1 (30%).

All errors are normalized by (Avg_j).

Notably, there is **no error term for months 0–5** because those months are **observed inputs** and not forecast in Scenario 2.

---

### 5.6 Phase 1-b – Aggregation across buckets

As in Phase 1-a, per-brand errors for Scenario 2 are aggregated bucket-wise:

[
PE =
\frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1}
+
\frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}
]

* **Bucket 1 (high erosion)** average error is weighted twice as much as Bucket 2.
* This again reinforces the **primary importance** of accurate predictions for high-erosion markets.

---

## 6. Summary

Putting everything together:

* The **BCN Digital Finance Hub** is a rapidly grown, multidisciplinary Novartis team in Barcelona, combining data science, finance, and engineering, embedded in a strong local tech ecosystem.
* The Datathon is focused on the **post-generic-entry phase** of a drug’s lifecycle, where **volume erosion** occurs.
* Erosion is quantified through **Mean Generic Erosion**, and markets are grouped into **Bucket 1 (high erosion)** and **Bucket 2 (medium/low erosion)**.
* Participants must:

  * Build forecasting models for **Scenario 1** (0–23 months post-Gx, no post-Gx actuals) and **Scenario 2** (6–23 months with 0–5 known).
  * Carry out a **business-oriented exploratory analysis**, especially on **high-erosion cases**.
* Data provided include:

  * `df_volume.csv` (monthly volumes around Gx),
  * `df_generics.csv` (number of generics over time),
  * `df_medicine_info.csv` (drug-level metadata by country and brand),
    along with a clearly defined train/test split and scenario assignments.
* Evaluation in **Phase 1** is driven by rigorously defined, **normalized prediction error metrics**, which:

  * place strong weight on early-post-Gx cumulative volume,
  * and weight high-erosion bucket errors twice as strongly as low-erosion ones.
* **Phase 2** complements quantitative performance with a Jury evaluation of methodology and insights to select the **top 3 winning teams**.
