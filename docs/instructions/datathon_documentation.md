# Forecasting Generic Erosion Datathon

## Complete Technical & Business Documentation

---

## 1. Business Context: The Generics Problem

### 1.1 Patents, Exclusivity, and Loss of Exclusivity (LoE)

In the pharmaceutical industry, when a company develops a new medicine, it usually protects its innovation via a **patent**. A patent grants the originator company **exclusive rights** to manufacture and commercialize the drug for a defined period (commonly around 20 years from filing, though effective market exclusivity may be shorter due to development time).

Once this protection ends, the product reaches its **Loss of Exclusivity (LoE)** date. From this moment:

* Legal barriers preventing competitors from selling equivalent products are removed.
* **Generic manufacturers** can enter the market with copies of the original drug.

The timeline is therefore:

> Innovation → Patent protection / Exclusivity → LoE → Open competition.

This transition is central to the Datathon: after LoE, the original brand typically faces a sharp decline in sales due to generic competition, a phenomenon called **generic erosion**.

---

### 1.2 What is a Generic Drug?

A **generic drug** is defined as a pharmaceutical product that is equivalent to its **reference (brand-name) product** in the following respects:

* **Dosage form** (e.g., tablet, capsule, vial)
* **Strength** (e.g., 80 mg)
* **Route of administration** (oral, intravenous, subcutaneous, etc.)
* **Quality** (purity, stability, manufacturing standards)
* **Performance** (how it behaves in the body)
* **Intended use** (indications and patient population)

Generic products may have different **inactive ingredients** (excipients), such as fillers, binders, coatings, or colorants. These must not alter the **therapeutic effect** of the medicine. Regulatory agencies focus on ensuring that the **active substance** behaves equivalently in the body.

---

### 1.3 Development Costs and Bioequivalence

Generic manufacturers benefit from **much lower development costs** compared to originator companies:

* The originator has already conducted extensive **pre-clinical and clinical trials** to demonstrate efficacy and safety.
* Generics do **not** repeat the full clinical program. Instead, they must show **bioequivalence** to the originator product.

**Bioequivalence** means that the generic and the reference product show **no relevant difference** in how the active ingredient becomes available in the body when administered at the same dose. This is evaluated through **pharmacokinetic properties**:

* **Absorption** – how quickly and how much drug enters the bloodstream.
* **Distribution** – how the drug is distributed across tissues.
* **Metabolism** – how the body transforms the drug (e.g., in the liver).
* **Elimination** – how the drug and its metabolites are removed (e.g., via kidneys).

Typical bioequivalence studies:

* Healthy volunteers receive both the reference and generic product (often in a crossover design).
* Blood samples are collected over time to create concentration–time profiles.
* Key parameters such as **Cmax** (maximum concentration) and **AUC** (area under the curve) are compared.
* If these fall within an accepted range (commonly 80–125% ratio limits), the products are considered bioequivalent.

Because only these focused studies are required, generic manufacturers can enter the market with **lower R&D cost**, which ultimately allows **lower prices**.

---

### 1.4 Market Impact of Generics

The entry of generics usually has profound consequences:

1. **Increased competition**

   * Multiple manufacturers begin producing the same active ingredient.
   * Competition on price and market share intensifies, putting downward pressure on prices.

2. **Improved affordability**

   * Generic medicines are typically sold at **lower prices** than their branded counterparts.
   * This reduces costs for patients and healthcare systems, especially for chronic treatments.

3. **Greater access to treatment**

   * Lower prices mean more patients can afford therapy.
   * This can improve adherence, broaden access, and lead to better overall **disease management outcomes**.

4. **Prescribing and substitution practices**

   * Physicians may prescribe generics directly.
   * In many healthcare systems, **pharmacists** are allowed (or required) to substitute a brand-name prescription with a generic equivalent, provided regulations and prescriber instructions allow it.
   * This accelerates the shift from the originator brand to generics after LoE.

---

### 1.5 Example: Diovan (valsartan)

A well-known internal example at Novartis is **Diovan**, whose active ingredient is **valsartan**, used in the treatment of **hypertension** and **heart failure**.

* When Diovan’s patent expired in **2012**, generics of valsartan entered the market.
* Multiple manufacturers launched generic versions, significantly increasing competition.
* As a result, prices dropped and **originator sales decreased sharply**, illustrating the generic erosion phenomenon.

---

### 1.6 Role of the Novartis Digital Finance Hub

The **Novartis Digital Finance Hub** applies advanced analytics and data science to Novartis’ financial processes. Among its responsibilities:

* **Forecast future sales**, including anticipating the impact of generic entries.
* Support **strategic planning**, financial risk management, and resource allocation.
* Provide country organizations with tools to generate **monthly and annual sales forecasts**, which feed into consolidated, company-wide financial accounting.

The Datathon challenge you are working on is set in this context: **predicting the sales erosion after generic entry** to support better financial planning and decision-making.

---

## 2. Lifecycle of a Drug and Generic Erosion

### 2.1 Drug Lifecycle Phases

A pharmaceutical product’s sales typically follow a **product lifecycle** with distinct phases:

1. **Introduction (Launch)**

   * Sales volume is initially **low**: physicians and patients are still becoming aware of the drug; reimbursement and guidelines may be ramping up.

2. **Growth**

   * As awareness and acceptance increase, sales **grow rapidly**.
   * New indications, promotional activities, and clinical guideline inclusion can accelerate adoption.

3. **Maturity**

   * Sales reach a **stable or plateau level**.
   * Most potential prescribers are aware; the market is relatively saturated.
   * Growth slows and stabilizes.

4. **Competitive Pressure & Decline**

   * New branded competitors may enter the same therapeutic area, slowing growth or causing gradual decline.
   * The major turning point is **LoE**, after which **generic entry** typically causes a sharper decline.

---

### 2.2 Generic Erosion

**Generic erosion** refers to the **rapid decline in the originator’s sales volume** after generic competitors enter the market. Typically:

* At **generic entry (Gx)**, prices and volumes begin to shift rapidly.
* The originator’s volume often shows a steep drop in the first months.
* After some time, the market may reach a new **lower equilibrium** where the originator retains a smaller share and generics dominate.

The Datathon is specifically focused on **modeling and forecasting this generic erosion**, particularly in the first **24 months** after generic entry.

---

### 2.3 Illustrative Lifecycle Figure (Figure 1.1)

Although not reproduced here as an image, Figure 1.1 depicts:

* Vertical axis: **Volume of sales**.
* Horizontal axis: **Time**.
* Key annotated milestones on the sales curve:

  * **Drug launch**
  * **New indication entry** (boosting sales)
  * **New competitor entry** (flattening growth)
  * **Loss of Exclusivity (LoE)** (peak region)
  * **Generics entry (Gx)** (aligned with LoE on the time axis)
* The post-generic region is highlighted as the **Datathon focus**, showing a sharp drop in volume and the subsequent low-level plateau.

This diagram visually summarizes the entire lifecycle and shows where the **Datathon problem window** lies.

---

## 3. Mean Generic Erosion and Erosion Classification

### 3.1 Mean Generic Erosion – Definition

To quantify generic erosion, we define the **Mean Generic Erosion (MGE)** as follows:

1. Compute the **baseline average monthly volume** in the 12 months **before** generic entry.
2. Normalize each monthly volume in the **24 months after generic entry** by this baseline.
3. Compute the **average** of these 24 normalized values.

Formally, for a given country–brand combination ( j ):

* Let ( y_{j,i}^{\text{act}} ) be the **actual volume** in month ( i ), where:

  * ( i = -12, \dots, -1 ) are the 12 months **before** generic entry,
  * ( i = 0, \dots, 23 ) are the 24 months **after** generic entry.

1. **Pre-generic baseline** (Equation 1.3):

[
\text{Avg}*j = \frac{1}{12} \sum*{i=-12}^{-1} y_{j,i}^{\text{act}}.
]

2. **Normalized post-generic volume** (Equation 1.2):

[
\text{vol}*{j,i}^{\text{norm}} = \frac{\text{Vol}*{j,i}}{\text{Avg}_j},
]

where (\text{Vol}_{j,i}) is the actual volume in month (i) after generic entry (typically (i = 0,\dots,23)).

3. **Mean Generic Erosion** (Equation 1.1):

[
\text{Mean Generic Erosion}*j = \frac{1}{24} \sum*{i=0}^{23} \text{vol}_{j,i}^{\text{norm}}.
]

Interpretation:

* A value **close to 1** means that, on average, post-generic volumes stay near the pre-generic baseline.
* A value **close to 0** means that volumes have largely collapsed compared to the pre-generic level.
* Because volumes are normalized, MGE enables **comparison across products** with different absolute sales levels.

---

### 3.2 Conceptual Erosion Levels

Conceptually, drugs can be grouped by how severe their erosion is:

1. **Low Erosion**

   * Volume remains relatively stable after generic entry.
   * Mean normalized erosion is **close to 1**.

2. **Medium Erosion**

   * Moderate decline in volume after generics.
   * Mean normalized erosion is between **0 and 1** (neither near 0 nor near 1).

3. **High Erosion**

   * Sharp drop in volume post-entry.
   * Mean normalized erosion is **close to 0** (only a small fraction of pre-generic volume remains).

---

### 3.3 Buckets for the Datathon

For modeling and evaluation purposes, the Datathon uses **two numeric buckets** based on Mean Generic Erosion:

* **Bucket 1 (B1) – High Erosion**

  * Mean erosion ( \in [0, 0.25] ).
  * These products are considered **high erosion** and are the primary focus of the Datathon.

* **Bucket 2 (B2) – Low/Medium Erosion**

  * Mean erosion ( \in (0.25, 1] ).
  * Includes products with moderate or low erosion.

These buckets are used later in the **final metric aggregation**, where Bucket 1 is given **double weight** compared to Bucket 2.

---

### 3.4 Illustrative Classification Figure (Figure 1.2)

Figure 1.2 shows idealized time-series patterns:

* Vertical axis: **Volume** (normalized).
* Horizontal axis: **Month_num** (relative to generic entry).
* A vertical dashed line marks the **Generics entry date**.

Three curves are shown:

* **Blue (Low Erosion):** Volume declines slightly and remains relatively high.
* **Yellow (Medium Erosion):** Clear decline but not catastrophic.
* **Red (High Erosion):** Very steep drop immediately after generic entry, staying near zero thereafter.

On the right, the buckets are summarized:

* **Bucket 2 (B2):** Mean Erosion ( \in (0.25, 1] ).
* **Bucket 1 (B1):** Mean Erosion ( \in [0, 0.25] ).

This figure provides an intuitive understanding of how different erosion profiles map into the two buckets.

---

## 4. The Forecasting Challenge

### 4.1 Problem Statement

The Datathon’s core **technical goal** is:

> To **forecast the monthly sales volume** of a drug in the **24 months following generic entry**, i.e., to model **volume erosion** after generics enter the market.

---

### 4.2 Two Forecasting Scenarios

Forecasts must be produced under **two real-world business scenarios**:

#### Scenario 1 – Forecast at LoE (no post-generic data)

* Forecast is made **immediately at generic entry (Month 0)**.
* **No actual post-entry data** are available.
* The task is to forecast monthly volumes from **Month 0 to Month 23**.
* Only **pre-generic history** and product/contextual features are available.
* This simulates the planning need at the moment of LoE: “What will happen over the next two years?”

#### Scenario 2 – Forecast after six months of erosion

* Forecast is made **six months after generic entry**.
* Participants have:

  * **Six months of post-entry actuals** (Months 0–5).
* The task is to forecast monthly volumes from **Month 6 to Month 23**.
* This reflects an **updated forecast** once early erosion behavior is observable.

In both scenarios, the target is the **volume trajectory**, not only the end state.

---

### 4.3 Business Dimension and Expectations

The challenge is **not purely technical**; it also has a strong **business dimension**:

* Participants should explain **why** they chose certain modeling strategies in terms of:

  * Business relevance,
  * Interpretability,
  * Practical applicability in a real finance/forecasting environment.

Finalist teams are expected to:

* Perform **deep exploratory data analysis**, especially on:

  * **High-erosion cases** (Bucket 1).
* Clearly describe:

  * Their **data preprocessing** logic,
  * Feature engineering approaches,
  * Modeling choices and trade-offs.
* Emphasize clear, **business-oriented visualizations** to make findings:

  * Understandable,
  * Interpretable,
  * Actionable for non-technical stakeholders.

Visualization and storytelling are integral to a successful solution.

---

### 4.4 Example Time Series (Figure 2.1)

Figure 2.1 illustrates a real-world volume profile around generic entry for a specific brand–country combination:

* Title:
  **“Volume Evolution Around Generic Entry — BRAND_75FD (COUNTRY_A67D)”**
* Axes:

  * Y-axis: **Volume**.
  * X-axis: **Months Post Generic Entry**, including negative (pre-generic) and positive (post-generic) months.
* A vertical dashed line at Month 0 marks the **generic entry date**.
* The background is partitioned:

  * Left (blue shading): **Pre-generic**.
  * Right (red shading): **Post-generic**.

The plotted line shows:

* Stable, high volume before generic entry (~50k–70k units per month).
* A **sharp drop** right after entry.
* A low, stable plateau at a much lower level in the following months.

This is a typical **high-erosion** pattern (Bucket 1).

---

## 5. Evaluation Process

The Datathon uses a **two-phase evaluation**:

1. **Phase 1 – Model Evaluation** (quantitative).
2. **Phase 2 – Jury Evaluation** (qualitative and business-oriented).

### 5.1 Phase 1 – Model Evaluation

All teams must submit predictions for the **entire test dataset**, which includes both Scenario 1 and Scenario 2 observations.

#### Step 1 – Phase 1-a: Scenario 1 Accuracy

* All teams are evaluated on **Scenario 1** predictions.
* Teams are **ranked** based on their **prediction error** (the custom metric described in Section 7).
* The **top 10 teams** with lowest error proceed to Step 2.

#### Step 2 – Phase 1-b: Scenario 2 Accuracy

* Only the **top 10 teams** are evaluated on **Scenario 2**.
* They are ranked by their Scenario 2 prediction error.
* The **top 5 teams** advance to **Phase 2**.

Thus, teams must perform well **both with no post-generic data (Scenario 1)** and with **partial post-generic data (Scenario 2)** to reach the final.

---

### 5.2 Phase 2 – Jury Evaluation

The final **five teams** present their solutions to a **jury panel** composed of both **technical and business experts**.

Teams must present:

* Their **methodology** and model architectures,
* **Data preprocessing** and feature engineering,
* Key **insights** and patterns discovered in the data,
* **Business conclusions** and recommendations.

The jury then selects the **top three winning teams** based on the combination of:

* Quantitative performance (Phase 1),
* Clarity, rigor, and business relevance of the presentation (Phase 2).

---

## 6. Data Description

### 6.1 Overall Structure: Train and Test

The dataset includes **2,293 country–brand combinations** (i.e., unique pairs of brand and country that experienced a generic entry). For each pair, monthly volumes are provided.

These combinations are split into:

* **Training Set – 1,953 observations**

  * Each observation is one country–brand pair.
  * For each:

    * Up to **24 months pre-generic** data,
    * Up to **24 months post-generic** data.
  * Used to learn generic erosion patterns and train models.

* **Test Set – 340 observations**

  * Also country–brand pairs, used solely for evaluation.
  * Split into two parts:

    * **Scenario 1 – 228 observations:** forecast Month 0–23 with no post-generic data.
    * **Scenario 2 – 112 observations:** forecast Month 6–23, with Months 0–5 provided.

Understanding this split is crucial for designing models aligned with the two scenarios.

---

### 6.2 The Three Main DataFrames

Data is delivered in **three main CSV files**:

1. **Volume Dataset – `df_volume.csv`**
2. **Generics Dataset – `df_generics.csv`**
3. **Medicine Information Dataset – `df_medicine_info.csv`**

These can be joined using `country` and `brand_name` keys.

---

### 6.3 Volume Dataset (`df_volume.csv`)

This dataset contains the **time series of sales volumes** around generic entry.

Fields:

* **`country`**

  * Market or country identifier.

* **`brand_name`**

  * Name or code of the branded product.

* **`month`**

  * Calendar month of the observation (e.g., year–month).

* **`months_postgx`**

  * Number of months relative to generic entry:

    * `0` = generic entry month,
    * Negative = months **before** entry,
    * Positive = months **after** entry.

* **`volume`**

  * Number of units sold in that month for the given country–brand pair.
  * This is the **target variable** in all forecasting tasks.

All historical sales are at **monthly granularity**, starting either at the **brand launch** or the **first available data point** for that series.

---

### 6.4 Generics Dataset (`df_generics.csv`)

This dataset describes the **evolution of generic competition** over time.

Fields:

* **`country`**, **`brand_name`**

  * Identify the branded originator drug–market pair.

* **`months_postgx`**

  * Same definition as in `df_volume.csv`.

* **`n_gxs`**

  * Number of **generic competitors** available in that month for the given country–brand pair.

Key points:

* `n_gxs` can change over time:

  * 0 before entry,
  * > 0 after entry,
  * Possibly increasing as more generics launch or decreasing if some leave the market.
* This variable is a critical **time-varying feature**, capturing **competitive intensity**.

---

### 6.5 Medicine Information Dataset (`df_medicine_info.csv`)

This dataset contains **contextual and largely time-invariant attributes** for each product.

Fields:

* **`country`**, **`brand_name`**

  * Keys used to join with other datasets.

* **`therapeutic_area`**

  * Therapeutic category of the medicine (e.g., cardiovascular, oncology).

* **`hospital_rate`**

  * Proportion of units distributed through hospitals (vs. retail channels).

* **`main_package`**

  * The dominant commercial format (e.g., pills, vials, packs).

* **`biological`** (boolean)

  * Indicates if the product is a **biologic**.

* **`small_molecule`** (boolean)

  * Indicates if the product is a **small-molecule synthetic** drug.

These variables help in:

* Segmenting products (e.g., by therapeutic area or molecule type),
* Explaining differences in erosion patterns,
* Building more informative predictive models.

**Note:** Categorical variables in this dataset can be assumed **constant over time**.

---

### 6.6 Additional Modeling Guidelines and Constraints

The documentation includes several important practical notes:

1. **Use of data for training**

   * You may **use all provided data** for model training.
   * For example, rows belonging to Scenario 2 in the test set may be used when training a model for Scenario 1.
   * The official evaluation will still be based on the held-out labels.

2. **Bucket information**

   * The dataset does **not** include bucket labels (B1/B2).
   * However, you can derive them using **Equation 1.1** (Mean Generic Erosion).

3. **Freedom of modeling approach**

   * Any modeling approach is allowed (statistical, ML, deep learning, hybrid).
   * Nevertheless, **explainability and simplicity** are explicitly valued, especially for business users.

4. **Units of volume**

   * `volume` may be expressed in different units (e.g., packs, pills, milligrams) depending on the country–brand.
   * Direct comparison of raw volumes across products may be misleading.
   * Normalization (e.g., via Mean Generic Erosion or per-series scaling) is recommended.

5. **Missing values**

   * Some columns may contain missing values.
   * It is up to participants to decide whether to:

     * Leave them as missing (if the model supports it),
     * Impute (e.g., mean, median, forward-fill),
     * Or apply other preprocessing strategies.
   * These decisions must be clearly explained and justified.

---

## 7. Evaluation Metric: Prediction Error (PE)

### 7.1 Metric Overview

Model accuracy in Phase 1 is assessed using a **custom Prediction Error (PE)** metric, specifically designed to:

* Capture **short- and long-term** deviations between predicted and actual volumes,
* Emphasize **early post-generic periods**,
* Account for **seasonality and month-to-month variation**,
* Emphasize **high-erosion cases** (Bucket 1) via bucket-level weighting.

All error components are normalized by the **pre-generic average volume** ( \text{Avg}_j ), ensuring comparability across different scales.

---

### 7.2 Scenario 1 Metric – Phase 1-a (Equation 4.1)

#### 7.2.1 Description

In **Scenario 1**:

* Predictions are required for **Months 0–23** (24 months post-entry),
* No actual post-entry data are available at forecasting time.

To evaluate performance, the metric for each series ( j ) combines four components:

1. **Absolute monthly error for all 24 months** – weight 20%
   Captures average month-by-month deviation.

2. **Absolute accumulated error for Months 0–5** – weight 50%
   Measures error in total volume over the first 6 months post-entry (early erosion).

3. **Absolute accumulated error for Months 6–11** – weight 20%
   Measures error in total volume in the latter half of the first post-entry year.

4. **Absolute accumulated error for Months 12–23** – weight 10%
   Measures longer-term cumulative error in the second post-entry year.

All four components are normalized by ( \text{Avg}_j ), defined as in Equation 1.3.

#### 7.2.2 Formula

For country–brand combination ( j ), Scenario-1 error ( PE_j ) is:

[
\begin{aligned}
PE_j &= 0.2 \left( \frac{\sum_{i=0}^{23} \left| Y^{\text{act}}*{j,i} - Y^{\text{pred}}*{j,i} \right|}{24 \cdot \text{Avg}*j} \right) \
&\quad + 0.5 \left( \frac{\left| \sum*{i=0}^{5} Y^{\text{act}}*{j,i} - \sum*{i=0}^{5} Y^{\text{pred}}*{j,i} \right|}{6 \cdot \text{Avg}*j} \right) \
&\quad + 0.2 \left( \frac{\left| \sum*{i=6}^{11} Y^{\text{act}}*{j,i} - \sum_{i=6}^{11} Y^{\text{pred}}*{j,i} \right|}{6 \cdot \text{Avg}*j} \right) \
&\quad + 0.1 \left( \frac{\left| \sum*{i=12}^{23} Y^{\text{act}}*{j,i} - \sum_{i=12}^{23} Y^{\text{pred}}_{j,i} \right|}{12 \cdot \text{Avg}_j} \right).
\end{aligned}
]

Where:

* ( Y^{\text{act}}_{j,i} ) = actual volume in month ( i ),
* ( Y^{\text{pred}}_{j,i} ) = predicted volume in month ( i ),
* ( \text{Avg}_j ) = average pre-generic monthly volume for series ( j ).

---

### 7.3 Scenario 2 Metric – Phase 1-b (Equation 4.2)

#### 7.3.1 Description

In **Scenario 2**:

* Actual volumes are known for **Months 0–5**,
* Predictions are required for **Months 6–23** (18 months).

The metric includes three components:

1. **Absolute monthly error for Months 6–23** – weight 20%
   Average monthly deviation across the 18 forecast months.

2. **Absolute accumulated error for Months 6–11** – weight 50%
   Error in total volume over Months 6–11 (first evaluated half-year).

3. **Absolute accumulated error for Months 12–23** – weight 30%
   Error in total volume in the second forecast year.

Again, all components are normalized by ( \text{Avg}_j ).

#### 7.3.2 Formula

For series ( j ) in Scenario 2, the error ( PE_j ) is:

[
\begin{aligned}
PE_j &= 0.2 \left( \frac{\sum_{i=6}^{23} \left| Y^{\text{act}}*{j,i} - Y^{\text{pred}}*{j,i} \right|}{18 \cdot \text{Avg}*j} \right) \
&\quad + 0.5 \left( \frac{\left| \sum*{i=6}^{11} Y^{\text{act}}*{j,i} - \sum*{i=6}^{11} Y^{\text{pred}}*{j,i} \right|}{6 \cdot \text{Avg}*j} \right) \
&\quad + 0.3 \left( \frac{\left| \sum*{i=12}^{23} Y^{\text{act}}*{j,i} - \sum_{i=12}^{23} Y^{\text{pred}}_{j,i} \right|}{12 \cdot \text{Avg}_j} \right).
\end{aligned}
]

As in Scenario 1, the metric blends monthly-level and cumulative-level performance, with higher emphasis on **earlier forecast months**.

---

### 7.4 Final Scenario-Level Metric and Buckets (Equation 4.3)

Once ( PE_j ) is computed for each series:

* Each country–brand pair appears **in only one scenario** (either Scenario 1 or Scenario 2).
* All series are classified into Bucket 1 or Bucket 2 based on **Mean Generic Erosion**.

Let:

* ( n_{B1} ) = number of series in **Bucket 1**,
* ( n_{B2} ) = number of series in **Bucket 2**,
* ( PE_{j,B1} ) = error for series ( j ) in Bucket 1,
* ( PE_{j,B2} ) = error for series ( j ) in Bucket 2.

The final scenario-level score ( PE ) is:

[
PE = \frac{2}{n_{B1}} \sum_{j=1}^{n_{B1}} PE_{j,B1} ;+; \frac{1}{n_{B2}} \sum_{j=1}^{n_{B2}} PE_{j,B2}.
]

Key points:

* Bucket 1 (high erosion) receives **double the weight** of Bucket 2.
* This reflects the business importance of accurately forecasting **high-erosion drugs**.
* The final metric is a **weighted average error** across all series in the scenario.

---

### 7.5 Interpretation of PE Values

For an individual series:

* Ideally, ( 0 \leq PE_j \leq 1 ).

  * ( PE_j = 0 ): perfect prediction (no deviation).
  * ( PE_j \approx 0 ): very accurate forecasts.
  * ( PE_j \approx 1 ): error magnitude comparable to the pre-generic baseline volume.
* ( PE_j > 1 ): very poor prediction (cumulative errors larger than typical pre-generic volumes).

At scenario level:

* If all ( PE_j = 0 ), then final ( PE = 0 ).
* If all ( PE_j = 1 ), then:
  [
  PE = 2 \cdot 1 + 1 \cdot 1 = 3.
  ]
* Because some ( PE_j ) can exceed 1, **scenario-level PE can also exceed 3**.

In all cases, **lower values are better**; teams compete to **minimize** this final scenario-level PE.

---

## 8. Metric Rationale

The Prediction Error metric is designed to reflect **three key dimensions** of the forecasting problem:

1. **Generic Erosion Emphasis**

   * Implemented via **bucket-level weighting** (Equation 4.3).
   * High-erosion cases (Bucket 1) are weighted **twice** as much as Bucket 2.
   * This ensures that models are optimized for the most critical and challenging products, where mis-forecasting can lead to significant financial impact.

2. **Time Since Generic Entry**

   * Captured via **differential weighting** across time segments in Equations 4.1 and 4.2:

     * Early post-entry months carry the highest weight (50%).
     * Later months have lower but non-zero weight.
   * This mirrors real business priorities: the first months after LoE are particularly crucial for planning and risk management.

3. **Seasonality and Short-Term Fluctuations**

   * The monthly-error components in both Scenario 1 and Scenario 2 formulas ensure that:

     * The model is evaluated on **month-to-month fit**, not just on long-run totals.
     * Seasonal peaks and troughs and local fluctuations influence the score.

Participants should therefore aim to build models that:

* **Capture early erosion dynamics** accurately,
* **Stabilize** into realistic long-term patterns,
* **Respect seasonality and short-term variations**,
* Perform especially well on **high-erosion products**.

Minimizing the final Prediction Error under these conditions will produce forecasts that are both **statistically robust** and **business-relevant**.

---

## 9. Summary

This documentation has outlined:

* The **business context** of generics, patents, and generic erosion.
* The **drug lifecycle** and the focus of the Datathon on the **post-generic** period.
* The definition of **Mean Generic Erosion**, **erosion categories**, and **buckets** (B1, B2).
* The **forecasting challenge**, including the two scenarios and the business expectations.
* The **evaluation process** (two phases, with filtering based on performance).
* The **data structure**, including three main DataFrames and their fields.
* The **custom Prediction Error metric**, in both Scenario 1 and Scenario 2 variants, and the final **bucket-weighted aggregation**.
* The **rationale** behind the metric’s design, emphasizing high-erosion cases, early months after LoE, and seasonal patterns.
