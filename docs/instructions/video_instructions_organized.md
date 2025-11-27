# Novartis–Barcelona Digital Finance Hub Datathon

## Official Challenge Documentation

---

## 1. Introduction

This document provides a complete and structured description of the Novartis Datathon hosted at the **Barcelona Digital Finance Hub**. It summarizes:

* the context and objectives of the event,
* the business problem and its pharmaceutical background,
* the forecasting task and scenarios,
* the datasets provided,
* the evaluation methodology and scoring metrics,
* the communication channels, technical platform, and submission process,
* the timeline and expectations for participating teams.

No substantive point from the original brief has been omitted; this is intended as a single reference document for all participants.

---

## 2. Organizers, Mission, and Context

### 2.1 Barcelona Digital Finance Hub

The Datathon is organized by the **Barcelona Digital Finance Hub**, a center dedicated to applying data science and advanced analytics to financial processes within Novartis.

Key characteristics:

* **History and growth**

  * Started in **2018** with **10 team members**.
  * Grew to **55 people in 2025**, reflecting significant expansion in skills and scope.

* **Team composition (2025)**

  * **34 data scientists**
  * **8 finance professionals**
  * **15 experts** in:

    * data visualization
    * machine learning
    * software engineering
    * DevOps

* **Mission**
  The Hub’s mission is to:

  * harness **data science** and **technology** to transform and optimize financial processes globally,
  * push **big data** and **AI** to the forefront of financial innovation in the pharmaceutical industry,
  * act as a **bridge between finance and digital innovation**, enabling data-driven decision-making at scale.

### 2.2 Novartis and the Role of the Hub

At the core of this initiative stands **Novartis’ mission**:

> To reimagine medicine in order to improve people’s lives.

Concretely, Novartis is committed to:

* using **innovative science and technology** to address healthcare challenges,
* discovering and developing **breakthrough treatments**,
* finding new ways to deliver these treatments to as many patients as possible.

Novartis plays a pivotal role in:

* **innovative medicine**,
* **global drug development**,
* **technical operations**.

Within this broader context:

* The **Barcelona Digital Finance Hub** acts as the **link between finance and digital innovation**,
* It helps to **propel the pharmaceutical industry into the future** using:

  * data,
  * automation,
  * artificial intelligence.

### 2.3 Why Barcelona?

The choice of **Barcelona** is strategic:

* It is a **European hub for tech and AI talent**, with a thriving tech ecosystem.
* It hosts major research infrastructures such as the **Barcelona Supercomputing Center**.
* It has strong **university programs in data science and mathematics**.
* It is an attractive city for **international talent**, making it an ideal home for a growing **digital finance community**.

---

## 3. Business Background: Drug Lifecycle and Generic Erosion

The business problem in this Datathon is rooted in the **commercial lifecycle of pharmaceutical products**, particularly the period after **patent expiry**.

### 3.1 Drug Sales Lifecycle

A typical drug’s commercial journey includes several phases:

1. **Launch phase**

   * The drug is first introduced to the market.
   * **Sales volume is initially low** because:

     * awareness is limited,
     * adoption by prescribers and patients is just beginning.

2. **Growth phase**

   * The product gains acceptance among prescribers and patients.
   * Sales volumes **increase rapidly**.

3. **Maturity phase**

   * The drug reaches a plateau in its market share.
   * Sales volumes **stabilize** at a relatively high level.

4. **Competitive pressure and decline**

   * New competitors may enter the market (e.g., other branded drugs).
   * This can **slow growth** or begin to **reduce volumes** before patent expiry.

5. **Loss of Exclusivity (LoE) and generic entry**

   * When the drug’s **patent expires**, **generic products** can enter the market.
   * Generics are usually cheaper and rapidly gain market share.
   * This often causes a **sharp drop in sales volume** for the original brand:

     * This phenomenon is called **generic erosion**.

### 3.2 Mean Generic Erosion: Definition

The Datathon focuses on quantifying and forecasting **generic erosion**.

**Mean generic erosion** is defined as:

* The **mean of the normalized sales volume** in the **24 months after generic entry**.

The normalization procedure:

1. Compute the **average monthly sales** over the **12 months before generic entry**.

2. For each month after generic entry (months 0 to 23), compute the ratio:

   [
   \text{Normalized Volume}_{t} =
   \frac{\text{Volume at month } t}{\text{Average monthly volume in the 12 months before generic entry}}
   ]

3. The **mean generic erosion** is the **average** of these normalized volumes across the **24 post-entry months**.

This creates a comparable measure of erosion across drugs with very different absolute sales magnitudes.

### 3.3 Erosion Levels and Buckets

Based on mean generic erosion, drugs can be classified into three conceptual **erosion patterns**:

1. **Low erosion drugs**

   * Experience **minimal impact** after generic entry.
   * Normalized volume remains close to **1**.
   * The mean normalized erosion is therefore **close to 1**, indicating stable volumes.

2. **Medium erosion drugs**

   * Show a **moderate decline** in sales after generic entry.
   * Mean normalized erosion lies between **0 and 1**, indicating partial loss.

3. **High erosion drugs**

   * Experience a **sharp drop in volume** after generics enter the market.
   * Mean normalized erosion is **close to 0**, reflecting severe loss of volume.

For the Datathon, brands are then grouped into **two numerical buckets**:

* **Bucket 1 – High Erosion**

  * Drugs with mean erosion between **0 and 0.25**.
  * These are **high-erosion brands** and constitute the **primary focus** of this Datathon.

* **Bucket 2 – Medium and Low Erosion**

  * Drugs with mean erosion **greater than 0.25** (up to 1).
  * This bucket contains both **medium-erosion** and **low-erosion** brands.

From a business perspective, understanding and predicting erosion, particularly for **Bucket 1**, is critical for:

* **Revenue forecasting**,
* **Portfolio and product planning**,
* **Strategic and pricing decisions**,
* **Managing the post-patent period**.

---

## 4. Datathon Problem Statement

### 4.1 Objective

The core technical task is to **forecast volume erosion** after generic entry, over a **24-month horizon** from the generic entry date.

The forecasting problem is defined under **two scenarios** that simulate real-world situations with different amounts of post-entry information.

### 4.2 Scenario 1 – Right After Generic Entry

* You are positioned **immediately after generic entry** (month 0).
* You have **no actual data** after the generic entry date.
* Task:

  * Forecast **monthly volumes** from **month 0 to month 23** (24 months).

This scenario simulates the situation where a country/brand has just lost patent protection and **early planning** is required without observed post-entry volumes.

### 4.3 Scenario 2 – Six Months After Generic Entry

* You are positioned **six months after generic entry**.
* You already have **actual volume data for the first 6 months** after generic entry (months 0–5).
* Task:

  * Forecast monthly volumes from **month 6 to month 23**.

This scenario represents a context where some post-entry behavior is already observed, and you must **update and refine** your forecasts accordingly.

### 4.4 Technical and Business Dimensions

The challenge is **not only** to build accurate models, but also to:

* justify **why** you chose particular modeling approaches,
* demonstrate a **business-oriented understanding** of generic erosion,
* connect technical decisions to **commercial implications**.

All teams presenting to the jury are expected to:

* provide a **deep exploratory analysis** of their **data preprocessing** and modeling choices,
* place **special focus on high-erosion cases** (Bucket 1),
* use **visualizations** liberally to make findings:

  * clear,
  * interpretable,
  * and aligned with business needs.

---

## 5. Data Structure and Datasets

### 5.1 Target and Observation Units

The **target variable** is:

* **Monthly volume** for **2,293 country–brand combinations** that have experienced a generic entry.

Each observation corresponds to a specific **country–brand pair** tracked over time (months).

#### Training Set

* Contains **1,953** country–brand combinations.
* For each combination, you are given:

  * up to **24 months of volume data before generic entry**, and
  * up to **24 months of volume data after generic entry**.

This structure allows you to analyze **pre- and post-entry dynamics**, and to learn how volumes evolve in response to generic competition.

#### Test Set

* Contains **340** country–brand combinations.
* Used for **evaluation** of your forecasts in the two scenarios.

Test set breakdown:

* **Scenario 1**:

  * ~two-thirds of the test set,
  * **228 observations**,
  * You must forecast volumes from **month 0 to 23** (no post-entry actuals available).

* **Scenario 2**:

  * ~one-third of the test set,
  * **112 observations**,
  * You must forecast volumes from **month 6 to 23** (actuals for months 0–5 are provided).

The **same erosion bucket structure** (Bucket 1 and Bucket 2) is maintained across both scenarios, ensuring consistency in evaluation. Scenario 2 simply contains **fewer total observations** due to the division of forecasting tasks.

On the **training side**, the almost 2,000 country–brand combinations represent the **full set of time series** available for modeling and exploration.

### 5.2 Datasets Provided

You will receive **three datasets**, each providing complementary information.

#### 5.2.1 Sales Volume Dataset

This dataset contains the **core time series** of sales volumes before and after generic entry.

Each row includes:

* **country**
  Market of reference (e.g., specific country).

* **brand_name**
  Name of the branded drug.

* **month**
  Calendar month of the observation (e.g., year–month).

* **months_post_gx**
  Number of months **relative** to generic entry:

  * 0 = month of generic entry,
  * negative values = months **before** generic entry,
  * positive values = months **after** generic entry.

* **volume**
  Number of units (drugs) sold in that month (target variable).

This dataset is central to the challenge, as the **volume** variable is what you will forecast under the two scenarios.

#### 5.2.2 DF_Generics Dataset

This dataset enriches the volume data with information about the **competitive environment** in terms of generics.

Each row includes:

* **country**
* **brand_name**
* **months_post_gx** (same definition as in the Sales Volume Dataset)
* **number_of_gx**
  Number of generic products available for that country–brand at that time.

Key points:

* The **number of generics** is **time-varying**.
* New generics can enter, and some may leave, leading to changes in **market competition** over time.

This dataset allows you to relate **volume erosion** to the **evolution of generic competition**.

#### 5.2.3 Drug Characteristics Dataset

This dataset provides **time-invariant characteristics** of each drug.

Each row includes:

* **country**

* **brand_name**

* **therapeutic_area**
  Therapeutic area in which the drug is used.

* **hospital_rate**
  Percentage of units delivered through **hospitals**.

* **main_package**
  Main product format (e.g., pills, vials).

* **biological** (Boolean)
  Indicates whether the drug is a **biological** product (derived from living organisms, e.g., proteins or antibodies).

* **small_molecule** (Boolean)
  Indicates whether the drug is a **small-molecule** compound (chemically synthesized, low molecular weight).

Important:

* These characteristics can be assumed **constant over time** for each country–brand pair.
* They can be used to:

  * segment the portfolio,
  * control for structural differences between products,
  * explain different erosion patterns across drugs.

---

## 6. Evaluation Framework

The evaluation process has **two main phases**:

1. **Phase 1 – Model Evaluation (Quantitative)**
2. **Phase 2 – Jury Evaluation (Qualitative + Quantitative)**

### 6.1 Phase 1 – Model Evaluation

In this phase, all teams submit predictions for the **entire test dataset**, which includes both Scenario 1 and Scenario 2. The phase is split into **two steps**:

#### Step 1 – Scenario 1 Evaluation

* All teams are evaluated based on their **prediction accuracy for Scenario 1** (forecasts from month 0 to 23, with no post-entry actuals).
* The **top 10 teams** with the **lowest prediction errors** in Scenario 1 advance to Step 2.

#### Step 2 – Scenario 2 Evaluation

* Only the **10 teams** selected in Step 1 are evaluated on **Scenario 2** (forecasts from month 6 to 23, with months 0–5 observed).
* Among these, the **top 5 teams** with the **lowest prediction errors** in Scenario 2 advance to **Phase 2**.

### 6.2 Phase 2 – Jury Evaluation

In Phase 2:

* The **5 finalist teams** present their:

  * methodology,
  * modeling choices,
  * data preprocessing and feature engineering strategies,
  * insights and conclusions.

* The jury is composed of both **technical and business experts**.

* After reviewing the presentations, the jury selects the **top 3 winning teams**.

The final decision is therefore based on a combination of:

* **Quantitative performance** (metrics from Phase 1), and
* **Qualitative assessment** (clarity, interpretability, business relevance, and innovation in Phase 2).

---

## 7. Scoring Metrics

Both Scenario 1 and Scenario 2 use metrics based on **prediction errors**, but they are defined over different time intervals and with different weightings.

All errors:

* are **normalized** by the **average monthly volume in the 12 months before generic entry**,
* are aggregated separately for **Bucket 1 (high erosion)** and **Bucket 2 (mid/low erosion)**,
* are then combined in a **weighted fashion**, with **Bucket 1 weighted twice as much** as Bucket 2.

### 7.1 Phase 1A – Metric for Scenario 1

In **Scenario 1**, participants must provide **24 months of predictions** with **no post-entry actuals**.

To evaluate predictions for each country–brand combination *j*, four error components are calculated:

1. **Monthly error across all 24 months**

   * Measures point-wise prediction accuracy from month 0 to 23.

2. **Accumulated error for months 0–4**

   * Focuses on the **first 5 months** after generic entry.
   * This is a crucial period where most of the **initial erosion dynamics** occur.

3. **Accumulated error for months 6–11**

   * Captures performance in the **medium horizon** after entry.

4. **Accumulated error for months 12–23**

   * Evaluates performance on the **longer-term horizon**.

These components are combined using weights so that:

* The **first 5 months (0–4)** receive the **highest importance**,
* The **middle horizon (6–11)** receives intermediate weight,
* The **global 24-month performance** also contributes but with less emphasis than the very early period.

All error components are **normalized** by the average monthly volume over the **12 months before generic entry**.

At the **bucket level**:

* For each bucket (1 and 2), the average prediction error is computed across all country–brand pairs in that bucket.
* To ensure independence from bucket size, each bucket’s total error is **divided by the number of pairs** in that bucket.

Then:

* The **overall team error** is computed as a **weighted sum of the bucket-level errors**, where:

  * **Bucket 1** errors are weighted **twice as much** as **Bucket 2** errors.

This reflects the strategic importance of **high-erosion (Bucket 1)** brands.

### 7.2 Phase 1B – Metric for Scenario 2

In **Scenario 2**, predictions are required from **month 6 to month 23**, with actual data available up to **month 5**.

For each country–brand combination *j*, three components are computed:

1. **Monthly error across months 6–23**

   * Measures prediction accuracy for all required forecast months.

2. **Accumulated error for months 6–11**

   * Focuses on the early part of the forecast horizon in this scenario.

3. **Accumulated error for months 12–23**

   * Evaluates performance in the later part of the horizon.

Again, all errors are **normalized** by the **average monthly volume in the 12 months before generic entry**.

The prediction error for each country–brand pair *j* is then computed as:

* **20%** of the **accumulated monthly error (months 6–23)**
* **50%** of the **accumulated error for months 6–11**
* **30%** of the **accumulated error for months 12–23**

So, in Scenario 2:

* The **early post-forecast segment (months 6–11)** is the most heavily weighted (50%).
* This is followed by the **later horizon (12–23)** with 30% weight.
* The overall monthly error (6–23) contributes 20%.

At the bucket level and overall team level, aggregation and weighting proceed in the same way as Scenario 1:

* Average errors are computed separately for **Bucket 1** and **Bucket 2**.
* Bucket totals are normalized by the number of country–brand pairs in each bucket.
* The final error is a **weighted sum**, with **Bucket 1 weighted twice as much** as Bucket 2.

---

## 8. Communication and Collaboration: Microsoft Teams

All **communication** between teams and mentors takes place through **Microsoft Teams**. Each participant has access via a user account.

Within Teams, two main channels are available:

### 8.1 Mentoring Channel

* This is a **private channel** for each team and its assigned mentors.

* Only:

  * members of that specific team, and
  * their mentors

  can access this channel.

* Use this channel to:

  * ask questions,
  * request feedback,
  * schedule and conduct mentoring meetings.

* **Mentoring meetings**:

  * Mentors will initiate meetings in this channel.
  * Teams join the meeting by clicking the **“Meet”** button at the agreed time.

### 8.2 Novartis Datathon Channel

This channel is intended for **general communication** across all participants.

It contains:

* A **General sub-channel**:

  * Only mentors can post messages here.
  * Used for **announcements**, **general information**, and **organisational updates** related to the Datathon.

* A **Files tab**, which contains:

  * A folder named **“Data for Participants”**.
  * Inside this folder, a **“Submissions”** folder with:

    * documentation on **metrics for cross-validation**,
    * **instructions for submitting results**,
    * **examples of the required data formats** for submissions.

In the Files tab, you will also find **slide templates** that finalist teams must use to prepare their **final presentations**.

---

## 9. Submission Platform and Process

A dedicated **submission platform** is used for:

* uploading your predicted volumes,
* computing metrics,
* and generating the **leaderboard**.

### 9.1 Accessing the Platform

* Log in using your **team’s username and password**.
* The access link is provided in the **submission instructions** document.

**First action after login:**

* **Change your password**:

  * Navigate to the options on the right,
  * Click on **“Profile”**,
  * Select **“Change password”**,
  * Enter the required fields.

### 9.2 Uploading Submissions

To submit your results:

1. Go to the left-hand menu.
2. Click on **“Dashboard/Panel”**.
3. Click on **“Checkpoint”**.
4. Use the **upload button** to submit your file.

If you see an **error message**, it indicates that:

* the **file structure is not correct** and needs to be adjusted
  (e.g., wrong columns, improper formatting, missing fields).

Once a **valid file** is uploaded:

* Your team appears in the **ranking**.
* Each team appears **only once**, showing its **best solution**.
* The ranking is **updated** every time a team uploads a new valid submission, if it improves the score.

### 9.3 Public vs Private Test Set

For leaderboard calculations during the competition:

* The test set is split into:

  * a **public test set** containing **30%** of the test data,
  * a **private test set** containing the remaining **70%**.

* During the Datathon:

  * Only the **public test set** is used to compute scores for the **online leaderboard**.

* Final evaluation:

  * After the competition ends, final results are computed on the **full test set**, including the **private portion**.

### 9.4 Submission Limits and Recommendations

* The number of submissions per team is **limited**:

  * You may submit **up to three times every eight hours**.

* Recommendation:

  * Make a **test submission** within the **first few hours** of the Datathon, to ensure:

    * your understanding of the required format is correct,
    * the platform accepts your file without structural errors.

### 9.5 Final Selection and Timeline

On **Sunday at 9:30 AM**, the **“Select your final option”** feature will be activated on the platform.

* You will then have until **10:30 AM** to:

  * **Select which submission** you want to use for the **final evaluation**.

* At **10:30 AM**, the Datathon **officially ends**:

  * No further changes or submissions will be allowed.
  * The final evaluation will be conducted on the **complete test set**, including the **private portion**.

After all final submissions are processed:

1. The **top 10 results** are published, ordered by their **Scenario 1 score**.
2. Among these, the **top 5 results** are published, ordered by their **Scenario 2 score**.

* These **5 teams** become the **finalists**.

### 9.6 Final Presentations and Code Submission

Only the **5 finalist teams** have additional obligations:

* They must prepare a **presentation** summarizing:

  * the methodology used,
  * their models and algorithms,
  * key data insights and exploratory analysis,
  * their main results and conclusions.

* Deadlines and requirements:

  * You have until **12:00 (noon)** to upload your presentation:

    * to your **private channel** in Microsoft Teams,
    * following the specified **naming convention** (provided in instructions).
  * Finalist teams must also upload the **code** used to generate the results of their final submission.

### 9.7 Final Event and Winners

The final live activities are scheduled as follows:

* **1:00 PM** – Finalist team presentations:

  * Each finalist presents to the jury and audience.

* **2:30 PM** – Jury deliberation and announcement:

  * The jury discusses the results and presentations.
  * The **winning teams** (top 3) are announced.

---

## 10. Participant Expectations and Best Practices

While not strictly enforced as rules, the brief highlights several expectations for how teams should approach the challenge:

* Combine **technical excellence** with **business understanding**.

* Pay particular attention to **high-erosion brands (Bucket 1)**, which:

  * are more heavily weighted in the metrics,
  * are strategically critical from a business point of view.

* Provide:

  * **Thorough data preprocessing** explanations,
  * **Exploratory data analysis** (EDA) focusing on erosion patterns,
  * Use of **visualizations** to:

    * illustrate findings,
    * support interpretations,
    * make results accessible to non-technical stakeholders.

* Be prepared in Phase 2 to **justify modeling choices** and to show how your solution:

  * supports better **forecasting**,
  * informs **planning and strategy** in the post-patent period.

---

## 11. Closing Remarks

The Novartis–Barcelona Digital Finance Hub Datathon brings together **data enthusiasts, innovators, and problem solvers from around the world** to tackle a **real-world, high-impact problem** at the intersection of:

* pharmaceutical business strategy,
* financial planning,
* advanced analytics and forecasting.

Participants are invited to:

* embrace the **thrill, creativity, and innovation** of the event,
* explore the rich datasets provided,
* develop robust and interpretable forecasting models,
* and connect their technical work to meaningful **business insights** on generic erosion.

We wish all teams a **productive, insightful, and enjoyable** Datathon, and we look forward to seeing your **ideas and solutions**.
