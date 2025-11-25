# Novartis Datathon 2025 – Deep‑Dive Guide

> *“Join the Online Novartis Datathon and help us solve a real‑world pharma problem!”*  

---

## 🧭 1. TL;DR

- **What:** Online data science & AI competition on **Novartis financial data**  
- **When:** **27–30 November 2025**  
- **Where:** Fully **online**, coordinated from the **Barcelona Finance Digital Hub**  
- **Who:** Teams of **2–4 data scientists** (no solo teams)  
- **Theme:** Use **big data + predictive models** to help Novartis **allocate resources and invest wisely**  
- **Prizes:**  
  - 🥇 **1st Prize – 4,000 €**  
  - 🥈 **2nd Prize – 2,000 €**  
  - 🥉 **3rd Prize – 1,000 €**  

---

## 📌 2. Event Snapshot

| Item              | Details                                                      |
|-------------------|--------------------------------------------------------------|
| Name              | Novartis Datathon 2025 (8th and final edition)              |
| Dates             | **27–30 November 2025**                                     |
| Format            | 100% **online**                                             |
| Main Focus        | **Financial data**, prediction, resource allocation         |
| Typical Data Type | Tabular / time‑series data from pharma finance context      |
| Team Size         | **2–4 participants** (individual participation not allowed) |
| Registration      | Online application (now marked as **closed**)               |
| Main Website      | https://godatathon.com                                      |
| Language          | English (for communication, docs, and presentations)        |

---

## 🧪 3. Why This Challenge Matters (Business Context)

Novartis states several long‑term strategic priorities, including:

- **Go big on data and digital** – using advanced analytics and AI to support decisions  
- **Embrace operational excellence** – make better, faster, more robust financial decisions  
- **Invest resources wisely** – decide *where* and *when* to allocate money to projects, brands, and countries  

To do this, they need **predictive models** that can answer questions like:

- *How will a portfolio of medicines behave financially in the next months/years?*  
- *Where should budget be increased or decreased to maximize impact?*  
- *How can we reduce the risk of under‑ or over‑investing in a brand or country?*  

The datathon gives you **realistic financial data** and challenges you to build models that support these decisions.

---

## 🧩 4. Challenge Theme & Data

From the official “2025 Challenge” description:

- You’ll receive **a data science challenge on financial data** on **Thursday 27 November 2025**.
- The problem is framed around **prediction and resource allocation** in a pharma‑finance context.
- The **exact dataset and target** are revealed at the kickoff, but based on past editions, you can expect:
  - **Time‑series / panel data**: country, brand, portfolio, time (monthly/quarterly/etc.)
  - **Financial variables**: revenues, costs, investments, maybe indicators of life‑cycle stage
  - **Side variables**: region, product class, portfolio groupings, etc.

Typical tasks might include:

- Predicting **future financial values** for specific units (country‑brand, brand‑cluster, etc.)  
- Using the predictions to **support smarter allocation** of a limited budget  
- Showing **how your model’s outputs could be plugged into real decision‑making**

> [!NOTE]
> You don’t just send predictions; you also need to **tell a story**:  
> how your models help Novartis decide where to put money and why your approach is robust.

---

## 👥 5. Participation & Eligibility

### 5.1 Team Requirements

- **International call:** people from many countries can participate.
- **Team size:** **2–4 people**.  
  - Individual (solo) teams are **not allowed**.
- **Profile:**  
  - Data Scientists or similar, with knowledge of:
    - **Data visualization**
    - **Resource allocation** problems
    - **Prediction / forecasting** problems

> [!TIP]
> Try to form a **complementary team**:
> - 1 strong in **ML/optimization**
> - 1 strong in **EDA & visualization**
> - 1 strong in **business / storytelling**
> - (optional) 1 strong in **MLOps / coding & pipeline design**

### 5.2 Skills & Tech

You can use any tools, but realistic core stack:

- **Python** with:
  - `pandas`, `numpy`, `scikit‑learn`
  - `xgboost`/`lightgbm`/`catboost` (for gradient boosting)
  - `matplotlib` / `seaborn` / `plotly` for visualization
- Or **R**, **Matlab**, or similar if your team prefers.

What matters is not the exact language, but that you can:

1. Clean and understand a business dataset quickly  
2. Train predictive models with **sane validation**  
3. Explain your approach clearly to **non‑ML people**

---

## 🕒 6. Official Timeline & Daily Schedule

### 6.1 Global Timeline

- 📝 **Application period:**  
  - Teams applied online before **20 November 2025, 12:00 CET**.
- ✅ **Selection:**  
  - Selected teams are notified by email in October–November.
- 🚪 **Registration status:**  
  - Currently marked as **“REGISTRATION IS CLOSED”** on the website.

### 6.2 Event Days (All Times CET)

From the official schedule:

#### **Day 1 – 27 November (Kick‑off)**

- **17:00–18:00** – Kick‑off session  
  - Welcome, rules, presentation of business context  
  - Release of **challenge description** and **dataset**

#### **Day 2 – 28 November (Work & Mentoring)**

- **09:00–18:00** –  
  - Team work on the case  
  - Mentoring sessions / Q&A with organizers

#### **Day 3 – 29 November (Work & Mentoring)**

- **09:00–18:00** –  
  - Continued modeling, improvement, validation  
  - More mentoring and refinement of approach and visualizations

#### **Day 4 – 30 November (Final Day)**

- **09:00** – Welcome  
- **09:30** – Final submissions open  
- **10:30** – **Deadline** for final submissions  
- **11:30** – Show preliminary results / rankings  
- **12:00** – Deadline to send **PPT slides** (Top 5 teams)  
- **13:00** – Presentations by **Top 5** teams  
- **14:30** – Jury deliberation  
- **15:00** – **Winners announcement** 🎉  

---

## 🏆 7. Prizes & Recognition

Official prizes:

- 🥇 **First Prize:** **4,000 €**  
- 🥈 **Second Prize:** **2,000 €**  
- 🥉 **Third Prize:** **1,000 €**  

Non‑monetary value:

- Direct exposure to Novartis’ **Digital Finance & AI** leadership  
- A **strong portfolio piece** for your CV, LinkedIn, and academic/professional profile  
- Experience with **real pharma‑finance problems**, which is rare in public competitions  

---

## 📊 8. Evaluation: What the Jury Looks For

The organizers explicitly say:

> “Our jury will look at your **results** as well as **how innovative is your approach**.”

So evaluation usually mixes:

1. **Quantitative performance**  
   - Error metric(s) on a hidden test set (e.g., RMSE/MAE/MAPE/etc.)
   - Stability across different subsets (not overfitting to a corner case)

2. **Technical quality**  
   - Sound validation strategy (time‑aware splits, no leakage)  
   - Consistent handling of missing values, outliers, etc.  
   - Reasonable model complexity vs. data size

3. **Innovation & creativity**  
   - Use of interesting features (e.g., portfolio‑level aggregates, lag features, trend indicators)  
   - Smart ways to connect predictions with **resource allocation decisions**  
   - Interpretable and insightful visualizations

4. **Communication & business storytelling**  
   - Clear explanation in your slides: **what you did, why, and so what**  
   - Ability to answer jury questions (for Top 5) in a **business‑savvy** way  
   - Explicit mapping from *“model output”* to *“finance action”*

> [!IMPORTANT]
> Fancy models without a clear **business story** usually lose to slightly simpler models that are:
> - well‑validated  
> - interpretable  
> - clearly useful for decision‑makers.

---

## 🛠️ 9. Suggested Tech & Workflow

Here’s one practical workflow you can copy for your team.

### 9.1 Project Structure

```text
novartis-datathon-2025/
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ notebooks/
│  ├─ 00_eda.ipynb
│  ├─ 01_feature_engineering.ipynb
│  ├─ 02_modeling_baseline.ipynb
│  ├─ 03_modeling_advanced.ipynb
├─ src/
│  ├─ data_loading.py
│  ├─ features.py
│  ├─ models.py
│  ├─ evaluation.py
├─ results/
│  ├─ submissions/
│  ├─ figures/
└─ README.md
```

### 9.2 Recommended Modeling Steps

1. **Quick EDA (Exploratory Data Analysis)**  
   - Check distributions, missingness, time coverage  
   - Identify key entities: brands, countries, portfolios  

2. **Baseline models first**  
   - Simple mean/naive forecasts, linear models, basic tree‑based methods  
   - Give you a **reference** that is fast and easy to beat

3. **Feature engineering**  
   - Time‑based: lags, moving averages, growth rates  
   - Hierarchical: country‑level and portfolio‑level aggregates  
   - Interaction features if needed

4. **Advanced models**  
   - Gradient boosting methods (`XGBoost`, `LightGBM`, `CatBoost`)  
   - Possibly simple neural networks, if time and data size allow

5. **Validation**  
   - Use **time‑based splits** if series is long enough  
   - Keep a separate “simulation of test” hold‑out set

6. **Explainability & communication**  
   - Feature importance plots  
   - Partial dependence / simple scenario simulations  
   - Portfolio‑level plots showing effect on resource allocation

---

## 🧠 10. Competition Strategy & Tips

> [!TIP]
> **Goal:** In 3.5 days, build something that is **robust, understandable, and business‑useful**.  
> You are not writing a PhD thesis; you’re shipping a convincing prototype.

### 10.1 Team Roles

- **Lead Data Scientist:**  
  - Guides modeling choices and validation strategy.
- **Data Engineer / MLOps‑ish Person:**  
  - Keeps code tidy, reproducible, and version‑controlled.
- **Business / Visualization Lead:**  
  - Focuses on slides, plots, story, and explaining value.
- **Research / Experimentation Support (optional 4th):**  
  - Tries alternative models/ideas and feeds results to the core pipeline.

### 10.2 Time Allocation

Rough suggestion:

- **Day 1 (Evening)**  
  - Understand the **business question** deeply.  
  - Clean data quickly, build **first baseline**.  

- **Day 2**  
  - Intensive EDA and **feature engineering**.  
  - Train **several candidate models**, start tracking results.  

- **Day 3**  
  - Consolidate into **one main pipeline** + 1–2 backup models.  
  - Build visualizations and refine business story.  

- **Day 4 (Morning)**  
  - Polish predictions, **generate final submission**.  
  - Finalize slides, rehearse key talking points.  

---

## ✅ 11. Checklist Before the Deadline

Use this as a last‑minute checklist:

- [ ] We have **at least one baseline** and one improved model  
- [ ] Our **validation scheme** matches how the test is likely evaluated  
- [ ] We checked for **data leakage** (no future info used in training)  
- [ ] We saved **reproducible code / notebooks**  
- [ ] Our **submission file** strictly follows the required format  
- [ ] We prepared **clear plots** (errors by segment, feature importances, etc.)  
- [ ] Slides clearly answer:
  - *What is the business problem?*  
  - *What data did we use and how?*  
  - *What model(s) did we build, and how do they perform?*  
  - *How can Novartis use our predictions to allocate resources?*  

---

## ❓ 12. Quick FAQ

### Q1. Can I join alone?

No. Teams must have **2–4 members**. Individual (solo) participation is not allowed.

### Q2. Is it fully online?

Yes. The event is completely **online**, with sessions and coordination done remotely.

### Q3. What kind of data will we get?

You’ll receive **financial data from Novartis** related to their business context (e.g., country/brand/portfolio‑level figures). Exact details are revealed at the kickoff.

### Q4. Are we allowed to use external libraries?

Yes, as long as you respect the rules in the Terms & Conditions (e.g., no using external private datasets that give an unfair advantage). Standard ML libraries (scikit‑learn, XGBoost, etc.) are fine.

### Q5. How important are slides and presentation?

Very important for the **Top 5** teams. The jury doesn’t only want a good error metric; they want to understand **why your solution is useful** and **how it can be trusted**.

---

## 🔚 13. Final Thoughts

Novartis Datathon 2025 is not just another Kaggle‑style competition. It sits at the intersection of:

- **Hands‑on machine learning**
- **Real‑world pharma finance**
- **Strategic decision‑making**

Treat it as a chance to **simulate being a data science team inside Novartis** for a few days:  
take messy, high‑impact financial questions and push them one step closer to **data‑driven answers**.

Good luck, and may your validation curves be smooth and your PPT animations not crash. 🚀
