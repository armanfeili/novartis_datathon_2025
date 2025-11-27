## A. Competition setup & objectives (1–10)

1. What is the exact objective of this datathon in one sentence, in business terms?
   *(For Novartis 2025: e.g., “Use financial time-series data to forecast key metrics and support smarter resource allocation across brands/countries/portfolios.”)*

2. Is the winner decided purely by a numeric leaderboard metric, or by a **combination** of metric + jury evaluation (presentation quality, novelty, interpretability, business impact), as in typical Novartis / GoDataThon events?

3. What are the **official evaluation criteria and weights** (e.g., main metric vs. innovation, robustness, business story, slide presentation)?
   *(Ask explicitly: “How much does the jury score count vs. the metric?”)*

4. What **deliverables** are required at each stage:

   * Final **prediction file** (mandatory format)?
   * **Slides** (for Top-5 or for all)?
   * Any **code/notebook** to submit or only for inspection upon request?

5. What explicit **constraints** apply:

   * Use of external data or internet?
   * Limits on libraries or frameworks?
   * Max number of submissions?
   * Any compute/runtime constraints (e.g., “must run on a normal laptop, no GPU assumption”)?

6. Given the **4-day schedule** (kickoff Day 1, work Days 2–3, submission Day 4), how much time do we realistically allocate to:

   * Understanding + EDA
   * Modeling + validation
   * Business framing + slides
   * Final polishing + submission?

7. Are there any **baseline models or starter notebooks** provided (e.g., last-value forecast, linear model)? If yes, what performance do they achieve on the official metric, and what margin above baseline is usually needed to be competitive in these datathons?

8. Are there **sample submissions or example winning solutions** from previous Novartis / GoDataThon editions that show:

   * Expected **code quality**
   * Depth of **business narrative**
   * Style of **visualizations**?

9. Are there rules around:

   * Model **explainability** (e.g., avoid fully opaque black-boxes without explanation)?
   * **Fairness / ethics** (especially for healthcare / finance data)?
   * Allowed vs. disallowed use of **generative AI** (e.g., ChatGPT for code vs. external data)?

10. Are there any **bonus points** for extra elements such as:

    * Simple **dashboard or UI mock-up**
    * Clear **decision-support scenario** (e.g., “how a finance manager would use this”)
    * Structured **deployment plan or MLOps sketch**,
      and if so, which of these the jury values most?

---

## B. Problem, stakeholder, and domain (11–20)

11. In plain language, what real-world **finance/operations problem** is the task solving (e.g., “prioritizing investments across brands/countries to maximize financial impact under budget constraints”)?

12. Who is the **primary stakeholder** for our solution in this datathon:

    * Finance director / portfolio manager?
    * Business analyst in digital finance hub?
    * Brand manager at country level?

13. Which concrete **decisions** will our model’s outputs support:

    * Allocate more / less budget to a brand/country/portfolio?
    * Decide which segments are “over-invested” or “under-invested”?
    * Identify high-risk / volatile revenue streams?

14. What is the **current baseline** in real life (without our model):

    * Manual expert judgment and Excel?
    * Rule-based heuristics (e.g., “last year + x%”)?
    * Simple static forecasting?

15. Is the main technical task:

    * **Time series forecasting** (most likely),
    * Regression on tabular panel data,
    * Or a **hybrid** of forecasting + ranking/prioritization?

16. What is the **exact target variable** (e.g., net sales, revenue, investment, margin, ROI), and in domain terms:

    * What does a high vs low value mean?
    * Are there negative or zero values, and what do they correspond to (losses, no sales, etc.)?

17. Are there **domain constraints** (pharma + finance) that affect how predictions can be used:

    * Compliance with internal risk policies?
    * Restrictions on using predictions to drive decisions in certain markets?

18. In real-world terms, what are the **consequences of prediction errors**:

    * Over-investing in low-return segments (wasted budget)?
    * Under-investing in high-potential markets (lost opportunity)?
    * Unstable forecasts leading to planning volatility?

19. Which **domain-specific patterns** should we explicitly check in EDA:

    * Seasonality (monthly/quarterly patterns in sales / investments)?
    * Product **life-cycle stages** (launch, growth, maturity, decline)?
    * Region-specific differences (e.g., emerging vs mature markets)?

20. How would stakeholders (e.g., Novartis finance leadership) define **“success”** in non-technical language:

    * More stable, reliable forecasts?
    * Better prioritization of investment decisions?
    * Increased expected ROI at portfolio level?

---

## C. Metric, risk tolerance, and constraints (21–30)

21. What is the **official evaluation metric** (e.g., RMSE, MAE, MAPE, custom score), and is it applied per overall dataset, per segment, or averaged across entities?

22. Does the metric treat **over- and under-prediction symmetrically** (e.g., squared error) or is there an implicit asymmetry (e.g., if MAPE punishes under-forecasting more when denominators are small)?

23. How well does the **official metric align with real-world cost** of errors in this finance context (e.g., is underestimating worst than overestimating, or vice versa)?

24. Which **secondary metrics** should we monitor internally (e.g., MAE, relative error per segment, stability across time, subgroup performance by region/portfolio)?

25. Is the metric computed on:

    * A single **hold-out test set**,
    * A **hidden leaderboard** (public/private split),
    * Or some averaged cross-validation scheme?

26. Are there any constraints on:

    * Prediction **latency** (probably not critical for a datathon)?
    * Model **size** or memory usage?
    * Training time (e.g., “your solution must be reasonably fast to retrain”)?

27. Are **black-box models** allowed as long as we provide explanations, or do the organizers prefer more **interpretable** models that can be explained to finance stakeholders?

28. Are there **class imbalance / scale issues** in the target (e.g., many small brands vs few huge ones) that could make a single global metric misleading?

29. Are there any **threshold-based business rules** we might want to simulate (even if the datathon doesn’t require it), such as:

    * “Recommend investment increase if predicted growth > X% and margin > Y”?

30. What **minimum improvement** in the official metric over baselines (or over our own previous models) will we consider **practically meaningful** and worth additional complexity?

---

## D. Data source, structure, and semantics (31–40)

31. What is the **origin** of the dataset in this datathon: transactional finance data, aggregated P&L, budgeting data, or internal reporting tables?

32. At what **granularity** is each row:

    * Per (country, brand, month)?
    * Per (portfolio, region, quarter)?
    * A mixture of different aggregation levels?

33. Are there **multiple related tables** (fact + dimensions) or a single flat panel table? If multiple, what is the **primary key** that connects them?

34. Which are the **temporal columns** (date, year, month, quarter), and how are they aligned with the target (e.g., target at t uses features from ≤ t)?

35. What identifiers exist (e.g., country_id, brand_id, portfolio_id) and how should we use them:

    * As grouping keys for time series?
    * As features (via encoding)?
    * Or only as IDs for joining/aggregation?

36. Roughly **how many rows and columns** are there in training data, and is there a separate **test file** (with missing labels) for submission?

37. What are the **data types** of each column: numeric, categorical, datetime, boolean, text, etc., and are there any columns with mixed or inconsistent types?

38. Are there **metadata columns** (indices, file names, constant fields) that should be excluded from modeling?

39. Are there **pre-engineered features** present (e.g., ratios, growth rates, flags) whose definitions we need to understand to avoid double-counting or leakage?

40. Is there a **data dictionary** or description for each column, and if not, can we infer likely meanings from names and distributions with enough confidence to use them?

---

## E. Data quality, missingness, leakage, and sampling bias (41–50)

41. What is the **missing value profile** per column (counts, percentages), and are there columns with extreme missingness (>80%) that require special handling or exclusion?

42. Does missingness appear **systematic** (e.g., certain countries/brands/time periods) and potentially informative (e.g., new launches having more missing data)?

43. Are there **duplicated or near-duplicate rows** (same IDs and timestamps) that should be deduplicated or aggregated?

44. Are there **impossible / inconsistent values** for finance data:

    * Negative revenues where not expected,
    * Suspicious spikes,
    * Inconsistent currency scales?

45. Are apparent **outliers** consistent with real rare events (e.g., product launch spikes, one-off write-downs) or more likely data errors?

46. Could any features **leak future information** relative to the prediction time, such as:

    * Aggregates computed over future periods,
    * Flags or status fields that are only known after the outcome?

47. Are there columns that effectively encode the **target** (e.g., a “final result” or “classification” column derived from it) and should be excluded?

48. How was the sample collected:

    * Full history over a time interval?
    * Selected countries / portfolios?
    * Only certain markets?
      and what **bias** might this introduce compared to the real global portfolio?

49. Does the distribution of the target (and key features) in train match that in test (if known), or are there signs of **covariate / distribution shift** (e.g., different time ranges, new entities)?

50. Does the dataset include **special period effects** (e.g., COVID years, macro shocks) that create non-stationarities we should be aware of?

---

## F. Splitting strategy, validation design, and EDA focus (51–60)

51. Given this is **time-series/panel** data, what is the most realistic way to split data:

    * Pure **time-based split** (train past, validate recent)?
    * Time-based split within each entity (country/brand)?
    * Any need for group splits?

52. Would a **random split** cause leakage (e.g., same (country, brand) appearing in both train and val at overlapping dates), and thus should be avoided?

53. Do we need **stratification** (e.g., by region, portfolio type, or magnitude of target) to ensure validation is representative?

54. Is **TimeSeriesSplit** or a custom rolling window more appropriate than standard K-fold for reliable evaluation?

55. How many **validation splits** or folds can we run given time and compute constraints during the 3.5-day datathon?

56. Which **target distribution statistics** should we compare across train/validation/test (mean, median, variance, quantiles, segment distributions)?

57. Which core **univariate plots** (histograms for numeric features, bar charts for categorical) will give us the fastest understanding of the data landscape?

58. Which **bivariate or temporal plots** are most informative here:

    * Target vs time for key entities
    * Target vs investment
    * Revenue vs country/portfolio?

59. Do we see clear **non-linearities or interactions** (e.g., different behavior by region or life-cycle stage) that suggest we should use tree-based models and/or segmented models?

60. After initial EDA, which **5–10 features** look most promising for forecasting and allocation (lags, growth rates, portfolio aggregates, investments), and which look suspicious or low-value?

---

## G. Feature engineering, preprocessing, and transformations (61–70)

61. How will we handle **missing values** per feature type:

    * Numeric (imputation with median/mean, forward fill in time, etc.)
    * Categorical (special “missing” category?)
    * Time-dependent fields?

62. Which numeric features are **highly skewed** (e.g., revenue, investment) and might benefit from log or other transformations for better model stability?

63. How will we **encode categorical variables** like country, brand, portfolio:

    * One-hot (if low cardinality)
    * Target / frequency encoding
    * Native handling by CatBoost?

64. Which **interactions or ratios** are likely predictive in a finance datathon:

    * Revenue / investment (ROI proxy)
    * Entity’s share of country/portfolio totals
    * Growth rate vs region dummy?

65. For temporal data, which features will we derive:

    * Lags (t-1, t-2, t-3, …)
    * Rolling means / std over windows
    * Time since product launch or first observation
    * Seasonality indicators (month, quarter)?

66. If there is any text (e.g., product descriptions), do we ignore it due to time constraints or use simple encodings (like dummy categories or basic NLP)?

67. For multi-table setups, which **aggregations** per entity (country, brand, portfolio) will we compute: counts, sums, means, volatility measures?

68. Are there features we should **bin or bucket** (e.g., investment level groups, size categories) to simplify patterns and robustness?

69. How will we avoid **target leakage** when engineering aggregates (making sure we only use information up to time t when predicting t+1)?

70. How will we maintain our feature transformations in a **reproducible pipeline** (e.g., sklearn Pipelines, common functions in `src/features.py`) that we can re-run for final submission?

---

## H. Model selection, training, hyperparameters, and experimentation (71–80)

71. What is our **simplest baseline model** for this time-series/finance setting:

    * Naive last value / moving average,
    * Linear regression on basic lags,
    * Very small tree?

72. What is our primary **“hero model” type** for this datathon (likely LightGBM / XGBoost / CatBoost for tabular time-series), and why is it a good fit?

73. Which **hyperparameters** matter most for that hero model in this context (e.g., `learning_rate`, `num_leaves` / `max_depth`, `n_estimators`, regularization parameters, column and row subsampling)?

74. What **tuning strategy** is realistic under time pressure:

    * Manual + small random search,
    * Light grid search with early stopping,
    * No heavy Bayesian optimization?

75. How will we prevent **overfitting during tuning**:

    * Proper time-based validation
    * Early stopping on validation
    * Restricting hyperparameter ranges?

76. How will we **log and compare experiments**: simple table/spreadsheet, naming conventions in notebooks, or minimal MLflow/CSV?

77. What **baseline performance** (metric value) do we aim to beat, and what **target score** would feel competitive given past datathon leaderboards?

78. Will we try more than one **model family** (e.g., gradient boosting + linear model, possibly a simple deep learning baseline), and in what priority order?

79. When do we decide to **stop adding new models** and instead invest our remaining time in:

    * Robustness checks
    * Error analysis
    * Business story and slides?

80. How will we ensure the chosen final model is **robust across seeds and folds**, and not just a lucky split?

---

## I. Ensembles, calibration, robustness, fairness, and ethics (81–90)

81. Does combining predictions from **different models** (e.g., averaging two strong GBMs or GBM + linear) significantly improve validation performance and stability?

82. If we build an **ensemble**, how do we keep it:

    * Simple to explain in slides,
    * Easy to reproduce at submission time?

83. Is **calibrated uncertainty** or probabilistic output relevant (e.g., for risk assessment), and if so, can we use simple methods (Platt scaling, isotonic regression, quantile models) within our time budget?

84. How **sensitive** is our model to:

    * Random seed
    * Train/validation split choice
    * Slight perturbations in data (e.g., noise, outlier removal)?

85. How will we perform **targeted error analysis**:

    * Examine worst-predicted entities (country/brand)
    * Look at time periods with biggest errors (launches, crises)?

86. Which features does the model mark as **most important**, and are those plausible given finance/pharma intuition (e.g., recent revenue, investment, market, life-cycle stage)?

87. Does performance differ substantially across **subgroups** (regions, portfolios, maturity stages), indicating any potential bias or weak spots?

88. Are there any **sensitive attributes or proxies** (e.g., region categories that correlate with regulation or socio-economic factors) that require careful interpretation in recommendations?

89. What **disclaimers or caveats** should we include so stakeholders do not over-trust the model (e.g., data coverage limits, not a causal model, not a guarantee of ROI)?

90. In which scenarios would we explicitly recommend **human override / review** of model suggestions (e.g., new launches, markets with very short history, extreme macro events)?

---

## J. Implementation, reproducibility, presentation, and future work (91–100)

91. Can someone **reproduce our entire pipeline** (from raw data to final predictions) with a small number of clear steps or a single script/notebook?

92. Is our **project structure** clean and intuitive (separate folders for data, notebooks, `src/`, results, slides), so a judge or mentor can navigate it quickly?

93. Do we have a short **README** that explains environment, dependencies, and how to run training and generate the final submission?

94. Are key **configuration values** (paths, seeds, features, model hyperparameters) centralized in one place, not scattered across notebooks?

95. Have we cleaned final notebooks/scripts of **dead code, debugging prints, and unused experiments**, leaving only a clear story of the final solution?

96. Can we explain our solution in **one minute** to a non-technical judge:

    * Problem → Data → Model → Value?

97. Do our **plots and tables** in the presentation clearly communicate:

    * EDA insights
    * Model performance
    * Resource allocation recommendations
      with readable labels and minimal clutter?

98. Have we clearly summarized:

    * Baseline approach and score
    * Best model and score
    * **Incremental improvement** and why it matters for finance decisions?

99. Have we explicitly listed the main **limitations** of our approach (data, model, assumptions) and at least 2–3 realistic **next steps** to improve it for a production setting?

100. If we had **1–2 extra days** after the datathon, what would be our top three priorities to turn this into a **deployment-ready, high-impact tool** (e.g., better uncertainty modeling, simple dashboard, integration with budgeting workflows)?

