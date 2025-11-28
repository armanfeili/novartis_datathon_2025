### A. Competition setup & objectives (1–10)

1. What is the exact objective of this datathon in one sentence?
2. Is the winner decided purely by a numeric metric (leaderboard), or also by qualitative criteria such as presentation, novelty, interpretability, and impact?
3. What are the official evaluation criteria and their relative weights (e.g., metric 60%, presentation 20%, innovation 20%)?
4. What deliverables are required (e.g., notebook/script, slides, written report, live demo, GitHub repo)?
5. Are there any explicit constraints (no external data, no internet, only one submission per team, limited compute, specific runtime limits)?
6. What is the exact timeline of the datathon, and how much time do I realistically have for each phase (understanding, EDA, modeling, evaluation, presentation)?
7. Are there baseline models or starter notebooks provided, and how do my results need to improve over them to be competitive?
8. Are there any sample submissions or example solutions from previous editions that indicate what “good” looks like for this organizer?
9. Are there any rules regarding model explainability, fairness, or use of specific libraries/frameworks?
10. Are there any bonus points for extra elements (dashboard, simple UI, interpretability report, deployment sketch), and if so, which are most valuable?

---

### B. Problem, stakeholder, and domain (11–20)

11. In plain language, what real-world problem is this task trying to solve?
12. Who is the primary stakeholder or end user (e.g., business analyst, doctor, policy maker, operations manager, end-customer)?
13. What decision will this model or analysis directly support for that stakeholder?
14. What would be the current “baseline” way of solving this problem without my model (e.g., manual rules, human expert judgment, no solution)?
15. Is the task classification, regression, ranking, time series forecasting, clustering, recommendation, or something else?
16. What is the target variable, and what does each possible value of the target mean in domain terms?
17. Are there any domain-specific constraints or regulations (e.g., in healthcare, finance, education) that affect how predictions may be used?
18. What are the most severe real-world consequences of model errors in this context (both false positives and false negatives)?
19. Are there obvious domain-specific patterns or hypotheses I should test during EDA (e.g., seasonality, age effects, regional differences)?
20. How would the stakeholder define “success” in non-technical terms (e.g., fewer cancellations, better risk stratification, higher revenue)?

---

### C. Metric, risk tolerance, and constraints (21–30)

21. What is the official evaluation metric (e.g., AUC, F1, accuracy, RMSE, MAE, log-loss, custom score)?
22. Does this metric treat false positives and false negatives symmetrically, or is it implicitly asymmetric?
23. If the metric is not directly aligned with real-world cost (e.g., AUC vs. actual misclassification cost), how big is that mismatch?
24. Are there secondary metrics I should track internally (e.g., precision/recall per class, calibration error, subgroup performance)?
25. Is the metric computed on a single test set, a cross-validation scheme, or a leaderboard split that may differ from my local validation?
26. Are there any constraints on prediction latency, model size, memory, or training time that might restrict model choice?
27. Are there constraints on using black-box models, or is full interpretability required?
28. Are there class imbalance issues that might make some metrics (e.g., accuracy) misleading?
29. Are there any threshold-based business rules (e.g., flag if probability > 0.8) that I should consider when evaluating performance?
30. How will I decide when performance improvements are practically meaningful and not just marginal noise (e.g., minimum delta in metric)?

---

### D. Data source, structure, and semantics (31–40)

31. What is the origin of the dataset (system logs, surveys, sensors, transactional data, clinical data, etc.)?
32. At what granularity is each row (per user, per transaction, per session, per time step, per product, etc.)?
33. Are there multiple related tables (relational schema) or just a single flat table?
34. Are there any temporal columns (timestamps, dates) and how do they relate to the target event?
35. Are there identifiers (user_id, customer_id, patient_id, device_id, etc.) and how are they supposed to be used or not used as features?
36. How many rows and columns are in the training data, and is there a separate test set provided?
37. What are the data types of each column (numeric, categorical, text, boolean, datetime, geospatial, image paths, etc.)?
38. Are there columns that are clearly metadata or artifacts (e.g., index, file name, constant columns) that should be excluded from modeling?
39. Are there any derived or engineered-looking columns already present (e.g., scores, ratios, flags), and do I understand how they were computed?
40. Is there any documentation or data dictionary explaining each column’s meaning in domain terms, and if not, can I infer it reliably?

---

### E. Data quality, missingness, leakage, and sampling bias (41–50)

41. What is the distribution of missing values per column, and are there columns with extreme missingness (e.g., >80%)?
42. Is missingness random, or does it correlate with the target or key features (which might carry signal)?
43. Are there duplicated rows or near-duplicates, and should they be removed or aggregated?
44. Are there impossible or inconsistent values (e.g., negative ages, future timestamps, inconsistent units) that need correction or removal?
45. Are there outliers, and do they look like genuine rare events or data errors?
46. Could any features leak future information relative to the prediction time (e.g., features recorded after the outcome, labels embedded in text, post-event aggregates)?
47. Are there any columns that directly or indirectly encode the target (e.g., outcome_description, status_code that is derived from the label)?
48. How was the sample collected (random sample, specific segments, only certain time windows), and could this introduce sampling bias?
49. Does the class distribution in the dataset match realistic real-world prevalences, or has it been artificially balanced?
50. Are there potential covariate shift or distribution shift issues between train and test (e.g., different time periods, regions, or populations)?

---

### F. Splitting strategy, validation design, and EDA focus (51–60)

51. What is the most realistic way to split data into train/validation/test given the real-world scenario (random, time-based, group-based)?
52. Does a naive random split risk putting related entities (e.g., same user or patient) into both train and validation, causing leakage?
53. Should I use stratified splits to preserve class proportions, and on which variable should stratification be done?
54. Is GroupKFold or TimeSeriesSplit more appropriate than simple K-fold due to grouping or temporal dependencies?
55. How many folds or validation splits can I realistically run with the available time and compute?
56. What target distribution statistics (class balance, mean, variance) should I compute and monitor in train vs validation vs test?
57. Which core univariate plots (histograms, KDEs, bar charts) should I generate to quickly understand each feature?
58. Which bivariate plots (feature vs target, feature vs feature) are most likely to reveal important patterns or interactions?
59. Are there clear indications of non-linearity or interactions that should influence model choice or feature engineering?
60. Based on EDA, which 5–10 features look most promising, and which look suspicious or likely useless?

---

### G. Feature engineering, preprocessing, and transformations (61–70)

61. How will I handle missing values for each type of feature (numeric, categorical, text, datetime)?
62. Which features might benefit from transformations (log, sqrt, standardization, normalization) due to skewness or scale differences?
63. How will I encode categorical variables (one-hot encoding, target encoding, ordinal encoding, CatBoost native encoding, embeddings)?
64. Are there interactions or combinations of features (ratios, products, differences) that are likely to be predictive?
65. For temporal data, what features can I derive (time since last event, rolling averages, trends, seasonality indicators)?
66. For text data, how will I represent it (bag-of-words, TF-IDF, pretrained embeddings, sentence embeddings, custom tokenization)?
67. For any multi-table relations, which aggregations (count, mean, max, min, std) per entity should I compute and use as features?
68. Are there features that should be bucketed or binned (age groups, value ranges) to capture non-linear effects or improve robustness?
69. How will I avoid target leakage when engineering features (especially aggregated statistics that might use future information)?
70. How will I keep track of all feature transformations in a reproducible way (pipelines, transformers, well-documented code)?

---

### H. Model selection, training, hyperparameters, and experimentation (71–80)

71. What is my simplest baseline model (e.g., constant predictor, logistic regression, linear regression, basic decision tree)?
72. What is my “hero” model type for this dataset (e.g., LightGBM, XGBoost, CatBoost, random forest, neural network), and why?
73. Which hyperparameters will I treat as most important to tune for the hero model (e.g., learning rate, max_depth, n_estimators, regularization terms)?
74. How will I design a tuning strategy that balances performance and time (manual search, grid search, random search, Bayesian optimization)?
75. How will I prevent overfitting during tuning (proper validation splits, early stopping, limiting hyperparameter ranges)?
76. How will I record and compare experiments (simple table, text log, spreadsheet, MLflow, clear naming conventions)?
77. What baseline performance do I aim to surpass, and what target metric value would I consider strong or competitive?
78. Will I try multiple model families (e.g., tree-based and linear models, maybe one NN), and in what order of priority?
79. How will I decide when to stop adding new models and focus on polishing the best one?
80. How will I ensure that the final chosen model is not just the one with the single best lucky run, but is robust across seeds and folds?

---

### I. Ensembles, calibration, robustness, fairness, and ethics (81–90)

81. Does combining predictions from different models (simple averaging, weighted averaging, stacking) significantly improve validation performance?
82. If I build an ensemble, how will I keep it simple enough to explain and reproduce under time pressure?
83. Is probability calibration important for this use case, and if so, will I apply methods like Platt scaling or isotonic regression?
84. How sensitive is my model to random seed, train/validation split, and data perturbations (i.e., does performance vary wildly)?
85. How will I perform targeted error analysis (e.g., inspect worst-predicted cases, confusion matrix, per-class metrics)?
86. Which features are most important according to the model, and are these importances domain-plausible or suspicious?
87. Does model performance differ substantially across subgroups (e.g., age ranges, regions, genders, categories), indicating potential bias?
88. Are there sensitive attributes (or proxies) that I should exclude or treat carefully due to fairness or ethical concerns?
89. What disclaimers or usage caveats are necessary so that stakeholders do not misuse or overtrust the model’s predictions?
90. Are there obvious failure modes or scenarios where I would recommend against using the model without human oversight?

---

### J. Implementation, reproducibility, presentation, and future work (91–100)

91. Can someone reproduce my full pipeline (from raw data to predictions) with a single clear command or minimal steps?
92. Do I have a clean project structure (separate data, notebooks, scripts, models, reports) that a judge can navigate easily?
93. Is there a README (or equivalent) that explains setup, dependencies, and how to run training/evaluation/prediction?
94. Are paths, seeds, and configuration values centralized (e.g., in a config file) instead of scattered and hard-coded?
95. Have I removed unnecessary debug code, redundant cells, and experimental clutter from the final notebooks or scripts?
96. Can I explain my solution in one minute to a non-technical judge, highlighting problem, approach, and result?
97. Do my plots and tables in the presentation clearly communicate the key insights (readable labels, no overload, aligned with the story)?
98. Have I clearly summarized: the baseline, my best model, the improvement achieved, and why that improvement matters in practice?
99. Have I explicitly listed the main limitations of my approach and at least 2–3 meaningful ideas for future improvements?
100. If I had 1–2 more days to work on this after the datathon, what would be my top three priorities to make this solution production-ready and even more impactful?

---
