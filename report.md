---
title: "Predicting Life Expectancy from Global Health, Economic, and Social Indicators"
---

# Cover

- **Project title:** Predicting Life Expectancy from Global Health, Economic, and Social Indicators
- **Course, section, semester:** CSE437, Data Science — [fill in your section] — [fill in your semester/year]
- **Group members:** [Full Name — Student ID], [Full Name — Student ID], [Full Name — Student ID]
- **GitHub repository link:** [paste your repo URL here]
- **Date:** September 2026

---

# Summary

*(150–200 words — written last, after everything else, per the template's instruction.)*

We predict a country's life expectancy in a given year using the WHO Life Expectancy dataset (2,938
rows, 22 columns, 193 countries, 2000–2015), a real and heavily messy dataset with structural missing
data, inconsistent scales, and genuine outlier country-years. After a three-pass imputation strategy,
log-transformation of skewed features, and a three-method feature selection process, we trained and
tuned two model families — Ridge Regression (linear) and Random Forest / Gradient Boosting (tree
ensembles) — validated with cross-validation and evaluated once on a held-out test set. Random Forest was
the best model (RMSE 1.84 years, R² 0.962), far ahead of both the mean-prediction baseline (RMSE 9.45)
and Ridge (RMSE 3.92), confirming the relationships in this data are meaningfully non-linear. The single
most important finding: **HIV/AIDS prevalence is the dominant predictor of life expectancy for
Developing countries (60% of Random Forest importance) but is statistically irrelevant for Developed
countries** — a sharp asymmetry that neither a single global model nor a simple correlation table alone
would reveal.

---

# 1. Problem and Dataset

## 1.1 Problem Statement

We predict a country's life expectancy in a given year from health, economic, and social indicators
reported for that country-year. This matters because life expectancy is a widely used summary measure
of population health and development; understanding which factors most strongly associate with it — and
whether those factors differ between wealthier and poorer nations — has direct relevance to how health
and development resources get prioritized. This is a **regression** problem: the model outputs a
continuous number of years, not a category.

## 1.2 Dataset

**Source:** WHO Life Expectancy dataset, originally compiled from the World Health Organization's Global
Health Observatory (GHO) and United Nations data repositories. Working link (Kaggle mirror):
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

**Collection method:** compiled by the original author from WHO's GHO data repository and UN population
records; each row is one country's reported statistics for one calendar year.

**Size:** 2,938 rows × 22 columns as downloaded.

**Time period covered:** 2000–2015 (16 years), 193 countries.

**License / terms of use:** distributed on Kaggle under Kaggle's standard open dataset terms (CC0 /
public-domain-equivalent per the dataset's Kaggle listing); underlying figures originate from WHO/UN
public statistics.

We did not scrape or merge additional sources — this is the dataset as downloaded, used directly.

## 1.3 Target Variable

**Name:** `life expectancy` (renamed `life_expectancy` in code for consistency).
**Type:** continuous (regression target), measured in years.
**Distribution:** ranges from 36.3 to 89.0 years, mean ≈ 69.2, left-skewed with a long low tail (Figure 1).
Class balance is not applicable since this is a regression target, not a classification label.

![Distribution of the target variable, and by development status](../figures/02_target_distribution.png)

## 1.4 Three Questions

*(From our approved proposal, unchanged. Answered directly, with evidence, in Section 7.4.)*

1. How much does healthcare expenditure (as % GDP) actually correlate with life expectancy vs. schooling
   years?
2. Do developing vs. developed status countries show different key predictors?
3. Which single factor (immunization rates, BMI, HIV/AIDS prevalence) has the largest negative effect?

---

# 2. Data Handling and Preprocessing

## 2.1 Data Quality Audit

- **Missing values per column** (raw file, top offenders): `population` 22.2% (652 rows), `hepatitis_b`
  18.8% (553), `gdp` 15.2% (448), `total_expenditure` 7.7% (226), `alcohol` 6.6% (194),
  `income_composition` 5.7% (167), `schooling` 5.6% (163), `thinness_10_19_years` / `thinness_5_9_years`
  / `bmi` 1.16% each (34), `diphtheria` / `polio` 0.65% each (19), `life_expectancy` / `adult_mortality`
  0.34% each (10). **2,563 missing cells in total** across the raw file (Figure 2).
- **Duplicate rows:** none found (0 exact duplicates, 0 duplicate country-year pairs).
- **Inconsistent categories:** none in the categorical column (`status` has exactly two clean values,
  Developed/Developing) — the inconsistency in this dataset is in column *naming*, not category values:
  raw column headers have irregular spacing (e.g. `"Life expectancy "`, `" BMI "`), standardized in code
  via an explicit rename map before any analysis.
- **Impossible values:** we found one genuine impossible-value case during error analysis (Section 7.3)
  — `income_composition = 0.000` for Saint Vincent and the Grenadines in 2000, which breaks a smooth,
  otherwise-continuous trend for that country (0.673 the very next year) and is almost certainly a
  placeholder rather than a true measurement. Because it is a valid float (`0.0`), not a null, our
  missing-value pipeline does not catch it automatically — this is flagged explicitly as a limitation in
  Section 8.

![Missing values by column](../figures/01_missing_values.png)

## 2.2 Missing Values

**Assumed mechanism:** missingness is **not** random (MCAR) — it clusters by country, consistent with
some countries simply not reporting certain indicators consistently across years (Missing At Random
conditional on country, MAR). Treating it as MCAR and using a single global mean/median would blur real
cross-country differences.

**Strategy per column (applied uniformly to every numeric column with missing values):**
1. Within-country linear interpolation across years.
2. Within-country forward/backward fill, for edge years interpolation can't reach.
3. Fallback to the country's development-status peer-group median, only for the rare case a country is
   missing a value in every single year on record.

**Dropping:** we dropped the 10 rows missing the **target variable** itself (`life_expectancy`) —
justified because there is no principled way to impute the value we are trying to predict, and it is
under 0.4% of rows.

## 2.3 Outliers

**Detection method:** boxplots across all numeric columns (Figure 3) plus manual inspection of the
lowest life-expectancy rows.

**Flagged:** 19 rows with life expectancy below 45 years.

**What we did with them:** we did **not** remove or cap them. We cross-checked each against real-world
history: Haiti (2010, the earthquake year) and Malawi/Sierra Leone/Zambia/Zimbabwe (early 2000s, the
peak of the southern-African HIV/AIDS epidemic). These are genuine extreme events, not data-entry
errors, so removing them would have deleted real signal the model should learn from.

![Outlier boxplots across all numeric features](../figures/03_outlier_boxplots.png)

## 2.4 Transformation and Scaling

**Encoding:** `status` (Developed/Developing) binary-encoded to `status_developed` (0/1).

**Log transform:** six heavily right-skewed count/monetary columns (`measles`, `infant_deaths`,
`under_five_deaths`, `population`, `gdp`, `percentage_expenditure`) were transformed with `log1p`
(Figure 4), chosen over deletion or capping since it preserves all data points while normalizing scale.

**Scaler choice:** `StandardScaler` (zero mean, unit variance), appropriate for Ridge Regression, which
is scale-sensitive; the tree-based models (Random Forest, Gradient Boosting) do not require scaling but
are unaffected by it when it's present in a shared pipeline.

**Where scaling happens relative to the split — leakage guard, stated explicitly:** scaling is **not**
fit on the full dataset before splitting. It is fit only on each training fold, inside an
`sklearn.Pipeline`, and re-fit independently for every cross-validation fold and for the final held-out
test evaluation. This guarantees no statistic computed from validation or test data ever influences how
training data is scaled. This is implemented in `notebooks/04_modeling_and_tuning.ipynb`.

![Effect of StandardScaler on three example features](../figures/07_scaling_demo.png)

## 2.5 Before and After

| Stage | Rows | Columns | Missing cells |
|---|---|---|---|
| Raw file (as downloaded, columns renamed only) | 2,938 | 22 | 2,563 |
| After preprocessing (target-missing dropped, imputed, `status` encoded) | 2,928 | 22 | 0 |
| After feature engineering (redundant + weak features dropped) | 2,928 | 19* | 0 |

*19 columns = 17 modeling features + `life_expectancy` target + `country` ID column (excluded from
modeling, kept only for error analysis in Section 7.3).

---

# 3. Statistical Analysis

## 3.1 Descriptive Statistics

Rather than a full raw `describe()` dump, the statistics that actually informed our modeling decisions:

- **Target (`life_expectancy`):** mean 69.2 years, median 72.1 years, std 9.5 — the gap between mean and
  median confirms the left skew visible in Figure 1.
- **`adult_mortality`:** mean 164.4, but with a very long right tail (max 723) — a handful of countries
  with extremely high adult mortality drive most of the spread, foreshadowing the outlier discussion in
  Section 2.3.
- **`gdp` and `population`:** both span multiple orders of magnitude (GDP per capita from double digits
  to over $100,000; population from a few thousand to over a billion), which is exactly why we
  log-transformed them (Section 2.4) rather than leaving them on a raw linear scale.
- **`status` (categorical):** 2,416 Developing rows vs. 512 Developed rows (82% / 18%) — an important
  imbalance we return to in Section 7.4 (Q2), since the Developed subgroup's statistics are inherently
  noisier with roughly a quarter the sample size.

## 3.2 Relationships

![Correlation matrix, all numeric features](../figures/04_correlation_heatmap.png)

`adult_mortality`, `income_composition`, and `schooling` show the strongest linear correlation with the
target among raw features. `hiv_aids` and `adult_mortality` are also correlated with each other — an
early signal of multicollinearity that we account for during feature selection (Section 4.3).

![Random Forest feature importance](../figures/08_feature_importance.png)

## 3.3 What the Data Says So Far

- Missingness is structural (clusters by country), not random — informs the within-country imputation
  strategy used in Section 2.2.
- Several features span orders of magnitude and need transformation before they're usable in a linear
  model.
- The lowest life-expectancy rows correspond to real historical crises, not noise — they should be kept.
- `adult_mortality`, `income_composition`, `schooling`, and `hiv_aids` are consistently the strongest raw
  correlates of the target and are likely to dominate feature selection.
- The Developed/Developing split is imbalanced (82%/18%), so any subgroup analysis on Developed countries
  alone should be read with that smaller sample size in mind.

---

# 4. Feature Engineering

## 4.1 Derived Features

We did not construct new engineered features from combinations of raw columns (e.g. ratios or
interaction terms) — the raw indicators were judged sufficiently informative on their own, and adding
synthetic combinations risked reducing interpretability without a clear justification for any specific
combination. Our feature engineering effort instead focused on **removing** redundancy and **selecting**
the most useful subset of the existing columns (Sections 4.2–4.4).

## 4.2 Dimensionality Reduction

**Method applied:** Principal Component Analysis (PCA) on the standardized, feature-selected set (17
features).

**Components retained / variance explained:** 12 of 17 components were needed to reach 90% cumulative
explained variance (14 for 95%) (Figure 5).

![PCA cumulative explained variance](../figures/09_pca_variance.png)

**Kept in the final pipeline?** **No.** With only a 17→12 compression at 90% variance, PCA's
dimensionality-reduction benefit here is small, while its cost is real: PCA components are linear
combinations of standardized features and are not interpretable, which conflicts directly with Research
Question 1/3 (identifying *which* factors matter, and in which direction). The final modeling pipeline
uses the 17 selected original features, not PCA components.

## 4.3 Feature Selection

**Method:** three independent scoring methods computed for every candidate feature — Pearson correlation
(linear relationships), mutual information (non-linear relationships), and Random Forest feature
importance (interaction effects). A feature was dropped only if it ranked in the **bottom quartile on
all three methods simultaneously**, to avoid any single method's blind spot driving the decision alone.

**Ranking / threshold:** of the (post-redundancy-removal) candidate features, only `measles` fell in the
bottom quartile across all three rankings and was dropped on this basis.

## 4.4 Final Feature Set

**Kept (17 features):** `year`, `adult_mortality`, `alcohol`, `percentage_expenditure`, `hepatitis_b`,
`bmi`, `under_five_deaths`, `polio`, `total_expenditure`, `diphtheria`, `hiv_aids`, `gdp`, `population`,
`thinness_10_19_years`, `income_composition`, `schooling`, `status_developed`.

**Dropped, and why:**
- `infant_deaths` (r = 0.996 with `under_five_deaths`, which structurally includes it — redundant).
- `thinness_5_9_years` (r = 0.939 with `thinness_10_19_years` — same malnutrition signal, redundant).
- `measles` (bottom quartile on correlation, mutual information, *and* Random Forest importance).
- `country` (193 categories — too high-cardinality to one-hot encode sensibly, and target-encoding it
  risks leaking country-level average life expectancy directly into the model; kept only as a
  non-feature ID column for error analysis).

**Justification for the final set as a whole:** the 17 remaining features span every conceptual category
in the original data — mortality (`adult_mortality`, `under_five_deaths`), disease burden (`hiv_aids`,
immunization rates), economic (`gdp`, `percentage_expenditure`, `total_expenditure`,
`income_composition`), social (`schooling`, `alcohol`, `bmi`, `thinness_10_19_years`), and structural
(`year`, `status_developed`) — while removing only features that were either redundant with another kept
feature or demonstrably uninformative across three independent tests.

---

# 5. Modeling and Validation

## 5.1 Validation Strategy

**Split:** 80% train / 20% held-out test, `random_state=42`, no stratification (regression target, and
the split is by row/country-year, not by country, so a given country can appear in both train and test
in different years — see Section 8 for the limitation this introduces).

**Cross-validation:** 5-fold CV (Ridge) or 3-fold CV (Random Forest, Gradient Boosting — fewer folds
used to keep search runtime reasonable on limited compute) within the training set only, used as the
validation layer for hyperparameter selection. The test set is touched exactly once, at the very end
(Section 7.1), never during model selection or tuning.

**Temporal/grouped structure:** the data is a country-year panel; we did not treat this as a time-series
forecasting problem (we are not predicting *future* years from *past* years) so a chronological split was
not required, but note in Section 8 that a random split can put the same country's earlier and later
years on both sides of the split, which is a soft form of leakage worth flagging honestly.

## 5.2 Baseline

**Trivial predictor:** predict the training-set mean life expectancy (69.10 years) for every row,
regardless of input features (`sklearn.dummy.DummyRegressor(strategy='mean')`).

**Baseline score on the held-out test set:** RMSE = 9.45 years, MAE = 7.77 years, R² = −0.004 (R² near
zero by construction, since predicting the mean minimizes squared error only on the training set it was
fit on).

## 5.3 Model Families

- **Ridge Regression** (linear family, L2-regularized). Suits this problem as an interpretable,
  fast-to-fit reference point; assumes a linear relationship between each feature and the target, and
  that regularization can control overfitting from the moderate multicollinearity noted in Section 3.2.
- **Random Forest** (bagging ensemble of decision trees). Suits this problem because it makes no
  linearity assumption and handles feature interactions automatically; assumes enough training data per
  region of feature space to average over many trees (a concern for the smaller Developed subgroup,
  Section 7.4).
- **Gradient Boosting** (boosting ensemble of decision trees). A second, independently-trained
  non-linear family, included to check whether Random Forest's advantage over Ridge is a general
  property of tree-based methods on this data, or specific to bagging.

## 5.4 Metrics

**Primary metric, named before results were seen:** RMSE (root mean squared error, in years), because it
penalizes large individual errors more heavily than MAE, and large errors (e.g. badly misjudging a
crisis-affected country) are the more costly kind of mistake for this problem. We also report MAE (a
more directly interpretable "typical error in years") and R² (proportion of variance explained) as
secondary metrics, since a single metric taken alone can obscure whether errors are evenly spread or
dominated by a few large misses.

---

# 6. Hyperparameter Tuning

## 6.1 Search Space

| Model | Hyperparameter | Range / grid |
|---|---|---|
| Ridge | `alpha` | {0.01, 0.1, 1, 5, 10, 50, 100} |
| Random Forest | `n_estimators` | {100, 200, 300} |
| Random Forest | `max_depth` | {None, 8, 15} |
| Random Forest | `min_samples_split` | {2, 5, 10} |
| Random Forest | `min_samples_leaf` | {1, 2, 4} |
| Random Forest | `max_features` | {sqrt, log2, None} |
| Gradient Boosting | `n_estimators` | {100, 150, 200} |
| Gradient Boosting | `learning_rate` | {0.01, 0.05, 0.1, 0.2} |
| Gradient Boosting | `max_depth` | {2, 3, 4} |
| Gradient Boosting | `subsample` | {0.7, 0.85, 1.0} |

## 6.2 Method

- **Ridge:** exhaustive grid search, 7 candidates, 5-fold CV (35 fits), scored on negative RMSE.
- **Random Forest / Gradient Boosting:** randomized search, **12 candidates each**, 3-fold CV (36 fits
  each), scored on negative RMSE. Randomized rather than exhaustive search was used for the two ensemble
  models because their full grids contain up to 405 combinations, and this project was run in a
  single-core environment where exhaustive search was not practical; random search covers the space
  broadly at a fraction of the cost.

## 6.3 Results

**Ridge — full trend across all 7 candidates (5-fold CV RMSE):**

| alpha | mean CV RMSE | rank |
|---|---|---|
| **50** | **4.0227** | **1 (best)** |
| 10 | 4.0230 | 2 |
| 5 | 4.0231 | 3 |
| 1 | 4.0233 | 4 |
| 0.1 | 4.0233 | 5 |
| 0.01 | 4.0233 | 6 |
| 100 | 4.0243 | 7 |

Note how flat this trend is — regularization strength barely moves Ridge's score at all, which is itself
evidence that Ridge's ceiling is set by its linear form, not its tuning (consistent with the large gap
to the tree ensembles below).

**Random Forest — top 5 of 12 candidates (3-fold CV RMSE):**

| n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features | mean CV RMSE | rank |
|---|---|---|---|---|---|---|
| **200** | **15** | **10** | **2** | **None** | **2.1075** | **1 (best)** |
| 200 | 15 | 5 | 4 | None | 2.1091 | 2 |
| 100 | None | 2 | 2 | sqrt | 2.1234 | 3 |
| 200 | 15 | 5 | 2 | sqrt | 2.1330 | 4 |
| 300 | 8 | 10 | 2 | None | 2.1959 | 5 |

**Gradient Boosting — top 5 of 12 candidates (3-fold CV RMSE):**

| n_estimators | learning_rate | max_depth | subsample | mean CV RMSE | rank |
|---|---|---|---|---|---|
| **200** | **0.10** | **4** | **0.85** | **2.0142** | **1 (best)** |
| 150 | 0.10 | 4 | 1.00 | 2.0656 | 2 |
| 150 | 0.20 | 3 | 1.00 | 2.1576 | 3 |
| 150 | 0.10 | 3 | 0.85 | 2.2044 | 4 |
| 100 | 0.05 | 4 | 0.70 | 2.2133 | 5 |

Unlike Ridge's flat trend, both ensemble methods show a clear, meaningful spread across candidates (best
to worst CV RMSE differs by more than a full year for Gradient Boosting, e.g. 2.01 vs. its worst
candidates near 4.7 for `learning_rate=0.01`) — confirming the search space and tuning budget were doing
real work for these models, not just marginal refinement.

---

# 7. Results, Visualization and Error Analysis

## 7.1 Test Set Performance

Reported once, on the held-out test set, after all tuning was finalized:

| Model | RMSE (years) | MAE (years) | R² |
|---|---|---|---|
| Baseline (predict mean) | 9.45 | 7.77 | −0.004 |
| Ridge Regression | 3.92 | 2.97 | 0.827 |
| Gradient Boosting | 1.87 | 1.26 | 0.961 |
| **Random Forest (best)** | **1.84** | **1.15** | **0.962** |

![Metric comparison across models](../figures/10_metric_comparison.png)

All three trained models comfortably beat the baseline. Random Forest is the best model overall, though
its margin over Gradient Boosting (RMSE 1.84 vs 1.87) is small compared to the much larger gap to Ridge
(3.92) — the real story is "linear vs. non-linear," not "which specific tree ensemble."

## 7.2 Visualization

![Actual vs. predicted, all three models](../figures/11_actual_vs_predicted.png)

Ridge visibly bows away from the diagonal below ~55 years of life expectancy, systematically
overpredicting for the worst-off countries. Both tree ensembles track the diagonal closely across the
full range.

![Residual plots, all three models](../figures/12_residuals.png)

Ridge's residuals show a curved, funnel-shaped pattern — a sign of a misspecified (too-simple) model —
while Random Forest and Gradient Boosting residuals scatter much closer to a random band around zero.

![Random Forest feature importance (repeated from Fig. 6 for reference)](../figures/08_feature_importance.png)

## 7.3 Error Analysis

Random Forest's 15 largest test-set errors were examined individually rather than only in aggregate.
Two concrete, explainable examples:

**Example 1 — Saint Vincent and the Grenadines, 2000.** True life expectancy 79.0 years; predicted 65.4
(error 13.6 years, the single largest miss). This is hard because the raw source data itself contains a
likely artifact: `income_composition = 0.000` for this exact row, while the very next year (2001) jumps
to 0.673 with no other indicator changing sharply — a near-impossible one-year swing that strongly
suggests 2000's value is a placeholder, not a real measurement. The model, trained on genuine
relationships between low income composition and low life expectancy, reasonably predicted a much lower
value given what looked like an extremely poor economic indicator. The error is therefore best
attributed to a **data-quality artifact**, not a genuine model failure.

**Example 2 — Zimbabwe, 2015.** True life expectancy 67.0 years; predicted 58.1 (error 8.9 years).
Unlike Example 1, this appears to be a genuine model limitation: Zimbabwe's 2015 `hiv_aids` (6.2%) and
elevated `adult_mortality` place it in a profile the model associates with lower life expectancy, based
on patterns learned from earlier years in the same country and similar countries where high HIV/AIDS
prevalence co-occurred with much lower life expectancy (e.g. Zimbabwe's own 2000–2005 values, Section
2.3, where HIV/AIDS prevalence was over 40% and life expectancy was in the mid-40s). By 2015, Zimbabwe's
life expectancy had recovered substantially, likely reflecting improved antiretroviral treatment access
that isn't fully captured by the `hiv_aids` prevalence percentage alone — the model under-weights this
because a 6.2% prevalence rate still statistically co-occurs with the country's own historically low
readings in the training data.

**Broader pattern (subgroup-level):** contrary to our initial hypothesis, mean absolute error does
**not** differ meaningfully between Developing (1.14 years) and Developed (1.18 years) countries (Figure
below) — subgroup alone is not predictive of error size. What actually characterizes the worst errors is
being an outlier on a specific indicator (extreme HIV/AIDS prevalence, or a suspected data artifact),
not a country's broad development category.

![Random Forest absolute error by development status](../figures/13_error_by_status.png)

## 7.4 Answers to Our Three Questions

**Q1 — How much does healthcare expenditure (as % GDP) correlate with life expectancy vs. schooling
years?**

The dataset does not contain a column literally named "healthcare expenditure as % of GDP" — the closest
available proxy is `percentage_expenditure` (health expenditure as % of GDP *per capita*, per the
source's own definition); we also checked `total_expenditure` (government health spending as % of
*total government* spending, a different concept) for completeness.

| Variable | Raw correlation with life expectancy | Standardized regression coefficient (controlling for the others + income) |
|---|---|---|
| `schooling` | **0.732** | **4.09** |
| `income_composition` (control) | — | 3.01 |
| `percentage_expenditure` | 0.374 | 1.00 |
| `total_expenditure` | 0.231 | 0.29 |

**Answer:** Schooling correlates with life expectancy roughly 2–3× more strongly than either healthcare
expenditure measure, both before and after controlling for income — this rules out the simplest
alternative explanation (that expenditure's relationship is just income in disguise), since schooling's
association holds up independently while the expenditure measures shrink further under the same control.

![Healthcare expenditure vs. schooling](../figures/14_q1_expenditure_vs_schooling.png)

**Q2 — Do Developing vs. Developed status countries show different key predictors?**

| | Developed (top predictors) | Developing (top predictors) |
|---|---|---|
| By correlation | income_composition (0.70), thinness measures (~0.59), adult_mortality (0.49) | adult_mortality (0.66), schooling (0.66), income_composition (0.61) |
| By Random Forest importance | adult_mortality (0.48), income_composition (0.23) | **hiv_aids (0.60)**, adult_mortality (0.18) |

**Answer:** Yes. The clearest and most consistent signal across both methods: `hiv_aids` is the single
dominant predictor for Developing countries (60% of Random Forest importance) but is statistically
irrelevant for Developed countries (near-zero correlation; HIV/AIDS prevalence sits almost uniformly at
the dataset's minimum in that subgroup). Disease burden differentiates outcomes for Developing countries
in a way it simply does not for Developed ones. *Caveat: the Developed subgroup has only 512 rows versus
2,416 for Developing, so its estimates are noisier and this finding should be read as suggestive rather
than definitive.*

![Top predictors, Developed vs. Developing](../figures/15_q2_developed_vs_developing.png)

**Q3 — Which single factor (immunization rates, BMI, HIV/AIDS prevalence) has the largest negative
effect?**

| Factor | Correlation with life expectancy |
|---|---|
| **hiv_aids** | **−0.557** |
| hepatitis_b (immunization) | +0.339 |
| polio (immunization) | +0.460 |
| diphtheria (immunization) | +0.475 |
| bmi | +0.564 |

**Answer:** HIV/AIDS prevalence, by a wide margin — it is the *only* factor among the three candidate
types with a negative relationship to life expectancy at all in this data; immunization coverage and BMI
are both positively correlated. **Honest caveat:** BMI's positive sign here is counterintuitive given
real-world evidence that very high BMI is a health risk; this dataset's BMI column is known to have
data-quality issues in the source, so we would not treat BMI's sign here as a reliable real-world finding
without further scrutiny of that specific column.

![Signed correlation of candidate negative factors](../figures/16_q3_negative_factors.png)

---

# 8. Limitations and Next Steps

**Data limitations:**
- The dataset ends in 2015; it cannot speak to any trends after that year.
- No measure of political stability, conflict intensity, or pandemic shocks (e.g. nothing resembling
  COVID-19) is present — factors we would expect to matter a great deal for life expectancy.
- At least one confirmed data-quality artifact exists (Section 7.3, Example 1: `income_composition =
  0.000` for Saint Vincent and the Grenadines, 2000) that our automated missing-value pipeline could not
  catch because it is a valid float, not a null. A more thorough audit would flag suspicious "round
  number" values (exact zeros, repeated identical values across consecutive years) for manual review.

**Method limitations:**
- Our train/test split is by row (country-year), not by country. A given country's different years can
  appear on both sides of the split, which is a soft form of information leakage — the model can benefit
  from having seen a country's *other* years even when a specific year is held out. A stricter
  evaluation would split by country entirely.
- Hyperparameter search used only 12 random candidates and 3-fold CV for the ensemble models (rather
  than a full grid and 5-fold CV) due to limited single-core compute available for this project; a wider
  search would likely close some of the small remaining gap between Random Forest and Gradient Boosting,
  though it is unlikely to change the overall linear-vs-non-linear conclusion.
- We did not model the Developed subgroup separately with a dedicated tuned model (Section 7.4, Q2) —
  only with correlation and a single untuned Random Forest per subgroup — so those findings should be
  treated as exploratory.

**What we would do with more time or better data:**
- Split by country for a stricter train/test evaluation.
- Investigate and clean suspected-artifact values (round zeros, flat repeated sequences) beyond simple
  null-checking.
- Bring in a conflict/political-stability index as an additional feature, particularly to test whether
  it explains some of the largest remaining Random Forest errors.
- Run a wider hyperparameter search with more compute to check whether Gradient Boosting can close the
  small remaining gap to Random Forest.

We do not claim this model is ready for any real-world policy or resource-allocation use — it is a
data-analysis exercise on a historical, incomplete dataset, and the limitations above should be read as
binding constraints on how far the findings generalize.

---

# 9. Contributions

| Member | Student ID | Contribution |
|---|---|---|
| [Full Name] | [Student ID] | [fill in — e.g. data audit & preprocessing (notebooks 01–02), report Sections 1–2] |
| [Full Name] | [Student ID] | [fill in — e.g. feature engineering & modeling (notebooks 03–04), report Sections 4–6] |
| [Full Name] | [Student ID] | [fill in — e.g. evaluation, error analysis & research questions (notebooks 05–06), report Sections 7–8] |

*Fill this in honestly based on who actually did what. Your instructor's template requires every member
to own identifiable work — this should reflect your real GitHub commit history, not be assigned after
the fact.*

---

# References

**Dataset:** Kumar Rajarshi, "Life Expectancy (WHO)," Kaggle, originally sourced from the World Health
Organization's Global Health Observatory and United Nations data repositories.
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

**Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter (see `requirements.txt` for
versions).

**AI assistance:** Anthropic's Claude (Sonnet) was used throughout this project — for writing
preprocessing/modeling code, generating and executing the Jupyter notebooks, drafting this report, and
checking analysis results against the actual computed output before writing conclusions. **Every group
member should read and understand the full codebase and this report before submission**, and be
prepared to explain and defend any decision in it, per standard academic-integrity expectations for
AI-assisted work. Adjust or expand this disclosure to match your course's specific AI-use policy.
