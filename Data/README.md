# Data

## Source

**WHO Life Expectancy dataset**, originally compiled from the World Health Organization's Global
Health Observatory (GHO) and United Nations data repositories, commonly distributed via Kaggle as
"Life Expectancy (WHO)":
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

## Size

2,938 rows × 22 columns (~300 KB as CSV). Well under the 50 MB raw-data limit, so the raw file is
committed directly at `data/raw/Life_Expectancy_Data.csv` — no external download link is required.

## Coverage

193 countries, years 2000–2015, one row per country-year.

## Contents

- `raw/Life_Expectancy_Data.csv` — the original file, untouched.
- `processed/life_expectancy_clean.csv` — output of `notebooks/02_preprocessing.ipynb`: missing
  values imputed, skewed features log-transformed, `status` encoded, `country` retained as an ID
  column.
- `processed/life_expectancy_features.csv` — output of `notebooks/03_feature_engineering.ipynb`:
  the 17 selected modeling features, `life_expectancy` target, and `country` ID column.
- `processed/test_predictions.csv` — output of `notebooks/04_modeling_and_tuning.ipynb`: the held-out
  test set with each tuned model's predictions attached, used for error analysis in notebook 05.

## How to obtain the data (if not already present)

The raw CSV is already committed in this repository at `data/raw/`. If you need to re-download the
original source file, it is available from the Kaggle link above (free Kaggle account required).
