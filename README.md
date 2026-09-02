# CSE437 Final Project — Predicting Life Expectancy

## Problem

We predict a country's life expectancy in a given year (a **regression** problem, target variable
`life expectancy`) from health, economic, and social indicators, and analyze which indicators matter
most and where the model's predictions can be trusted.

## Dataset

**WHO Life Expectancy dataset** — 2,938 rows × 22 columns, 193 countries, 2000–2015, originally
compiled from WHO/UN data (commonly distributed via Kaggle; see `data/README.md` for the exact source
link). Raw file committed at `data/raw/Life_Expectancy_Data.csv`.

## Three Questions

1. How much does healthcare expenditure (as % of GDP) correlate with life expectancy vs. schooling
   years?
2. Do Developing vs. Developed status countries show different key predictors of life expectancy?
3. Which single factor — immunization rates, BMI, or HIV/AIDS prevalence — has the largest negative
   effect on life expectancy?

See `report/report.md` (or `report/report.pdf`) for the full write-up and answers.

**Answers at a glance:**

| Question | Answer |
|---|---|
| Q1: Expenditure vs. schooling | Schooling correlates ~2-3x more strongly (r≈0.73) than healthcare expenditure (r≈0.37/0.23), even after controlling for income. |
| Q2: Developed vs. Developing predictors | Yes, they differ — HIV/AIDS is the dominant predictor for Developing countries but nearly irrelevant for Developed ones. |
| Q3: Largest negative factor | HIV/AIDS prevalence (r≈-0.56) — the only negative factor among the three candidates; immunization and BMI are both positive. |

## How to run everything

1. **Clone and set up an environment:**
   ```bash
   git clone <this-repo-url>
   cd cse437-life-expectancy-group1
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **In VS Code:** open this folder (`File > Open Folder...`), install the Python and Jupyter
   extensions if prompted, then open each notebook in `notebooks/` and select the `.venv` kernel
   (top-right of the notebook — "Select Kernel").
3. **Run the notebooks in numeric order, top to bottom, on a fresh kernel each time:**
   - `01_data_audit_and_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_feature_engineering.ipynb`
   - `04_modeling_and_tuning.ipynb`
   - `05_evaluation_and_error_analysis.ipynb`
   - `06_research_questions.ipynb`

   Each notebook reads what the previous one saved to `data/processed/`, so **order matters**. Use
   "Run All" (VS Code Jupyter toolbar) for a clean, fresh-kernel run.

## Repository structure

```
├── data/            # raw + processed data (see data/README.md)
├── notebooks/       # 01-05, run in order
├── src/utils.py      # shared functions imported by the notebooks
├── models/          # saved trained models + test-set results
├── figures/          # every figure used in the report
└── report/           # report.md / report.pdf (10-page write-up)
```

## Results summary

| Model | RMSE (years) | MAE (years) | R² |
|---|---|---|---|
| Ridge Regression | 3.92 | 2.97 | 0.827 |
| **Random Forest (best)** | **1.84** | **1.15** | **0.962** |
| Gradient Boosting | 1.87 | 1.26 | 0.961 |

Full discussion, including feature engineering rationale and error analysis, is in `report/report.pdf`.
