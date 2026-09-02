"""
Shared utility functions for the Life Expectancy project.
Imported by the notebooks in notebooks/ so logic isn't duplicated.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RAW_PATH = "../data/raw/Life_Expectancy_Data.csv"
PROCESSED_PATH = "../data/processed/life_expectancy_clean.csv"

# The raw column names have inconsistent spacing (e.g. "Life expectancy ",
# " BMI "). We standardize them once, here, so every notebook agrees.
COLUMN_RENAME_MAP = {
    "Life expectancy ": "life_expectancy",
    "Adult Mortality": "adult_mortality",
    "infant deaths": "infant_deaths",
    "Alcohol": "alcohol",
    "percentage expenditure": "percentage_expenditure",
    "Hepatitis B": "hepatitis_b",
    "Measles ": "measles",
    " BMI ": "bmi",
    "under-five deaths ": "under_five_deaths",
    "Polio": "polio",
    "Total expenditure": "total_expenditure",
    "Diphtheria ": "diphtheria",
    " HIV/AIDS": "hiv_aids",
    "GDP": "gdp",
    "Population": "population",
    " thinness  1-19 years": "thinness_10_19_years",
    " thinness 5-9 years": "thinness_5_9_years",
    "Income composition of resources": "income_composition",
    "Schooling": "schooling",
    "Country": "country",
    "Year": "year",
    "Status": "status",
}


def load_raw(path=RAW_PATH):
    """Load the raw CSV and standardize column names (nothing else)."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def load_processed(path=PROCESSED_PATH):
    return pd.read_csv(path)


def regression_report(y_true, y_pred, label=""):
    """Return a dict of standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"model": label, "RMSE": rmse, "MAE": mae, "R2": r2}
