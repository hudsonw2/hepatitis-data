import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "hepatitis.data"

COLUMNS = [
    "class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise",
    "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
    "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
    "protime", "histology",
]

CATEGORICAL_COLS = [
    "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia",
    "liver_big", "liver_firm", "spleen_palpable", "spiders", "ascites",
    "varices", "histology",
]
CONTINUOUS_COLS = ["age", "bilirubin", "alk_phosphate", "sgot", "albumin", "protime"]


def load_raw() -> pd.DataFrame:
    """Load the raw comma-separated file, treating '?' as missing."""
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, na_values="?")

 
    df["class"] = df["class"].map({1: 1, 2: 0})  


    for col in CATEGORICAL_COLS:
        df[col] = df[col].map({1: 0, 2: 1})

    return df


def make_dropna_version(df: pd.DataFrame) -> pd.DataFrame:
    """Paper's approach: drop any row with at least one missing value."""
    return df.dropna().reset_index(drop=True)


def make_imputed_version(df: pd.DataFrame) -> pd.DataFrame:
    """Improved approach: median-impute continuous, mode-impute categorical."""
    df_imputed = df.copy()

    for col in CONTINUOUS_COLS:
        median_val = df_imputed[col].median()
        df_imputed[col] = df_imputed[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        mode_val = df_imputed[col].mode().iloc[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)

    return df_imputed


def summarize(df: pd.DataFrame, label: str) -> None:
    n_die = int((df["class"] == 1).sum())
    n_live = int((df["class"] == 0).sum())
    print(f"[{label}] instances: {len(df)}  |  DIE: {n_die}  LIVE: {n_live}")


def main() -> None:
    raw = load_raw()
    print(f"Loaded raw data: {raw.shape[0]} rows, {raw.shape[1]} columns")
    print(f"Missing values per column:\n{raw.isna().sum()}\n")

    dropna_df = make_dropna_version(raw)
    imputed_df = make_imputed_version(raw)

    summarize(dropna_df, "drop-missing")
    summarize(imputed_df, "imputed")

    dropna_out = DATA_DIR / "hepatitis_dropna.csv"
    imputed_out = DATA_DIR / "hepatitis_imputed.csv"

    dropna_df.to_csv(dropna_out, index=False)
    imputed_df.to_csv(imputed_out, index=False)

    print(f"\nSaved: {dropna_out}")
    print(f"Saved: {imputed_out}")


if __name__ == "__main__":
    main()
