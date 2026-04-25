"""Data loading, cleaning, feature engineering and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import SEED

TARGET = "TransactionAmount (INR)"
RAW_COLUMNS = [
    "TransactionID",
    "CustomerID",
    "CustomerDOB",
    "CustGender",
    "CustLocation",
    "CustAccountBalance",
    "TransactionDate",
    "TransactionTime",
    TARGET,
]


def _parse_two_digit_year_date(series: pd.Series, pivot_year: int = 25) -> pd.Series:
    """Parse dates like '10/1/94' (d/m/yy). Two-digit years > pivot_year => 19xx, else 20xx.

    Default `pivot_year=25` matches the dataset's transaction year (2016) — DOBs above 25
    cannot be customers born in 2025+.
    """
    parsed = pd.to_datetime(series, format="%d/%m/%y", errors="coerce")
    mask_future = parsed.dt.year > (2000 + pivot_year)
    parsed.loc[mask_future] = parsed.loc[mask_future] - pd.DateOffset(years=100)
    return parsed


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw bank transactions CSV with parsed dates."""
    df = pd.read_csv(path)
    df["CustomerDOB"] = _parse_two_digit_year_date(df["CustomerDOB"], pivot_year=25)
    df["TransactionDate"] = _parse_two_digit_year_date(df["TransactionDate"], pivot_year=25)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, drop garbage rows, fill missing values, drop zero-amount rows."""
    df = df.drop_duplicates(subset=["TransactionID"]).copy()

    df = df[df[TARGET] > 0].copy()

    df = df[df["CustGender"].isin(["M", "F"]) | df["CustGender"].isna()].copy()

    df["CustGender"] = df["CustGender"].fillna("unknown")
    df["CustLocation"] = df["CustLocation"].fillna("unknown")
    df["CustAccountBalance"] = df["CustAccountBalance"].fillna(df["CustAccountBalance"].median())

    dob_median = df["CustomerDOB"].dropna().median()
    df["CustomerDOB"] = df["CustomerDOB"].fillna(dob_median)

    return df.reset_index(drop=True)


def _safe_age_years(dob: pd.Series, ref: pd.Series) -> pd.Series:
    delta_days = (ref - dob).dt.days
    age = delta_days / 365.25
    age = age.where((age >= 0) & (age <= 110), np.nan)
    return age.fillna(age.median())


def engineer_features(df: pd.DataFrame, location_freq: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Build derived features. Returns (features_df, fitted_artifacts_dict).

    `location_freq` should be passed for val/test (computed on train) to avoid leakage.
    """
    out = df.copy()

    out["Age"] = _safe_age_years(out["CustomerDOB"], out["TransactionDate"])
    out["AgeBucket"] = pd.cut(
        out["Age"],
        bins=[0, 25, 35, 45, 55, 65, 110],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
        include_lowest=True,
    ).astype(str)

    time_str = out["TransactionTime"].astype(int).astype(str).str.zfill(6)
    out["TransactionHour"] = time_str.str[:2].astype(int).clip(0, 23)
    out["TransactionDayOfWeek"] = out["TransactionDate"].dt.dayofweek
    out["TransactionMonth"] = out["TransactionDate"].dt.month

    out["LogBalance"] = np.log1p(out["CustAccountBalance"].clip(lower=0))

    if location_freq is None:
        location_freq = out["CustLocation"].value_counts().to_dict()
    out["LocationFreq"] = out["CustLocation"].map(location_freq).fillna(0).astype(int)

    out = pd.get_dummies(out, columns=["CustGender", "AgeBucket"], drop_first=False)

    drop_cols = [
        "TransactionID",
        "CustomerID",
        "CustomerDOB",
        "CustLocation",
        "TransactionDate",
        "TransactionTime",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out, {"location_freq": location_freq}


def _stratify_bins(y: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(np.log1p(y), q=n_bins, labels=False, duplicates="drop")


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train / val / test using log1p(target) quantile bins."""
    bins = _stratify_bins(df[TARGET])
    train_full, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=bins)
    bins_train_full = _stratify_bins(train_full[TARGET])
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_full,
        test_size=relative_val,
        random_state=seed,
        stratify=bins_train_full,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int = 200_000, seed: int = SEED) -> pd.DataFrame:
    """Down-sample preserving the log1p(target) distribution."""
    if len(df) <= n:
        return df.copy()
    bins = _stratify_bins(df[TARGET])
    sampled, _ = train_test_split(df, train_size=n, random_state=seed, stratify=bins)
    return sampled.reset_index(drop=True)


def fit_transform_pipeline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], dict]:
    """Apply feature engineering with no leakage: fit artifacts on train, apply to val/test."""
    train_feat, artifacts = engineer_features(train)
    val_feat, _ = engineer_features(val, location_freq=artifacts["location_freq"])
    test_feat, _ = engineer_features(test, location_freq=artifacts["location_freq"])

    val_feat = val_feat.reindex(columns=train_feat.columns, fill_value=0)
    test_feat = test_feat.reindex(columns=train_feat.columns, fill_value=0)

    y_train = train_feat.pop(TARGET)
    y_val = val_feat.pop(TARGET)
    y_test = test_feat.pop(TARGET)

    return (train_feat, y_train), (val_feat, y_val), (test_feat, y_test), artifacts
