"""Smoke tests for the preprocessing/modeling pipeline. Use a tiny synthetic frame — fast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import SEED  # noqa: E402
from src.modeling import (  # noqa: E402
    evaluate,
    metrics,
    set_seed,
    train_baseline,
    train_model,
)
from src.preprocessing import (  # noqa: E402
    TARGET,
    clean,
    engineer_features,
    fit_transform_pipeline,
    make_split,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 2000
    dob = pd.to_datetime("1980-01-01") + pd.to_timedelta(rng.integers(0, 365 * 30, size=n), unit="D")
    txn_date = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 90, size=n), unit="D")
    df = pd.DataFrame(
        {
            "TransactionID": [f"T{i}" for i in range(n)],
            "CustomerID": [f"C{i % 500}" for i in range(n)],
            "CustomerDOB": dob,
            "CustGender": rng.choice(["M", "F"], size=n),
            "CustLocation": rng.choice(["MUMBAI", "DELHI", "BANGALORE", "CHENNAI", "KOLKATA"], size=n),
            "CustAccountBalance": rng.uniform(100, 100_000, size=n),
            "TransactionDate": txn_date,
            "TransactionTime": rng.integers(0, 235959, size=n),
            TARGET: rng.lognormal(mean=6, sigma=1.2, size=n),
        }
    )
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)
    df.loc[df.sample(20, random_state=SEED).index, TARGET] = 0
    return df


def test_clean_drops_duplicates_and_zero_amounts(synthetic_df):
    cleaned = clean(synthetic_df)
    assert cleaned["TransactionID"].is_unique
    assert (cleaned[TARGET] > 0).all()


def test_engineer_features_no_target_leakage(synthetic_df):
    cleaned = clean(synthetic_df)
    features, _ = engineer_features(cleaned)
    assert TARGET in features.columns, "Target should pass through engineer_features (dropped later in pipeline)."
    leak_candidates = [c for c in features.columns if c != TARGET and "Amount" in c]
    assert leak_candidates == [], f"Unexpected target-derived features: {leak_candidates}"


def test_make_split_shapes_and_seed_reproducible(synthetic_df):
    cleaned = clean(synthetic_df)
    train1, val1, test1 = make_split(cleaned, test_size=0.2, val_size=0.1, seed=SEED)
    train2, val2, test2 = make_split(cleaned, test_size=0.2, val_size=0.1, seed=SEED)
    total = len(train1) + len(val1) + len(test1)
    assert total == len(cleaned)
    assert abs(len(test1) / len(cleaned) - 0.2) < 0.02
    assert abs(len(val1) / len(cleaned) - 0.1) < 0.02
    assert train1["TransactionID"].tolist() == train2["TransactionID"].tolist()


def test_fit_transform_pipeline_no_leakage(synthetic_df):
    cleaned = clean(synthetic_df)
    train, val, test = make_split(cleaned, test_size=0.2, val_size=0.1, seed=SEED)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), artifacts = fit_transform_pipeline(train, val, test)
    assert list(X_tr.columns) == list(X_val.columns) == list(X_te.columns)
    assert TARGET not in X_tr.columns
    assert "location_freq" in artifacts
    assert len(y_tr) == len(X_tr)


def test_metrics_returns_expected_keys():
    set_seed()
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 305.0, 395.0])
    m = metrics(y_true, y_pred)
    assert set(m) == {"MAE", "RMSE", "R2", "MAPE"}
    assert m["MAE"] > 0
    assert m["RMSE"] >= m["MAE"]


def test_train_baseline_predicts_correct_shape(synthetic_df):
    cleaned = clean(synthetic_df)
    train, val, test = make_split(cleaned, test_size=0.2, val_size=0.1, seed=SEED)
    (X_tr, y_tr), (X_val, y_val), _, _ = fit_transform_pipeline(train, val, test)
    model = train_baseline(X_tr, y_tr)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_lightgbm_runs_smoke(synthetic_df):
    cleaned = clean(synthetic_df)
    train, val, test = make_split(cleaned, test_size=0.2, val_size=0.1, seed=SEED)
    (X_tr, y_tr), (X_val, y_val), _, _ = fit_transform_pipeline(train, val, test)
    model = train_model("lightgbm", X_tr, y_tr, params={"n_estimators": 50})
    scores = evaluate(model, X_val, y_val)
    assert scores["MAE"] > 0
    assert np.isfinite(scores["MAE"])
