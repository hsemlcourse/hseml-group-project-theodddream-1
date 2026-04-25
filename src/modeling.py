"""Model training, evaluation, hyper-parameter tuning, ensembling and persistence."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PYTHONHASHSEED for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R², MAPE — the multi-metric report. Primary metric for selection: MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    nz = y_true > 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def evaluate(model, X, y) -> dict[str, float]:
    return metrics(y, model.predict(X))


def train_baseline(X_train, y_train) -> LinearRegression:
    """LinearRegression out-of-the-box, as required by the CP1 criteria."""
    set_seed()
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def _make_model(name: str, params: dict[str, Any] | None = None):
    params = params or {}
    if name == "linear":
        return LinearRegression(**params)
    if name == "ridge":
        return Pipeline(
            [("scaler", StandardScaler(with_mean=False)), ("model", Ridge(random_state=SEED, **params))]
        )
    if name == "knn":
        return Pipeline(
            [("scaler", StandardScaler(with_mean=False)), ("model", KNeighborsRegressor(**params))]
        )
    if name == "random_forest":
        defaults = {"n_estimators": 200, "n_jobs": -1, "random_state": SEED}
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if name == "lightgbm":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "n_jobs": -1,
            "random_state": SEED,
            "verbose": -1,
        }
        defaults.update(params)
        return LGBMRegressor(**defaults)
    raise ValueError(f"Unknown model name: {name!r}")


def train_model(name: str, X_train, y_train, params: dict[str, Any] | None = None):
    set_seed()
    model = _make_model(name, params)
    model.fit(X_train, y_train)
    return model


def tune_lightgbm(X_train, y_train, n_iter: int = 20, cv_folds: int = 3) -> tuple[LGBMRegressor, dict]:
    """RandomizedSearchCV over a small LightGBM grid. Returns (best_estimator, best_params)."""
    set_seed()
    base = LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_random_forest(X_train, y_train, n_iter: int = 10, cv_folds: int = 3) -> tuple[RandomForestRegressor, dict]:
    set_seed()
    base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def build_ensemble(named_models: list[tuple[str, Any]], weights: list[float] | None = None) -> VotingRegressor:
    """VotingRegressor over already-instantiated estimators (not yet fitted)."""
    return VotingRegressor(estimators=named_models, weights=weights, n_jobs=-1)


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    return joblib.load(path)


def experiments_table(rows: list[dict]) -> pd.DataFrame:
    """Compose a tidy DataFrame from rows like {'model': ..., 'MAE': ..., ...}."""
    cols_order = ["model", "hypothesis", "params", "MAE", "RMSE", "R2", "MAPE", "fit_seconds", "notes"]
    df = pd.DataFrame(rows)
    ordered = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    return df[ordered]
