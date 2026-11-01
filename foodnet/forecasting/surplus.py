"""Surplus forecaster: predicts tomorrow's surplus (kg) per donor x category.

The forecast is made the evening before (day t-1) so recipients and volunteers
can be booked a day ahead. Features therefore only use information available
at that point: past surplus and sales, the calendar, the weather forecast and
planned promotions.

Models
------
* ``SeasonalNaive``: surplus on the same weekday last week (the baseline).
* ``QuantileSurplusForecaster``: gradient-boosted trees (scikit-learn's
  HistGradientBoostingRegressor) fitted with quantile loss at P10, P50 and P90,
  giving a range rather than one number. LightGBM is a drop-in alternative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from foodnet.forecasting import metrics

KEYS = ["donor_id", "category"]
CAT_FEATURES = ["donor_type", "category", "donor_id"]
NUM_FEATURES = [
    "lag1", "lag7", "lag14", "roll7", "roll28", "roll7_std", "zero_share28",
    "demand_lag7", "demand_roll7",
    "dow", "month", "is_holiday", "is_pre_holiday", "temp_c", "rain", "promo",
]
FEATURES = NUM_FEATURES + CAT_FEATURES
QUANTILES = (0.1, 0.5, 0.9)


def build_features(sales: pd.DataFrame) -> pd.DataFrame:
    """Add lag / rolling features. Every lag is shifted so day t only sees <= t-1."""
    df = sales.sort_values(KEYS + ["date"]).copy()
    g = df.groupby(KEYS, sort=False)
    s = g["surplus_kg"]
    df["lag1"] = s.shift(1)
    df["lag7"] = s.shift(7)
    df["lag14"] = s.shift(14)
    df["roll7"] = s.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df["roll28"] = s.transform(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
    df["roll7_std"] = s.transform(lambda x: x.shift(1).rolling(7, min_periods=3).std())
    df["zero_share28"] = s.transform(lambda x: (x.shift(1) == 0).rolling(28, min_periods=7).mean())
    d = g["demand_units"]
    df["demand_lag7"] = d.shift(7)  # drives tomorrow's stock order
    df["demand_roll7"] = d.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    for c in ("is_holiday", "is_pre_holiday", "rain", "promo"):
        df[c] = df[c].astype(int)
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df.dropna(subset=["lag14", "roll28"]).reset_index(drop=True)


class SeasonalNaive:
    """Forecast = surplus on the same weekday last week."""

    def fit(self, df):  # noqa: D401 - nothing to learn
        return self

    def predict(self, df) -> pd.DataFrame:
        p = df["lag7"].to_numpy()
        return pd.DataFrame({"p10": p, "p50": p, "p90": p}, index=df.index)


class QuantileSurplusForecaster:
    def __init__(self, quantiles=QUANTILES, max_iter=300, learning_rate=0.06, random_state=0):
        self.quantiles = quantiles
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=31, min_samples_leaf=40,
                           categorical_features="from_dtype", random_state=random_state)
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, df: pd.DataFrame):
        X, y = df[FEATURES], df["surplus_kg"]
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
            self.models[q] = m.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles]).clip(0)
        preds.sort(axis=1)  # guard against crossing quantiles
        cols = [f"p{int(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(preds, columns=cols, index=df.index)


def evaluate(y, pred: pd.DataFrame) -> dict:
    return {
        "mae": metrics.mae(y, pred["p50"]),
        "wape": metrics.wape(y, pred["p50"]),
        "pinball_p10": metrics.pinball(y, pred["p10"], 0.1),
        "pinball_p50": metrics.pinball(y, pred["p50"], 0.5),
        "pinball_p90": metrics.pinball(y, pred["p90"], 0.9),
        "coverage_p10_p90": metrics.coverage(y, pred["p10"], pred["p90"]),
    }


def rolling_backtest(feat: pd.DataFrame, n_folds=4, fold_days=21, model_factory=QuantileSurplusForecaster):
    """Rolling-origin backtest: train on everything before each fold, test on the fold.

    Returns (per-fold metrics DataFrame, out-of-sample predictions DataFrame).
    """
    dates = np.sort(feat["date"].unique())
    rows, preds = [], []
    for k in range(n_folds, 0, -1):
        test_start = dates[-k * fold_days]
        test_end = dates[-(k - 1) * fold_days - 1] if k > 1 else dates[-1]
        train = feat[feat["date"] < test_start]
        test = feat[(feat["date"] >= test_start) & (feat["date"] <= test_end)]
        for name, model in (("seasonal_naive", SeasonalNaive()), ("gbm_quantile", model_factory())):
            p = model.fit(train).predict(test)
            m = evaluate(test["surplus_kg"], p)
            m.update(model=name, fold_start=str(pd.Timestamp(test_start).date()), n=len(test))
            rows.append(m)
            out = test[["date", "donor_id", "category", "surplus_kg"]].copy()
            out["donor_id"] = out["donor_id"].astype(str)
            out["category"] = out["category"].astype(str)
            out[["p10", "p50", "p90"]] = p[["p10", "p50", "p90"]].to_numpy()
            out["model"] = name
            preds.append(out)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
