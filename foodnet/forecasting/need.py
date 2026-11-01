"""Need forecaster (light): expected daily need (kg) per recipient.

Gives the matching engine a demand signal. Baseline = the mean of the same
weekday over the previous four weeks; the model adds calendar and trend
features with a gradient-boosted regressor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from foodnet.forecasting import metrics

FEATURES = ["dow", "month", "is_holiday", "temp_c", "lag7", "same_dow_4wk", "roll28", "recipient_id"]


def build_features(need: pd.DataFrame) -> pd.DataFrame:
    df = need.sort_values(["recipient_id", "date"]).copy()
    g = df.groupby("recipient_id", sort=False)["need_kg"]
    df["lag7"] = g.shift(7)
    df["same_dow_4wk"] = sum(g.shift(7 * k) for k in range(1, 5)) / 4
    df["roll28"] = g.transform(lambda x: x.shift(1).rolling(28, min_periods=14).mean())
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["recipient_id"] = df["recipient_id"].astype("category")
    return df.dropna(subset=["same_dow_4wk", "roll28"]).reset_index(drop=True)


class NeedForecaster:
    def __init__(self, random_state=0):
        self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                                   categorical_features="from_dtype",
                                                   random_state=random_state)

    def fit(self, df):
        self.model.fit(df[FEATURES], df["need_kg"])
        return self

    def predict(self, df) -> np.ndarray:
        return self.model.predict(df[FEATURES]).clip(0)


def backtest(feat: pd.DataFrame, test_days=84) -> pd.DataFrame:
    cutoff = np.sort(feat["date"].unique())[-test_days]
    train, test = feat[feat["date"] < cutoff], feat[feat["date"] >= cutoff]
    gbm = NeedForecaster().fit(train).predict(test)
    base = test["same_dow_4wk"].to_numpy()
    y = test["need_kg"]
    return pd.DataFrame([
        {"model": "same_weekday_4wk_mean", "mae": metrics.mae(y, base), "wape": metrics.wape(y, base)},
        {"model": "gbm", "mae": metrics.mae(y, gbm), "wape": metrics.wape(y, gbm)},
    ])
