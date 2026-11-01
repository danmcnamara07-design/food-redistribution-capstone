"""Uncollected-listing predictor: probability that a planned collection fails.

Trained on past collections. Used by the simulation to add a backup volunteer
when the predicted failure probability crosses a threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["lead_time_h", "distance_km", "volunteer_past_rate", "rain", "hour", "dow",
            "listing_kg", "recipient_accept_rate"]


def _X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["rain"] = X["rain"].astype(int)
    X["log_lead"] = np.log1p(X["lead_time_h"])
    return X


class RiskPredictor:
    """Gradient-boosted classifier (default) or logistic regression baseline."""

    def __init__(self, kind: str = "gbm", threshold: float = 0.25, random_state: int = 0):
        self.kind, self.threshold = kind, threshold
        if kind == "gbm":
            self.model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                                        min_samples_leaf=50, random_state=random_state)
        elif kind == "logistic":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        else:
            raise ValueError(kind)

    def fit(self, df: pd.DataFrame):
        self.model.fit(_X(df), df["failed"].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_X(df))[:, 1]

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df) >= self.threshold


def evaluate(y, p, threshold) -> dict:
    y = np.asarray(y).astype(int)
    flag = p >= threshold
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "precision_at_threshold": float(precision_score(y, flag, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, flag, zero_division=0)),
        "flag_rate": float(flag.mean()),
        "base_rate": float(y.mean()),
        "threshold": threshold,
    }


def time_split_eval(collections: pd.DataFrame, test_share: float = 0.2, threshold: float = 0.25) -> pd.DataFrame:
    """Train on the earliest 80% of collections by date, test on the latest 20%."""
    df = collections.sort_values("date")
    cut = int(len(df) * (1 - test_share))
    train, test = df.iloc[:cut], df.iloc[cut:]
    rows = [{"model": "always_collected", **evaluate(test["failed"], np.zeros(len(test)), threshold)}]
    for kind in ("logistic", "gbm"):
        m = RiskPredictor(kind, threshold).fit(train)
        rows.append({"model": kind, **evaluate(test["failed"], m.predict_proba(test), threshold)})
    return pd.DataFrame(rows)
