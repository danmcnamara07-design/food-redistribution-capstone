"""Decision-tree experiment for the uncollected-listing predictor (course module 9).

A shallow tree is used as an explainable benchmark, not as the production model.
This module tunes it with time-ordered cross-validation, compares it with the
logistic regression and boosting models in `predictor.py`, checks its feature
importance with permutation importance, and measures how stable its splits are.

Everything is fitted on the earliest 80% of collections and scored on the latest
20%, the same split as `predictor.time_split_eval`, so the numbers line up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text

from foodnet.risk.predictor import RiskPredictor, _X

TARGET = "failed"
DEPTHS = tuple(range(2, 13))
GRID = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
    "min_samples_leaf": [20, 30, 50],
    "min_impurity_decrease": [0.0, 1e-4, 5e-4],
}


# --------------------------------------------------------------------------- #
# Time-ordered folds
# --------------------------------------------------------------------------- #
def time_split(collections: pd.DataFrame, test_share: float = 0.2):
    """Earliest (1 - test_share) of collections by date for training, the rest for test."""
    df = collections.sort_values("date", ignore_index=True)
    cut = int(len(df) * (1 - test_share))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def time_folds(dates: pd.Series, n_folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over whole days: train on every earlier block, validate on the next.

    Days are never split across train and validation, and validation is always later
    than training, so the model never learns from future weeks.
    """
    days = np.sort(pd.Series(dates).unique())
    blocks = np.array_split(days, n_folds + 1)
    d = pd.Series(dates).to_numpy()
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.flatnonzero(d < blocks[k][0])
        val_idx = np.flatnonzero(np.isin(d, blocks[k]))
        folds.append((train_idx, val_idx))
    return folds


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_tree(max_depth=4, min_samples_leaf=30, min_impurity_decrease=0.0,
              class_weight="balanced", random_state: int = 0) -> DecisionTreeClassifier:
    """Failed collections are about 1 in 5, so they get more weight by default."""
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  min_impurity_decrease=min_impurity_decrease,
                                  class_weight=class_weight, random_state=random_state)


def make_forest(random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=30, class_weight="balanced",
                                  n_jobs=-1, random_state=random_state)


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def classification_report(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    flag = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, p)) if p.std() > 0 else float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if p.std() > 0 else 0.5,
        "precision": float(precision_score(y, flag, zero_division=0)),
        "recall": float(recall_score(y, flag, zero_division=0)),
        "f1": float(f1_score(y, flag, zero_division=0)),
        "accuracy": float(accuracy_score(y, flag)),
        "misclassification_rate": float(1 - accuracy_score(y, flag)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def depth_sweep(train: pd.DataFrame, depths=DEPTHS, min_samples_leaf: int = 30, n_folds: int = 4) -> pd.DataFrame:
    """Train vs validation PR-AUC and F1 as max_depth rises. The gap shows overfitting."""
    X, y = _X(train), train[TARGET].astype(int).to_numpy()
    rows = []
    for depth in depths:
        for fold, (tr, va) in enumerate(time_folds(train["date"], n_folds)):
            m = make_tree(max_depth=depth, min_samples_leaf=min_samples_leaf).fit(X.iloc[tr], y[tr])
            p_tr, p_va = m.predict_proba(X.iloc[tr])[:, 1], m.predict_proba(X.iloc[va])[:, 1]
            thr = best_f1_threshold(y[tr], p_tr)
            rows.append({
                "max_depth": depth, "fold": fold, "n_leaves": m.get_n_leaves(),
                "train_pr_auc": average_precision_score(y[tr], p_tr),
                "val_pr_auc": average_precision_score(y[va], p_va),
                "train_f1": f1_score(y[tr], p_tr >= thr),
                "val_f1": f1_score(y[va], p_va >= thr),
            })
    out = pd.DataFrame(rows).groupby("max_depth").mean(numeric_only=True).drop(columns="fold")
    out["gap_pr_auc"] = out["train_pr_auc"] - out["val_pr_auc"]
    return out.reset_index()


def grid_search(train: pd.DataFrame, grid: dict = GRID, n_folds: int = 4) -> tuple[dict, pd.DataFrame]:
    """Small grid over depth, leaf size and impurity decrease, scored by validation PR-AUC."""
    X, y = _X(train), train[TARGET].astype(int)
    gs = GridSearchCV(make_tree(), grid, scoring="average_precision",
                      cv=time_folds(train["date"], n_folds), return_train_score=True, n_jobs=-1)
    gs.fit(X, y)
    res = pd.DataFrame(gs.cv_results_)
    cols = ["param_max_depth", "param_min_samples_leaf", "param_min_impurity_decrease",
            "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    res = res[cols].rename(columns=lambda c: c.replace("param_", "").replace("test_score", "val_pr_auc")
                           .replace("train_score", "train_pr_auc")).sort_values("rank_val_pr_auc")
    return gs.best_params_, res.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Comparison, importance, stability
# --------------------------------------------------------------------------- #
def _threshold_from_last_fold(model, train: pd.DataFrame, n_folds: int = 4) -> float:
    """Pick each model's F1-best threshold on the last training fold, never on test data."""
    tr, va = time_folds(train["date"], n_folds)[-1]
    m = model.fit(train.iloc[tr]) if isinstance(model, RiskPredictor) else \
        clone(model).fit(_X(train.iloc[tr]), train[TARGET].iloc[tr].astype(int))
    p = m.predict_proba(train.iloc[va]) if isinstance(m, RiskPredictor) else \
        m.predict_proba(_X(train.iloc[va]))[:, 1]
    return best_f1_threshold(train[TARGET].iloc[va], p)


def compare_models(train: pd.DataFrame, test: pd.DataFrame, tree_params: dict) -> tuple[pd.DataFrame, dict]:
    """Score the tuned tree against the baseline, logistic regression, boosting and a random forest."""
    Xtr, ytr = _X(train), train[TARGET].astype(int)
    Xte, yte = _X(test), test[TARGET].astype(int)
    sk = {
        "decision_tree": make_tree(**tree_params),
        "decision_tree_unweighted": make_tree(**tree_params, class_weight=None),
        "random_forest": make_forest(),
    }
    rows = [{"model": "always_collected", **classification_report(yte, np.zeros(len(yte)), 0.5)}]
    fitted = {}
    for kind in ("logistic", "gbm"):
        thr = _threshold_from_last_fold(RiskPredictor(kind), train)
        m = RiskPredictor(kind).fit(train)
        rows.append({"model": kind, **classification_report(yte, m.predict_proba(test), thr)})
    for name, est in sk.items():
        thr = _threshold_from_last_fold(est, train)
        m = clone(est).fit(Xtr, ytr)
        fitted[name] = m
        rows.append({"model": name, **classification_report(yte, m.predict_proba(Xte)[:, 1], thr)})
    return pd.DataFrame(rows), fitted


def importance_table(tree: DecisionTreeClassifier, test: pd.DataFrame, n_repeats: int = 10,
                     random_state: int = 0) -> pd.DataFrame:
    """The tree's own (impurity) importance next to permutation importance on test data.

    Impurity importance favours features with many distinct values, such as distance,
    so permutation importance is the one to trust.
    """
    X, y = _X(test), test[TARGET].astype(int)
    perm = permutation_importance(tree, X, y, scoring="average_precision", n_repeats=n_repeats,
                                  random_state=random_state, n_jobs=-1)
    return pd.DataFrame({
        "feature": X.columns,
        "impurity_importance": tree.feature_importances_,
        "permutation_importance_pr_auc": perm.importances_mean,
        "permutation_std": perm.importances_std,
        "distinct_values": [X[c].nunique() for c in X.columns],
    }).sort_values("permutation_importance_pr_auc", ascending=False, ignore_index=True)


def stability(train: pd.DataFrame, tree_params: dict, n_resamples: int = 20, seed: int = 0) -> pd.DataFrame:
    """Refit on bootstrap resamples and record the top two levels of splits.

    Small changes to the data often give a different tree; this shows how often.
    """
    rng = np.random.default_rng(seed)
    X, y = _X(train), train[TARGET].astype(int).to_numpy()
    rows = []
    for i in range(n_resamples):
        idx = rng.integers(0, len(train), len(train))
        t = make_tree(**tree_params).fit(X.iloc[idx], y[idx]).tree_
        name = lambda node: X.columns[t.feature[node]] if t.feature[node] >= 0 else "leaf"  # noqa: E731
        rows.append({"resample": i, "root": name(0), "root_threshold": float(t.threshold[0]),
                     "left": name(t.children_left[0]), "right": name(t.children_right[0])})
    return pd.DataFrame(rows)


def tree_rules(tree: DecisionTreeClassifier, feature_names) -> str:
    """Plain-text rules a coordinator could read."""
    return export_text(tree, feature_names=list(feature_names), show_weights=False, decimals=2)
