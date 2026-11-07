"""Fast checks on a small network. Run with:  python -m pytest -q"""
import warnings

import numpy as np
import pandas as pd
import pytest

from foodnet.config import CATEGORIES, NetworkConfig
from foodnet.data.generator import failure_probability, generate_network
from foodnet.forecasting.surplus import FEATURES, build_features, rolling_backtest
from foodnet.matching.engine import MatchParams, allocation_summary, feasibility, greedy_nearest, optimise_allocation
from foodnet.risk.predictor import time_split_eval

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def net():
    return generate_network(NetworkConfig(n_donors=8, n_recipients=6, n_volunteers=15, n_days=200, seed=1))


@pytest.fixture(scope="module")
def day_inputs(net):
    s = net["sales"]
    day = s["date"].max()
    L = s[(s["date"] == day) & (s["surplus_kg"] >= 1)].merge(net["donors"][["donor_id", "lat", "lon"]], on="donor_id")
    L = L.assign(listing_id=[f"L{i}" for i in range(len(L))], kg=L["surplus_kg"],
                 shelf_life_h=L["category"].map(lambda c: CATEGORIES[c]["shelf_life_h"]))
    R = net["recipients"].merge(net["need"][net["need"]["date"] == day][["recipient_id", "need_kg"]], on="recipient_id")
    return L, R


def test_generator_is_reproducible_and_sane(net):
    again = generate_network(NetworkConfig(n_donors=8, n_recipients=6, n_volunteers=15, n_days=200, seed=1))
    pd.testing.assert_frame_equal(net["sales"], again["sales"])
    s = net["sales"]
    assert (s["surplus_units"] >= 0).all()
    assert (s["sold_units"] <= s["stock_units"]).all()
    assert 0.05 < net["collections"]["failed"].mean() < 0.4


def test_failure_probability_direction():
    base = dict(distance_km=5, reliability=0.9, rain=0, hour=18, listing_kg=20, accept_rate=0.92)
    assert failure_probability(lead_time_h=24, **base) < failure_probability(lead_time_h=1, **base)
    assert failure_probability(lead_time_h=4, **{**base, "reliability": 0.95}) < \
        failure_probability(lead_time_h=4, **{**base, "reliability": 0.6})


def test_features_do_not_leak_today(net):
    f = build_features(net["sales"])
    assert set(FEATURES) <= set(f.columns)
    one = f[(f["donor_id"] == f["donor_id"].iloc[0]) & (f["category"] == f["category"].iloc[0])].sort_values("date")
    # lag1 must equal the previous row's actual surplus
    assert np.allclose(one["lag1"].to_numpy()[1:], one["surplus_kg"].to_numpy()[:-1])


def test_forecaster_beats_seasonal_naive(net):
    m, _ = rolling_backtest(build_features(net["sales"]), n_folds=1, fold_days=21)
    mae = m.set_index("model")["mae"]
    assert mae["gbm_quantile"] < mae["seasonal_naive"]


@pytest.mark.parametrize("alloc_fn", [greedy_nearest, optimise_allocation])
def test_matching_respects_hard_constraints(day_inputs, alloc_fn):
    L, R = day_inputs
    p = MatchParams()
    a = alloc_fn(L, R, p)
    # never more than offered, never beyond capacity
    sent = a.groupby("listing_id")["kg"].sum()
    assert (sent <= L.set_index("listing_id")["kg"].reindex(sent.index) + 1e-6).all()
    got = a.groupby("recipient_id")["kg"].sum()
    cap = R.set_index("recipient_id")["capacity_kg"].reindex(got.index)
    assert (got <= cap + 1e-6).all()
    # only feasible pairs: chilled -> chilled storage, meat -> accepts meat, within distance
    dist, ok = feasibility(L, R, p)
    li = L.reset_index().set_index("listing_id")["index"]
    ri = R.reset_index().set_index("recipient_id")["index"]
    assert all(ok[li[x.listing_id], ri[x.recipient_id]] for x in a.itertuples())
    assert (a["distance_km"] <= p.max_distance_km + 1e-9).all()


def test_optimiser_respects_trip_cap_and_is_fairer(day_inputs):
    L, R = day_inputs
    a = optimise_allocation(L, R, max_trips=5)
    assert len(a) <= 5
    g = allocation_summary(greedy_nearest(L, R), L, R)
    o = allocation_summary(optimise_allocation(L, R), L, R)
    assert o["min_fill_rate"] >= g["min_fill_rate"] - 1e-9


def test_risk_models_beat_base_rate(net):
    r = time_split_eval(net["collections"]).set_index("model")
    assert r.loc["logistic", "pr_auc"] > r.loc["always_collected", "pr_auc"]
    assert r.loc["gbm", "roc_auc"] > 0.6


def test_tree_folds_follow_time_order(net):
    from foodnet.risk.tree import time_folds, time_split
    train, _ = time_split(net["collections"])
    for tr, va in time_folds(train["date"], n_folds=3):
        assert train["date"].iloc[tr].max() < train["date"].iloc[va].min()


def test_tuned_tree_beats_base_rate(net):
    from foodnet.risk.tree import compare_models, time_split
    train, test = time_split(net["collections"])
    r, _ = compare_models(train, test, dict(max_depth=4, min_samples_leaf=30))
    r = r.set_index("model")
    assert r.loc["decision_tree", "pr_auc"] > r.loc["always_collected", "pr_auc"]
    assert r.loc["decision_tree", "recall"] > 0
