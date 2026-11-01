"""Run the full capstone pipeline and write results to ./results.

    python scripts/run_all.py            # full run (about 3-6 minutes)
    python scripts/run_all.py --quick    # smaller network, fewer days, one seed

Steps
1. Generate the synthetic network (data/ CSVs)
2. Backtest the surplus forecaster against seasonal naive
3. Backtest the need forecaster against the 4-week same-weekday mean
4. Evaluate the uncollected-listing predictor on a time split
5. Simulate the strategies end to end on the forecast test days
6. Write metrics, tables and charts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foodnet.config import NetworkConfig  # noqa: E402
from foodnet.data.generator import generate_network, save_network  # noqa: E402
from foodnet.forecasting import need as need_fc  # noqa: E402
from foodnet.forecasting.surplus import build_features, rolling_backtest  # noqa: E402
from foodnet.matching.engine import MatchParams  # noqa: E402
from foodnet.risk.predictor import RiskPredictor, time_split_eval  # noqa: E402
from foodnet.simulation.simulate import SENSITIVITY, STRATEGIES, SimInputs, run_simulation  # noqa: E402

warnings.filterwarnings("ignore")


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main(quick: bool = False, out_dir: Path = ROOT / "results", data_dir: Path = ROOT / "data"):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = NetworkConfig(n_donors=15, n_recipients=10, n_volunteers=30, n_days=400) if quick else NetworkConfig()
    folds, fold_days, seeds = (2, 14, (0,)) if quick else (4, 21, (0, 1, 2))

    # 1. data
    T = generate_network(cfg)
    save_network(T, data_dir)
    log(f"generated network: {len(T['donors'])} donors, {len(T['recipients'])} recipients, "
        f"{len(T['sales']):,} donor-category-days", t0)

    # 2. surplus forecaster
    feat = build_features(T["sales"])
    fc_metrics, fc_preds = rolling_backtest(feat, n_folds=folds, fold_days=fold_days)
    fc_metrics.to_csv(out_dir / "surplus_forecast_backtest.csv", index=False)
    fc_summary = fc_metrics.groupby("model")[["mae", "wape", "pinball_p10", "pinball_p50",
                                              "pinball_p90", "coverage_p10_p90"]].mean()
    log("surplus forecaster backtest:\n" + fc_summary.round(3).to_string(), t0)

    # 3. need forecaster
    nfeat = need_fc.build_features(T["need"])
    need_metrics = need_fc.backtest(nfeat, test_days=folds * fold_days)
    need_metrics.to_csv(out_dir / "need_forecast_backtest.csv", index=False)
    log("need forecaster backtest:\n" + need_metrics.round(3).to_string(index=False), t0)

    # 4. risk predictor
    risk_metrics = time_split_eval(T["collections"])
    risk_metrics.to_csv(out_dir / "risk_predictor_eval.csv", index=False)
    log("risk predictor (time split):\n" + risk_metrics[["model", "pr_auc", "roc_auc", "brier",
        "precision_at_threshold", "recall_at_threshold"]].round(3).to_string(index=False), t0)
    best_kind = risk_metrics[risk_metrics["model"] != "always_collected"].sort_values("pr_auc").iloc[-1]["model"]
    test_start = fc_preds["date"].min()
    risk_model = RiskPredictor(best_kind).fit(T["collections"][T["collections"]["date"] < test_start])

    # 5. simulation on the forecast test days (out-of-sample forecasts only)
    gbm = fc_preds[fc_preds["model"] == "gbm_quantile"].copy()
    # P30 / P70 for the sensitivity runs: linear blend between the fitted quantiles
    gbm["p30"] = gbm["p10"] + 0.5 * (gbm["p50"] - gbm["p10"])
    gbm["p70"] = gbm["p50"] + 0.5 * (gbm["p90"] - gbm["p50"])
    need_test = nfeat[nfeat["date"] >= test_start]
    inp = SimInputs(
        cfg=cfg, donors=T["donors"], recipients=T["recipients"], volunteers=T["volunteers"],
        calendar=T["calendar"],
        actual=T["sales"][T["sales"]["date"] >= test_start][["date", "donor_id", "category", "surplus_kg"]],
        forecast=gbm[["date", "donor_id", "category", "p10", "p30", "p50", "p70", "p90"]],
        need_actual=T["need"][["date", "recipient_id", "need_kg"]],
        need_forecast=need_test[["date", "recipient_id", "same_dow_4wk"]].rename(columns={"same_dow_4wk": "need_kg"}),
        risk_model=risk_model,
        match=MatchParams(max_distance_km=cfg.max_distance_km, speed_kmh=cfg.avg_speed_kmh),
    )
    daily, summary = run_simulation(inp, STRATEGIES, seeds=seeds)
    daily.to_csv(out_dir / "simulation_daily.csv", index=False)
    cols = ["share_placed", "placed_kg", "failed_trip_rate", "empty_trips", "backups", "km",
            "min_fill_rate", "fill_rate_spread"]
    sim = summary[[(c, "mean") for c in cols]]
    sim.columns = cols
    sim.to_csv(out_dir / "simulation_summary.csv")
    log("simulation summary (mean over seeds):\n" + sim.round(3).to_string(), t0)

    _, sens = run_simulation(inp, [s for s in STRATEGIES if s.name == "full"] + SENSITIVITY, seeds=seeds[:1])
    sens = sens[[(c, "mean") for c in cols]]
    sens.columns = cols
    sens.to_csv(out_dir / "prebook_quantile_sensitivity.csv")
    log("pre-book quantile sensitivity:\n" + sens.round(3).to_string(), t0)

    # 6. charts + headline JSON
    _charts(fc_preds, sim, sens, out_dir)
    base, full = sim.loc["baseline"], sim.loc["full"]
    headline = {
        "surplus_forecast_mae_reduction_vs_naive": float(1 - fc_summary.loc["gbm_quantile", "mae"] / fc_summary.loc["seasonal_naive", "mae"]),
        "surplus_forecast_p10_p90_coverage": float(fc_summary.loc["gbm_quantile", "coverage_p10_p90"]),
        "risk_model_used": str(best_kind),
        "risk_pr_auc": float(risk_metrics.set_index("model").loc[best_kind, "pr_auc"]),
        "share_placed_baseline": float(base["share_placed"]),
        "share_placed_full": float(full["share_placed"]),
        "extra_kg_placed_full_vs_baseline": float(full["placed_kg"] - base["placed_kg"]),
        "min_fill_rate_baseline": float(base["min_fill_rate"]),
        "min_fill_rate_full": float(full["min_fill_rate"]),
        "test_days": int(daily["date"].nunique()),
        "seeds": list(seeds),
    }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))
    log("headline:\n" + json.dumps(headline, indent=2), t0)
    return headline


def _charts(fc_preds, sim, sens, out_dir):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    # forecast example: busiest donor-category in the test window
    g = fc_preds[fc_preds["model"] == "gbm_quantile"]
    key = g.groupby(["donor_id", "category"])["surplus_kg"].sum().idxmax()
    ex = g[(g["donor_id"] == key[0]) & (g["category"] == key[1])].sort_values("date")
    naive = fc_preds[(fc_preds["model"] == "seasonal_naive") & (fc_preds["donor_id"] == key[0])
                     & (fc_preds["category"] == key[1])].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(ex["date"], ex["p10"], ex["p90"], alpha=0.2, color="#2a6fdb", label="P10-P90 forecast")
    ax.plot(ex["date"], ex["p50"], color="#2a6fdb", lw=1.6, label="P50 forecast")
    ax.plot(naive["date"], naive["p50"], color="#999", lw=1, ls="--", label="Seasonal naive")
    ax.plot(ex["date"], ex["surplus_kg"], color="#111", lw=1.2, marker="o", ms=2.5, label="Actual")
    ax.set_title(f"Surplus forecast vs actual: {key[0]} {key[1]} (out of sample)", loc="left", fontsize=11)
    ax.set_ylabel("Surplus (kg)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_example.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = {"baseline": "Baseline", "optimiser_only": "Optimiser only",
              "forecast_prebook": "+ Forecast pre-book", "full": "+ Risk backups (full)"}
    names = [labels.get(i, i) for i in sim.index]
    axes[0].barh(names, sim["share_placed"] * 100, color=["#bbb", "#8fb3ee", "#5b8fe0", "#2a6fdb"][: len(sim)])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Surplus placed before expiry (%)")
    axes[0].set_title("Share of surplus that reaches a recipient", loc="left", fontsize=11)
    for i, v in enumerate(sim["share_placed"] * 100):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].barh(names, sim["min_fill_rate"] * 100, color="#2a6fdb")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Worst-served recipient: % of need met")
    axes[1].set_title("Fairness", loc="left", fontsize=11)
    for i, v in enumerate(sim["min_fill_rate"] * 100):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_comparison.png")
    plt.close(fig)

    order = ["full_p10", "full_p30", "full", "full_p70", "full_p90"]
    s = sens.reindex([o for o in order if o in sens.index])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(["P10", "P30", "P50", "P70", "P90"][: len(s)], s["share_placed"] * 100, marker="o", color="#2a6fdb")
    ax.set_ylabel("Surplus placed (%)")
    ax.set_xlabel("Forecast quantile used to pre-book")
    ax.set_title("How much of the forecast to pre-book", loc="left", fontsize=11)
    ax2 = ax.twinx()
    ax2.bar(["P10", "P30", "P50", "P70", "P90"][: len(s)], s["empty_trips"], alpha=0.18, color="#d9534f")
    ax2.set_ylabel("Empty pre-booked trips", color="#d9534f")
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "prebook_sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick)
