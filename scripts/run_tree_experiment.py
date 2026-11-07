"""Decision-tree experiment for the uncollected-listing predictor (course module 9).

    python scripts/run_tree_experiment.py            # full network (about 15 seconds)
    python scripts/run_tree_experiment.py --quick    # smaller network

Steps
1. Generate the synthetic network and split collections by time (latest 20% = test)
2. Sweep max_depth 2-12 with time-ordered CV: train vs validation PR-AUC and F1
3. Grid search max_depth, min_samples_leaf (20-50), min_impurity_decrease
4. Compare the tuned tree with always-collected, logistic, boosting and a random forest
5. Permutation importance vs the tree's own importance
6. Stability of the top splits across bootstrap resamples
7. Write tables, the depth chart and a readable depth-3 rule set to results/
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foodnet.config import NetworkConfig  # noqa: E402
from foodnet.data.generator import generate_network  # noqa: E402
from foodnet.risk.predictor import _X  # noqa: E402
from foodnet.risk.tree import (compare_models, depth_sweep, grid_search, importance_table, make_tree,  # noqa: E402
                               stability, time_split, tree_rules)

warnings.filterwarnings("ignore")


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main(quick: bool = False, out_dir: Path = ROOT / "results"):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = NetworkConfig(n_donors=15, n_recipients=10, n_volunteers=30, n_days=400) if quick else NetworkConfig()
    T = generate_network(cfg)
    train, test = time_split(T["collections"])
    log(f"collections: {len(train):,} train, {len(test):,} test, failure rate {train['failed'].mean():.1%}", t0)

    # 2. depth sweep
    sweep = depth_sweep(train)
    sweep.to_csv(out_dir / "tree_depth_sweep.csv", index=False)
    log("depth sweep (mean over 4 time-ordered folds):\n" + sweep.round(3).to_string(index=False), t0)

    # 3. grid search
    best, grid = grid_search(train)
    grid.to_csv(out_dir / "tree_grid_search.csv", index=False)
    log(f"best params: {best}\n" + grid.head(10).round(4).to_string(index=False), t0)

    # 4. comparison on the held-out latest 20%
    comp, fitted = compare_models(train, test, best)
    comp.to_csv(out_dir / "tree_model_comparison.csv", index=False)
    log("model comparison (test):\n" + comp[["model", "pr_auc", "precision", "recall", "f1", "accuracy",
        "tn", "fp", "fn", "tp"]].round(3).to_string(index=False), t0)

    # 5. importance
    imp = importance_table(fitted["decision_tree"], test)
    imp.to_csv(out_dir / "tree_permutation_importance.csv", index=False)
    log("importance:\n" + imp.round(4).to_string(index=False), t0)

    # 6. stability
    stab = stability(train, best)
    stab.to_csv(out_dir / "tree_stability.csv", index=False)
    log("root split across 20 resamples:\n" + stab["root"].value_counts().to_string() +
        "\nsecond-level splits:\n" + (stab["left"] + " / " + stab["right"]).value_counts().to_string(), t0)

    # 7. a depth-3 tree a coordinator could read, plus the chart
    small = make_tree(max_depth=3, min_samples_leaf=50).fit(_X(train), train["failed"].astype(int))
    (out_dir / "tree_rules.txt").write_text(
        "Depth-3 decision tree for failed collections (class-weighted; 'class: 1' = likely to fail)\n\n"
        + tree_rules(small, _X(train).columns))
    _chart(sweep, best, out_dir)

    c = comp.set_index("model")
    summary = {
        "best_tree_params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in best.items()},
        "test_pr_auc": {m: float(c.loc[m, "pr_auc"]) for m in c.index},
        "test_f1": {m: float(c.loc[m, "f1"]) for m in c.index},
        "top_permutation_feature": str(imp.iloc[0]["feature"]),
        "top_impurity_feature": str(imp.sort_values("impurity_importance").iloc[-1]["feature"]),
        "root_split_agreement": float(stab["root"].value_counts(normalize=True).iloc[0]),
    }
    (out_dir / "tree_summary.json").write_text(json.dumps(summary, indent=2))
    log("summary:\n" + json.dumps(summary, indent=2), t0)
    return summary


def _chart(sweep, best, out_dir):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(sweep["max_depth"], sweep["train_pr_auc"], marker="o", color="#999", label="Training")
    ax.plot(sweep["max_depth"], sweep["val_pr_auc"], marker="o", color="#2a6fdb", label="Validation (later weeks)")
    ax.axvline(best["max_depth"], color="#d9534f", ls="--", lw=1, label=f"Chosen depth ({best['max_depth']})")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Deeper trees memorise training weeks", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "tree_depth_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick)
