"""End-to-end simulation: how much surplus reaches recipients before it expires?

Strategies compared on the same synthetic test days:

* ``baseline``        surplus known only at closing, nearest-recipient matching,
                      short-notice volunteers, no risk checks
* ``optimiser_only``  same-day, but allocation by the LP matching engine
* ``forecast_prebook`` the evening before, the P50 forecast is matched and
                      volunteers are booked a day ahead; surplus above the
                      forecast is matched same-day
* ``full``            forecast_prebook + the risk predictor adds a backup
                      volunteer to high-risk collections

Collection outcomes are drawn from the generator's ground-truth failure model,
so every strategy faces the same rules of the world.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from foodnet.config import CATEGORIES, NetworkConfig
from foodnet.data.generator import failure_probability, haversine_km
from foodnet.matching.engine import MatchParams, greedy_nearest, optimise_allocation


@dataclass
class Strategy:
    name: str
    allocator: str = "greedy"         # "greedy" | "optimise"
    prebook_quantile: str | None = None  # None (same-day only) | "p10" | "p50" | "p90"
    use_risk: bool = False


STRATEGIES = [
    Strategy("baseline", "greedy"),
    Strategy("optimiser_only", "optimise"),
    Strategy("forecast_prebook", "optimise", "p50"),
    Strategy("full", "optimise", "p50", use_risk=True),
]
SENSITIVITY = [Strategy(f"full_{q}", "optimise", q, use_risk=True) for q in ("p10", "p30", "p70", "p90")]


@dataclass
class SimInputs:
    cfg: NetworkConfig
    donors: pd.DataFrame
    recipients: pd.DataFrame
    volunteers: pd.DataFrame
    calendar: pd.DataFrame
    actual: pd.DataFrame        # date, donor_id, category, surplus_kg
    forecast: pd.DataFrame      # date, donor_id, category, p10, p50, p90 (out-of-sample)
    need_actual: pd.DataFrame   # date, recipient_id, need_kg
    need_forecast: pd.DataFrame  # date, recipient_id, need_kg (forecast)
    risk_model: object = None
    match: MatchParams = field(default_factory=MatchParams)


# --------------------------------------------------------------------------- #
def _listings(rows: pd.DataFrame, kg_col: str, donors: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = rows[rows[kg_col] >= 1.0].merge(donors[["donor_id", "lat", "lon", "closing_hour"]], on="donor_id")
    return pd.DataFrame({
        "listing_id": df["donor_id"] + "|" + df["category"] + "|" + tag,
        "donor_id": df["donor_id"], "category": df["category"], "kg": df[kg_col].to_numpy(),
        "shelf_life_h": df["category"].map(lambda c: CATEGORIES[c]["shelf_life_h"]).to_numpy(),
        "lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy(),
        "closing_hour": df["closing_hour"].to_numpy(),
    })


def _allocate(kind, listings, recipients, p, max_trips=None):
    """The optimiser knows how many volunteer trips are free; the greedy baseline does not."""
    if kind == "optimise":
        return optimise_allocation(listings, recipients, p, max_trips=max_trips)
    return greedy_nearest(listings, recipients, p)


def _assign_volunteers(trips: pd.DataFrame, donors: pd.DataFrame, vols: pd.DataFrame, slots: dict) -> np.ndarray:
    """Greedy: biggest trips first, each to the nearest volunteer with a free slot. -1 = nobody free."""
    out = np.full(len(trips), -1)
    if trips.empty or not slots:
        return out
    d = trips[["donor_id"]].merge(donors[["donor_id", "lat", "lon"]], on="donor_id", how="left")
    vidx = np.array(sorted(slots))
    dist = haversine_km(d["lat"].to_numpy()[:, None], d["lon"].to_numpy()[:, None],
                        vols["lat"].to_numpy()[vidx][None, :], vols["lon"].to_numpy()[vidx][None, :])
    for i in np.argsort(-trips["kg"].to_numpy()):
        for j in np.argsort(dist[i]):
            v = vidx[j]
            if slots.get(v, 0) > 0:
                out[i] = v
                slots[v] -= 1
                break
    return out


def _run_trips(trips, lead_h, vol_idx, inp: SimInputs, day_cal, rng, use_risk, slots, past_rate):
    """Decide backups, draw outcomes. Returns trips with collected flag and backup info."""
    t = trips.copy()
    t["lead_time_h"] = lead_h
    t["volunteer"] = vol_idx
    if t.empty:
        t["p_fail"] = pd.Series(dtype=float)
        t["collected"] = pd.Series(dtype=bool)
        t["backup"] = pd.Series(dtype=bool)
        return t
    meta = t[["donor_id"]].merge(inp.donors[["donor_id", "closing_hour"]], on="donor_id", how="left")
    acc = t[["recipient_id"]].merge(inp.recipients[["recipient_id", "accept_rate"]], on="recipient_id", how="left")
    has_vol = vol_idx >= 0
    rel = np.where(has_vol, inp.volunteers["reliability"].to_numpy()[np.maximum(vol_idx, 0)], 0.0)
    feat = pd.DataFrame({
        "lead_time_h": lead_h, "distance_km": t["distance_km"].to_numpy(),
        "volunteer_past_rate": np.where(has_vol, past_rate[np.maximum(vol_idx, 0)], 0.0),
        "rain": bool(day_cal["rain"]), "hour": meta["closing_hour"].to_numpy(),
        "dow": int(day_cal["dow"]), "listing_kg": t["kg"].to_numpy(),
        "recipient_accept_rate": acc["accept_rate"].to_numpy(),
    })
    p_true = failure_probability(feat["lead_time_h"], feat["distance_km"], rel, feat["rain"],
                                 feat["hour"], feat["listing_kg"], feat["recipient_accept_rate"])
    p_true = np.where(has_vol, p_true, 1.0)
    backup = np.zeros(len(t), dtype=bool)
    if use_risk and inp.risk_model is not None:
        flag = has_vol & inp.risk_model.flag(feat)
        backup_vol = _assign_volunteers(t[flag], inp.donors, inp.volunteers, slots)
        idx = np.flatnonzero(flag)
        for k, v in zip(idx, backup_vol):
            if v >= 0:
                backup[k] = True
                rel_b = inp.volunteers["reliability"].to_numpy()[v]
                p_b = failure_probability(lead_h if np.isscalar(lead_h) else lead_h[k], feat["distance_km"].iat[k],
                                          rel_b, feat["rain"].iat[k], feat["hour"].iat[k],
                                          feat["listing_kg"].iat[k], feat["recipient_accept_rate"].iat[k])
                p_true[k] = p_true[k] * p_b
    t["p_fail"] = p_true
    t["backup"] = backup
    t["collected"] = rng.random(len(t)) >= p_true
    return t


# --------------------------------------------------------------------------- #
def simulate_day(day, strat: Strategy, inp: SimInputs, rng) -> dict:
    cfg, p = inp.cfg, inp.match
    day_cal = inp.calendar.set_index("date").loc[day]
    actual = inp.actual[inp.actual["date"] == day]
    need_f = inp.need_forecast[inp.need_forecast["date"] == day].set_index("recipient_id")["need_kg"]
    rec = inp.recipients.copy()
    rec["need_kg"] = rec["recipient_id"].map(need_f).fillna(rec["base_need_kg"]).to_numpy()

    n_v = len(inp.volunteers)
    past_rate = np.clip(inp.volunteers["reliability"].to_numpy() + rng.normal(0, 0.04, n_v), 0, 1)
    avail_pre = rng.random(n_v) < cfg.prebook_volunteer_share
    avail_same = rng.random(n_v) < cfg.same_day_volunteer_share
    all_trips = []

    planned_delivered = pd.Series(dtype=float)
    excess = actual[["donor_id", "category", "surplus_kg"]].rename(columns={"surplus_kg": "kg_left"})

    if strat.prebook_quantile:
        fc = inp.forecast[inp.forecast["date"] == day][["donor_id", "category", strat.prebook_quantile]]
        plan_l = _listings(fc, strat.prebook_quantile, inp.donors, "plan")
        slots_pre = {v: cfg.trips_per_volunteer for v in np.flatnonzero(avail_pre)}
        # keep some booked slots spare for risk backups when the risk predictor is on
        reserve = int(0.15 * sum(slots_pre.values())) if strat.use_risk else 0
        plan = _allocate(strat.allocator, plan_l, rec, p, max_trips=sum(slots_pre.values()) - reserve)
        vol = _assign_volunteers(plan, inp.donors, inp.volunteers, slots_pre)
        # actual surplus fills the plan first, pro rata across that listing's planned trips
        act = actual.set_index(["donor_id", "category"])["surplus_kg"]
        planned_tot = plan.groupby(["donor_id", "category"])["kg"].transform("sum")
        key = list(zip(plan["donor_id"], plan["category"]))
        avail_kg = np.array([act.get(k, 0.0) for k in key])
        ratio = np.minimum(avail_kg / planned_tot.clip(1e-9), 1.0)
        plan = plan.assign(planned_kg=plan["kg"], kg=(plan["kg"] * ratio).round(2))
        plan["empty"] = plan["kg"] < p.min_alloc_kg
        t = _run_trips(plan, cfg.prebook_lead_h, vol, inp, day_cal, rng, strat.use_risk, slots_pre, past_rate)
        t.loc[t["empty"], "collected"] = False
        t["stage"] = "prebooked"
        all_trips.append(t)
        used = t[~t["empty"]].groupby(["donor_id", "category"])["kg"].sum()
        excess = excess.assign(kg_left=excess["kg_left"] - [used.get(k, 0.0) for k in zip(excess["donor_id"], excess["category"])])
        planned_delivered = t[~t["empty"]].groupby("recipient_id")["kg"].sum()
        # volunteers booked a day ahead are not free again same-day
        avail_same &= ~np.isin(np.arange(n_v), t["volunteer"][t["volunteer"] >= 0].to_numpy())

    # same-day stage: whatever was not covered by a plan
    rec2 = rec.copy()
    got = rec2["recipient_id"].map(planned_delivered).fillna(0).to_numpy()
    rec2["capacity_kg"] = (rec2["capacity_kg"] - got).clip(0)
    rec2["need_kg"] = (rec2["need_kg"] - got).clip(0)
    # chilled room is reduced by chilled kg delivered on plan
    if all_trips:
        t0 = all_trips[0]
        ch = t0[(~t0["empty"]) & t0["category"].map(lambda c: CATEGORIES[c]["chilled"])].groupby("recipient_id")["kg"].sum()
        rec2["chilled_capacity_kg"] = (rec2["chilled_capacity_kg"] - rec2["recipient_id"].map(ch).fillna(0)).clip(0)
    same_l = _listings(excess, "kg_left", inp.donors, "same")
    slots_same = {v: cfg.trips_per_volunteer for v in np.flatnonzero(avail_same)}
    reserve = int(0.15 * sum(slots_same.values())) if strat.use_risk else 0
    same = _allocate(strat.allocator, same_l, rec2, p, max_trips=sum(slots_same.values()) - reserve)
    vol = _assign_volunteers(same, inp.donors, inp.volunteers, slots_same)
    same = same.assign(empty=False, planned_kg=same["kg"])
    t = _run_trips(same, cfg.same_day_lead_h, vol, inp, day_cal, rng, strat.use_risk, slots_same, past_rate)
    t["stage"] = "same_day"
    all_trips.append(t)

    trips = pd.concat([t for t in all_trips if len(t)], ignore_index=True) if any(len(t) for t in all_trips) \
        else all_trips[-1]
    trips["collected"] = trips["collected"].astype(bool)
    trips["empty"] = trips["empty"].astype(bool)
    trips["backup"] = trips["backup"].astype(bool)
    real = trips[~trips["empty"]]
    placed = real.loc[real["collected"], "kg"].sum()
    received = real[real["collected"]].groupby("recipient_id")["kg"].sum()
    return {
        "date": day, "strategy": strat.name,
        "surplus_kg": float(actual["surplus_kg"].sum()),
        "placed_kg": float(placed),
        "matched_kg": float(real["kg"].sum()),
        "trips": int(len(real)), "empty_trips": int(trips["empty"].sum()),
        "no_volunteer_trips": int((real["volunteer"] < 0).sum()),
        "failed_trips": int((~real["collected"]).sum()),
        "backups": int(real["backup"].sum()),
        "km": float((real["distance_km"]).sum()),
        "_received": received,
    }


def run_simulation(inp: SimInputs, strategies=STRATEGIES, days=None, seeds=(0, 1, 2)):
    """Run every strategy over the test days for several random seeds.

    Returns (daily results DataFrame, summary DataFrame).
    """
    days = days if days is not None else np.sort(inp.forecast["date"].unique())
    need = inp.need_actual[inp.need_actual["date"].isin(days)].groupby("recipient_id")["need_kg"].sum()
    rows, summaries = [], []
    for seed in seeds:
        for strat in strategies:
            got = pd.Series(0.0, index=need.index)
            for i, day in enumerate(days):
                # common random numbers: every strategy sees the same volunteer availability per day
                rng = np.random.default_rng([seed, i])
                r = simulate_day(day, strat, inp, rng)
                got = got.add(r.pop("_received"), fill_value=0)
                r["seed"] = seed
                rows.append(r)
            fill = (got.reindex(need.index).fillna(0) / need).clip(upper=1.0)
            summaries.append({"strategy": strat.name, "seed": seed,
                              "min_fill_rate": fill.min(), "fill_rate_spread": fill.max() - fill.min()})
    daily = pd.DataFrame(rows)
    agg = daily.groupby(["strategy", "seed"]).agg(
        surplus_kg=("surplus_kg", "sum"), placed_kg=("placed_kg", "sum"), trips=("trips", "sum"),
        empty_trips=("empty_trips", "sum"), failed_trips=("failed_trips", "sum"),
        no_volunteer_trips=("no_volunteer_trips", "sum"), backups=("backups", "sum"), km=("km", "sum"),
    ).reset_index().merge(pd.DataFrame(summaries), on=["strategy", "seed"])
    agg["share_placed"] = agg["placed_kg"] / agg["surplus_kg"]
    agg["failed_trip_rate"] = agg["failed_trips"] / agg["trips"].clip(lower=1)
    summary = agg.groupby("strategy").agg(["mean", "std"])
    order = [s.name for s in strategies]
    summary = summary.reindex([o for o in order if o in summary.index])
    return daily, summary
