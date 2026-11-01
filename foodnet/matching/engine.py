"""Matching engine: allocate surplus listings to recipients.

Two allocators share the same safety rules:

* ``greedy_nearest`` (baseline): each listing goes to the nearest compatible
  recipient with spare capacity, spilling to the next nearest.
* ``optimise_allocation``: a mixed-integer programme (scipy.optimize.milp /
  HiGHS) that places as much food as possible within the volunteer trips
  available, prefers short trips and urgent items, and lifts the worst-served
  recipient through a max-min fairness term.

Food-safety rules are hard constraints, never soft scores:
chilled food only goes to recipients with chilled storage, meat only to
recipients that accept it, and a trip must leave enough shelf life on arrival.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from foodnet.config import CATEGORIES
from foodnet.data.generator import haversine_km


@dataclass
class MatchParams:
    max_distance_km: float = 12.0
    speed_kmh: float = 18.0
    min_life_on_arrival_h: float = 8.0
    need_headroom: float = 1.15     # don't send a recipient more than 115% of its expected need
    distance_cost_per_km: float = 0.03  # value lost per kg per km travelled
    urgency_bonus: float = 0.20      # extra value per kg for items with <= 24h shelf life
    fairness_weight: float = 0.30    # reward for raising the worst-served recipient's fill rate
    min_alloc_kg: float = 2.0        # drop tiny fragments a volunteer wouldn't make a trip for
    trip_cost_kg: float = 4.0        # fixed cost of one collection trip, in kg-equivalent value
    mip_gap: float = 0.01


def feasibility(listings: pd.DataFrame, recipients: pd.DataFrame, p: MatchParams):
    """Return (distance matrix, boolean feasibility matrix) of shape (L, R)."""
    dist = haversine_km(listings["lat"].to_numpy()[:, None], listings["lon"].to_numpy()[:, None],
                        recipients["lat"].to_numpy()[None, :], recipients["lon"].to_numpy()[None, :])
    chilled = listings["category"].map(lambda c: CATEGORIES[c]["chilled"]).to_numpy()[:, None]
    meat = listings["category"].map(lambda c: CATEGORIES[c]["meat"]).to_numpy()[:, None]
    life = listings["shelf_life_h"].to_numpy()[:, None]
    ok = dist <= p.max_distance_km
    ok &= ~chilled | (recipients["chilled_capacity_kg"].to_numpy()[None, :] > 0)
    ok &= ~meat | recipients["accepts_meat"].to_numpy()[None, :]
    ok &= (life - dist / p.speed_kmh) >= p.min_life_on_arrival_h
    return dist, ok


def _limits(recipients: pd.DataFrame, p: MatchParams):
    total = np.minimum(recipients["capacity_kg"].to_numpy(),
                       recipients["need_kg"].to_numpy() * p.need_headroom).clip(0)
    chilled = recipients["chilled_capacity_kg"].to_numpy().clip(0)
    return total, chilled


def _to_frame(listings, recipients, alloc, dist, p):
    li, ri = np.nonzero(alloc >= p.min_alloc_kg)
    return pd.DataFrame({
        "listing_id": listings["listing_id"].to_numpy()[li],
        "donor_id": listings["donor_id"].to_numpy()[li],
        "category": listings["category"].to_numpy()[li],
        "recipient_id": recipients["recipient_id"].to_numpy()[ri],
        "kg": alloc[li, ri].round(2),
        "distance_km": dist[li, ri].round(2),
    })


def greedy_nearest(listings: pd.DataFrame, recipients: pd.DataFrame, p: MatchParams | None = None) -> pd.DataFrame:
    p = p or MatchParams()
    if listings.empty:
        return _to_frame(listings, recipients, np.zeros((0, len(recipients))), np.zeros((0, len(recipients))), p)
    dist, ok = feasibility(listings, recipients, p)
    total, chilled_cap = _limits(recipients, p)
    total, chilled_cap = total.copy(), chilled_cap.copy()
    is_chilled = listings["category"].map(lambda c: CATEGORIES[c]["chilled"]).to_numpy()
    alloc = np.zeros(dist.shape)
    for i, kg in enumerate(listings["kg"].to_numpy()):
        left = kg
        for j in np.argsort(dist[i]):
            if left <= 0:
                break
            if not ok[i, j]:
                continue
            room = min(total[j], chilled_cap[j]) if is_chilled[i] else total[j]
            take = min(left, room)
            if take <= 0:
                continue
            alloc[i, j] += take
            total[j] -= take
            if is_chilled[i]:
                chilled_cap[j] -= take
            left -= take
    return _to_frame(listings, recipients, alloc, dist, p)


def optimise_allocation(listings: pd.DataFrame, recipients: pd.DataFrame, p: MatchParams | None = None,
                        max_trips: int | None = None, exact: bool = False,
                        time_limit_s: float = 10.0) -> pd.DataFrame:
    """Solve the allocation as a mixed-integer programme (scipy.optimize.milp / HiGHS).

    Variables per feasible (listing, recipient) pair: x = kg sent, y = 1 if a trip
    is made. Plus z = the minimum fill rate across recipients (max-min fairness).

    maximise   sum value_lr * x_lr  -  trip_cost * sum y_lr  +  fairness * z * total_need
    subject to sum_r x_lr <= kg_l                       (can't send more than exists)
               sum_l x_lr <= min(capacity_r, headroom * need_r)
               sum_l chilled x_lr <= chilled_capacity_r
               x_lr <= kg_l * y_lr                      (food only moves on a trip)
               sum y_lr <= max_trips                    (volunteer slots available)
               z * need_r <= sum_l x_lr                 (z is the worst fill rate)

    ``exact=False`` (default) uses LP-and-round: solve the LP relaxation, keep the
    ``max_trips`` largest flows as trips, then re-solve the LP on those pairs only.
    It runs in well under a second and is usually within a few percent of the
    exact MILP, which can take 10 s or more per day.
    """
    p = p or MatchParams()
    L, R = len(listings), len(recipients)
    if L == 0:
        return _to_frame(listings, recipients, np.zeros((0, R)), np.zeros((0, R)), p)
    dist, ok = feasibility(listings, recipients, p)
    pairs = np.argwhere(ok)
    if len(pairs) == 0 or max_trips == 0:
        return _to_frame(listings, recipients, np.zeros((L, R)), dist, p)
    total, chilled_cap = _limits(recipients, p)
    need = recipients["need_kg"].to_numpy().clip(1e-6)
    kg = listings["kg"].to_numpy()
    life = listings["shelf_life_h"].to_numpy()
    is_chilled = listings["category"].map(lambda c: CATEGORIES[c]["chilled"]).to_numpy()

    P = len(pairs)
    n = 2 * P + 1  # x (P), y (P), z
    li, ri = pairs[:, 0], pairs[:, 1]
    value = 1.0 + p.urgency_bonus * (life[li] <= 24) - p.distance_cost_per_km * dist[li, ri]
    c = np.concatenate([-value, np.full(P, p.trip_cost_kg), [-p.fairness_weight * need.sum()]])

    n_rows = L + 3 * R + P + 1
    A = lil_matrix((n_rows, n))
    for k, (i, j) in enumerate(pairs):
        A[i, k] = 1
        A[L + j, k] = 1
        if is_chilled[i]:
            A[L + R + j, k] = 1
        A[L + 2 * R + j, k] = -1
        A[L + 3 * R + k, k] = 1          # x_k - kg_i * y_k <= 0
        A[L + 3 * R + k, P + k] = -kg[i]
        A[n_rows - 1, P + k] = 1         # total trips
    for j in range(R):
        A[L + 2 * R + j, n - 1] = need[j]
    trips_cap = np.inf if max_trips is None else float(max_trips)
    ub = np.concatenate([kg, total, chilled_cap, np.zeros(R), np.zeros(P), [trips_cap]])
    lb = np.full(n_rows, -np.inf)
    bounds = Bounds(np.zeros(n), np.concatenate([kg[li], np.ones(P), [1.0]]))
    cons = LinearConstraint(A.tocsr(), lb, ub)
    alloc = np.zeros((L, R))
    if exact:
        integrality = np.concatenate([np.zeros(P), np.ones(P), [0]])
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit_s, "disp": False, "mip_rel_gap": p.mip_gap})
        if res.x is not None:
            alloc[li, ri] = res.x[:P]
        return _to_frame(listings, recipients, alloc, dist, p)

    # LP-and-round: relaxation -> choose trips -> LP restricted to chosen trips
    res = milp(c, constraints=cons, bounds=bounds, options={"disp": False})
    if res.x is None:
        return _to_frame(listings, recipients, alloc, dist, p)
    flow = res.x[:P]
    keep = np.flatnonzero(flow >= p.min_alloc_kg)
    keep = keep[np.argsort(-flow[keep])]
    if max_trips is not None:
        keep = keep[: int(max_trips)]
    if len(keep) == 0:
        return _to_frame(listings, recipients, alloc, dist, p)
    sub = np.zeros((L, R), dtype=bool)
    sub[li[keep], ri[keep]] = True
    alloc = _solve_lp(listings, recipients, p, dist, sub, kg, total, chilled_cap, need, life, is_chilled)
    return _to_frame(listings, recipients, alloc, dist, p)


def _solve_lp(listings, recipients, p, dist, ok, kg, total, chilled_cap, need, life, is_chilled):
    """Continuous allocation over a fixed set of (listing, recipient) trips."""
    L, R = ok.shape
    pairs = np.argwhere(ok)
    P = len(pairs)
    li, ri = pairs[:, 0], pairs[:, 1]
    value = 1.0 + p.urgency_bonus * (life[li] <= 24) - p.distance_cost_per_km * dist[li, ri]
    c = np.concatenate([-value, [-p.fairness_weight * need.sum()]])
    A = lil_matrix((L + 3 * R, P + 1))
    for k, (i, j) in enumerate(pairs):
        A[i, k] = 1
        A[L + j, k] = 1
        if is_chilled[i]:
            A[L + R + j, k] = 1
        A[L + 2 * R + j, k] = -1
    for j in range(R):
        A[L + 2 * R + j, P] = need[j]
    ub = np.concatenate([kg, total, chilled_cap, np.zeros(R)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), np.full(L + 3 * R, -np.inf), ub),
               bounds=Bounds(np.zeros(P + 1), np.concatenate([kg[li], [1.0]])), options={"disp": False})
    alloc = np.zeros((L, R))
    if res.x is not None:
        alloc[li, ri] = res.x[:P]
    return alloc


def allocation_summary(alloc: pd.DataFrame, listings: pd.DataFrame, recipients: pd.DataFrame) -> dict:
    placed = alloc["kg"].sum()
    got = alloc.groupby("recipient_id")["kg"].sum().reindex(recipients["recipient_id"]).fillna(0).to_numpy()
    fill = np.clip(got / recipients["need_kg"].to_numpy().clip(1e-6), 0, None)
    return {
        "kg_offered": float(listings["kg"].sum()),
        "kg_placed": float(placed),
        "share_placed": float(placed / max(listings["kg"].sum(), 1e-9)),
        "km_per_100kg": float(100 * (alloc["kg"] * alloc["distance_km"]).sum() / max(placed, 1e-9)),
        "min_fill_rate": float(fill.min()) if len(fill) else 0.0,
        "fill_rate_spread": float(fill.max() - fill.min()) if len(fill) else 0.0,
        "trips": int(len(alloc)),
    }
