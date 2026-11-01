"""Synthetic data generator for the food-redistribution network.

No public dataset records surplus donations at item level, so this module
builds a realistic stand-in:

* donors (supermarkets, wholesalers, cafes, restaurants, bakeries) with daily
  sales, stock and resulting surplus per product category
* recipients (food banks, community kitchens, shelters, pantries) with daily need
* volunteers with a latent reliability
* a history of past collections with outcomes, for the risk predictor

The surplus mechanism is deliberately simple and explainable: each donor stocks
to last week's sales for the same weekday plus a safety buffer, and surplus is
whatever is left after today's demand. Demand responds to weekday, season,
weather, bank holidays and promotions, so surplus is partly predictable.

Swap ``load_sales_history`` for a Favorita or M5 loader to train on real sales.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from foodnet.config import (
    CATEGORIES, DONOR_TYPES, LAT_RANGE, LON_RANGE, RECIPIENT_TYPES, NetworkConfig,
)


# --------------------------------------------------------------------------- #
# Calendar and weather
# --------------------------------------------------------------------------- #
def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _first_monday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def _last_monday(year: int, month: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=d.weekday())


def uk_bank_holidays(years) -> set[date]:
    """England and Wales bank holidays (substitute days ignored for simplicity)."""
    out = set()
    for y in years:
        e = _easter(y)
        out |= {
            date(y, 1, 1), e - timedelta(days=2), e + timedelta(days=1),
            _first_monday(y, 5), _last_monday(y, 5), _last_monday(y, 8),
            date(y, 12, 25), date(y, 12, 26),
        }
    return out


def make_calendar(cfg: NetworkConfig, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    hols = uk_bank_holidays(range(dates[0].year, dates[-1].year + 1))
    doy = dates.dayofyear.to_numpy()
    # London-ish seasonal temperature with noise (deg C)
    temp = 11.5 - 7.0 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2.5, len(dates))
    rain_p = 0.45 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    rain = rng.random(len(dates)) < rain_p
    cal = pd.DataFrame({
        "date": dates,
        "dow": dates.dayofweek,
        "month": dates.month,
        "is_holiday": [d.date() in hols for d in dates],
        "temp_c": temp.round(1),
        "rain": rain,
    })
    cal["is_pre_holiday"] = cal["is_holiday"].shift(-1, fill_value=False)
    return cal


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
def _coords(rng, n):
    return (rng.uniform(*LAT_RANGE, n).round(5), rng.uniform(*LON_RANGE, n).round(5))


def _pick_types(rng, types: dict, n: int) -> list[str]:
    names = list(types)
    p = np.array([types[t]["share"] for t in names], dtype=float)
    return list(rng.choice(names, size=n, p=p / p.sum()))


def make_donors(cfg: NetworkConfig, rng) -> pd.DataFrame:
    types = _pick_types(rng, DONOR_TYPES, cfg.n_donors)
    lat, lon = _coords(rng, cfg.n_donors)
    return pd.DataFrame({
        "donor_id": [f"D{i:03d}" for i in range(cfg.n_donors)],
        "donor_type": types,
        "size": rng.lognormal(0, 0.35, cfg.n_donors).round(2),
        # how cautious the store is when ordering stock (safety buffer)
        "buffer": rng.uniform(1.08, 1.30, cfg.n_donors).round(3),
        "closing_hour": [int(rng.choice([17, 18, 20, 21, 22])) for _ in range(cfg.n_donors)],
        "lat": lat, "lon": lon,
    })


def make_recipients(cfg: NetworkConfig, rng) -> pd.DataFrame:
    types = _pick_types(rng, RECIPIENT_TYPES, cfg.n_recipients)
    lat, lon = _coords(rng, cfg.n_recipients)
    base = np.array([RECIPIENT_TYPES[t]["daily_need_kg"] for t in types]) * rng.lognormal(0, 0.3, cfg.n_recipients)
    chilled_share = np.array([RECIPIENT_TYPES[t]["chilled_share"] for t in types])
    return pd.DataFrame({
        "recipient_id": [f"R{i:03d}" for i in range(cfg.n_recipients)],
        "recipient_type": types,
        "base_need_kg": base.round(1),
        # storage: daily intake capacity is a bit above typical need; chilled is a subset
        "capacity_kg": (base * rng.uniform(1.1, 1.5, cfg.n_recipients)).round(1),
        "chilled_capacity_kg": (base * chilled_share * rng.uniform(0.8, 1.3, cfg.n_recipients)).round(1),
        "accepts_meat": rng.random(cfg.n_recipients) > 0.2,
        "accept_rate": rng.uniform(0.85, 0.99, cfg.n_recipients).round(3),
        "lat": lat, "lon": lon,
    })


def make_volunteers(cfg: NetworkConfig, rng) -> pd.DataFrame:
    lat, lon = _coords(rng, cfg.n_volunteers)
    return pd.DataFrame({
        "volunteer_id": [f"V{i:03d}" for i in range(cfg.n_volunteers)],
        "reliability": rng.beta(9, 1.5, cfg.n_volunteers).round(3),  # latent, not observed directly
        "lat": lat, "lon": lon,
    })


# --------------------------------------------------------------------------- #
# Sales, stock and surplus
# --------------------------------------------------------------------------- #
_DOW = {
    "supermarket": [0.90, 0.88, 0.92, 0.97, 1.10, 1.25, 1.00],
    "wholesaler":  [1.15, 1.05, 1.00, 1.05, 1.10, 0.70, 0.40],
    "cafe":        [1.10, 1.15, 1.15, 1.10, 1.00, 0.75, 0.60],
    "restaurant":  [0.70, 0.75, 0.85, 1.00, 1.30, 1.40, 1.05],
    "bakery":      [0.95, 0.95, 0.95, 1.00, 1.05, 1.25, 0.90],
}


def make_sales_history(cfg: NetworkConfig, donors: pd.DataFrame, cal: pd.DataFrame, rng) -> pd.DataFrame:
    """Daily units demanded, stocked and left over per donor x category."""
    frames = []
    n = len(cal)
    doy = cal["date"].dt.dayofyear.to_numpy()
    season = 1 + 0.08 * np.cos(2 * np.pi * (doy - 355) / 365.25)  # small December peak
    for d in donors.itertuples():
        dt = DONOR_TYPES[d.donor_type]
        dow_f = np.array(_DOW[d.donor_type])[cal["dow"].to_numpy()]
        for cat in dt["categories"]:
            mu = dt["scale"] * d.size * rng.uniform(0.5, 1.2)
            demand = mu * dow_f * season
            # weather: rain hits footfall at cafes/restaurants, heat lifts produce and cuts bakery
            if d.donor_type in ("cafe", "restaurant"):
                demand = demand * np.where(cal["rain"], 0.85, 1.0)
            if cat == "produce":
                demand = demand * (1 + 0.012 * (cal["temp_c"].to_numpy() - 12))
            if cat == "bakery":
                demand = demand * (1 - 0.006 * (cal["temp_c"].to_numpy() - 12))
            # holidays: offices shut (cafes quiet), shoppers stock up the day before
            if d.donor_type == "cafe":
                demand = demand * np.where(cal["is_holiday"], 0.45, 1.0)
            if d.donor_type == "supermarket":
                demand = demand * np.where(cal["is_pre_holiday"], 1.20, 1.0) * np.where(cal["is_holiday"], 0.85, 1.0)
            if d.donor_type == "wholesaler":
                demand = demand * np.where(cal["is_holiday"], 0.30, 1.0)
            promo = rng.random(n) < 0.05
            lift = np.where(promo, rng.uniform(1.05, 1.55, n), 1.0)
            demand = demand * lift
            sales_units = rng.poisson(np.clip(demand, 0.1, None)).astype(float)
            # stock = same weekday last week x buffer (x 1.3 if a promo is planned)
            ref = pd.Series(sales_units).shift(7).bfill().to_numpy()
            stock = np.round(ref * d.buffer * np.where(promo, 1.30, 1.0) + rng.normal(0, 1.5, n)).clip(0)
            sold = np.minimum(stock, sales_units)
            leftover = stock - sold
            # shelf-stable items are mostly kept for tomorrow; only damaged / short-dated stock is surplus
            if cat == "ambient":
                leftover = np.round(leftover * 0.15)
            frames.append(pd.DataFrame({
                "date": cal["date"], "donor_id": d.donor_id, "donor_type": d.donor_type,
                "category": cat, "promo": promo, "stock_units": stock,
                "demand_units": sales_units, "sold_units": sold,
                "surplus_units": leftover,
                "surplus_kg": (leftover * CATEGORIES[cat]["kg_per_unit"]).round(2),
            }))
    sales = pd.concat(frames, ignore_index=True)
    return sales.merge(cal, on="date", how="left")


def make_need_history(recipients: pd.DataFrame, cal: pd.DataFrame, rng) -> pd.DataFrame:
    doy = cal["date"].dt.dayofyear.to_numpy()
    winter = 1 + 0.18 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    dow = np.array([1.05, 1.0, 1.0, 1.0, 1.10, 0.85, 0.75])[cal["dow"].to_numpy()]
    trend = np.linspace(1.0, 1.08, len(cal))  # need creeping up over two years
    frames = []
    for r in recipients.itertuples():
        mu = r.base_need_kg * winter * dow * trend
        frames.append(pd.DataFrame({
            "date": cal["date"], "recipient_id": r.recipient_id,
            "need_kg": rng.gamma(25, mu / 25).round(1),
        }))
    return pd.concat(frames, ignore_index=True).merge(cal, on="date", how="left")


# --------------------------------------------------------------------------- #
# Collection outcomes (ground truth shared with the simulation)
# --------------------------------------------------------------------------- #
def failure_probability(lead_time_h, distance_km, reliability, rain, hour, listing_kg, accept_rate):
    """Ground-truth probability that a planned collection does not happen.

    Short notice, long trips, wet evenings, late hours, unreliable volunteers and
    choosy recipients all raise the chance of a failed collection.
    """
    z = (
        -2.6
        - 0.55 * np.log1p(np.asarray(lead_time_h, dtype=float))
        + 0.11 * np.asarray(distance_km, dtype=float)
        - 4.0 * (np.asarray(reliability, dtype=float) - 0.85)
        + 0.45 * np.asarray(rain, dtype=float)
        + 0.12 * np.clip(np.asarray(hour, dtype=float) - 18, 0, None)
        + 0.004 * np.asarray(listing_kg, dtype=float)
        - 6.0 * (np.asarray(accept_rate, dtype=float) - 0.92)
        + 0.5  # calibrates to roughly 1 in 4 short-notice and 1 in 9 pre-booked failures
    )
    return 1 / (1 + np.exp(-z))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def make_collection_history(cfg, donors, recipients, volunteers, cal, rng, n=20000) -> pd.DataFrame:
    """Past collections with outcomes, for training the risk predictor."""
    di = rng.integers(0, len(donors), n)
    ri = rng.integers(0, len(recipients), n)
    vi = rng.integers(0, len(volunteers), n)
    ci = rng.integers(0, len(cal), n)
    d, r, v, c = donors.iloc[di], recipients.iloc[ri], volunteers.iloc[vi], cal.iloc[ci]
    dist = haversine_km(d["lat"].to_numpy(), d["lon"].to_numpy(), r["lat"].to_numpy(), r["lon"].to_numpy())
    lead = np.where(rng.random(n) < 0.5, rng.uniform(0.5, 4, n), rng.uniform(12, 36, n))
    hour = d["closing_hour"].to_numpy() + rng.integers(0, 2, n)
    kg = rng.gamma(2.0, 12.0, n)
    rel = v["reliability"].to_numpy()
    # what we can observe: past completion rate, a noisy version of latent reliability
    past_rate = np.clip(rel + rng.normal(0, 0.04, n), 0, 1)
    p = failure_probability(lead, dist, rel, c["rain"].to_numpy(), hour, kg, r["accept_rate"].to_numpy())
    return pd.DataFrame({
        "date": c["date"].to_numpy(), "donor_id": d["donor_id"].to_numpy(),
        "recipient_id": r["recipient_id"].to_numpy(), "volunteer_id": v["volunteer_id"].to_numpy(),
        "lead_time_h": lead.round(2), "distance_km": dist.round(2),
        "volunteer_past_rate": past_rate.round(3), "rain": c["rain"].to_numpy(),
        "hour": hour, "dow": c["dow"].to_numpy(), "listing_kg": kg.round(1),
        "recipient_accept_rate": r["accept_rate"].to_numpy(),
        "failed": rng.random(n) < p,
    }).sort_values("date", ignore_index=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_network(cfg: NetworkConfig | None = None) -> dict[str, pd.DataFrame]:
    """Build every table the project needs from one seed."""
    cfg = cfg or NetworkConfig()
    rng = np.random.default_rng(cfg.seed)
    cal = make_calendar(cfg, rng)
    donors = make_donors(cfg, rng)
    recipients = make_recipients(cfg, rng)
    volunteers = make_volunteers(cfg, rng)
    sales = make_sales_history(cfg, donors, cal, rng)
    need = make_need_history(recipients, cal, rng)
    collections = make_collection_history(cfg, donors, recipients, volunteers, cal, rng)
    return {
        "config": pd.DataFrame([asdict(cfg)]),
        "calendar": cal, "donors": donors, "recipients": recipients,
        "volunteers": volunteers, "sales": sales, "need": need,
        "collections": collections,
    }


def save_network(tables: dict[str, pd.DataFrame], out_dir) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
