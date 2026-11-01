"""Shared constants for the synthetic food-redistribution network.

Everything here is an assumption used by the synthetic generator and the
simulation. Change values here (or pass overrides) to test sensitivity.
"""
from dataclasses import dataclass, field

# Product categories: shelf life after the donor's closing time (hours),
# storage requirement and whether the category counts as meat.
CATEGORIES = {
    "bakery":   {"shelf_life_h": 24,  "chilled": False, "meat": False, "kg_per_unit": 0.35},
    "produce":  {"shelf_life_h": 72,  "chilled": False, "meat": False, "kg_per_unit": 0.60},
    "dairy":    {"shelf_life_h": 72,  "chilled": True,  "meat": False, "kg_per_unit": 0.80},
    "prepared": {"shelf_life_h": 24,  "chilled": True,  "meat": False, "kg_per_unit": 0.40},
    "meat":     {"shelf_life_h": 48,  "chilled": True,  "meat": True,  "kg_per_unit": 0.50},
    "ambient":  {"shelf_life_h": 720, "chilled": False, "meat": False, "kg_per_unit": 0.70},
}

# Which categories each donor type sells, and a typical daily demand scale (units).
DONOR_TYPES = {
    "supermarket": {"categories": ["bakery", "produce", "dairy", "prepared", "meat", "ambient"], "scale": 120, "share": 0.30},
    "wholesaler":  {"categories": ["produce", "dairy", "meat", "ambient"], "scale": 300, "share": 0.10},
    "cafe":        {"categories": ["bakery", "prepared"], "scale": 40, "share": 0.30},
    "restaurant":  {"categories": ["prepared", "produce"], "scale": 35, "share": 0.20},
    "bakery":      {"categories": ["bakery"], "scale": 80, "share": 0.10},
}

RECIPIENT_TYPES = {
    "food_bank":         {"daily_need_kg": 180, "chilled_share": 0.25, "share": 0.30},
    "community_kitchen": {"daily_need_kg": 90,  "chilled_share": 0.60, "share": 0.30},
    "shelter":           {"daily_need_kg": 60,  "chilled_share": 0.40, "share": 0.20},
    "pantry":            {"daily_need_kg": 70,  "chilled_share": 0.20, "share": 0.20},
}

# Rough bounding box for a London-style network (lat, lon).
LAT_RANGE = (51.42, 51.56)
LON_RANGE = (-0.30, 0.02)


@dataclass
class NetworkConfig:
    n_donors: int = 30
    n_recipients: int = 20
    n_volunteers: int = 60
    start_date: str = "2024-01-01"
    n_days: int = 730
    seed: int = 7
    # Ops assumptions used by matching and simulation
    avg_speed_kmh: float = 18.0          # urban van / cargo bike average
    max_distance_km: float = 12.0        # hard cap on a single collection
    same_day_lead_h: float = 2.0         # lead time when surplus is only known at closing
    prebook_lead_h: float = 24.0         # lead time when a forecast lets us book a day ahead
    same_day_volunteer_share: float = 0.35  # share of volunteers free at short notice
    prebook_volunteer_share: float = 0.80   # share free when asked a day ahead
    trips_per_volunteer: int = 2
    extra: dict = field(default_factory=dict)
