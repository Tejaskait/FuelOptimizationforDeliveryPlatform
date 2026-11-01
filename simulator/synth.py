"""Synthetic city + ground-truth physics used by the simulator and for bootstrapping ML models."""
import numpy as np
import pandas as pd

from app.core.eta import RUSH_HOURS
from app.core.geo import offset_point
from app.core import h3_index

CENTER = (12.9716, 77.5946)


def ground_truth_eta(km, hour: int):
    """What reality does: slower than the naive model in rush hour, plus a fixed handoff time."""
    speed = 11.0 if hour in RUSH_HOURS else 20.0
    return np.asarray(km, dtype=float) / speed * 60.0 + 4.0


def random_point(center, radius_km, rng):
    t, d = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform()) * radius_km
    return offset_point(center[0], center[1], d * np.cos(t), d * np.sin(t))


def make_restaurants(center, radius_km, n, rng):
    return [random_point(center, radius_km * 0.6, rng) for _ in range(n)]


def eta_training_data(n=20000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    km = rng.uniform(0.3, 14, n)
    hour = rng.integers(8, 24, n)
    minutes = np.array([ground_truth_eta(k, h) for k, h in zip(km, hour)]) * rng.lognormal(0, 0.15, n)
    return pd.DataFrame({"km": km, "hour": hour, "minutes": minutes})


def order_log(n_days=21, per_day=900, center=CENTER, radius_km=6.0, seed=0) -> pd.DataFrame:
    """Historical order log with lunch/dinner peaks and restaurant hotspots."""
    rng = np.random.default_rng(seed)
    rests = make_restaurants(center, radius_km, 10, rng)
    weights = rng.dirichlet(np.ones(10) * 0.7)
    rows = []
    hour_p = np.array([0.5, 1, 4, 9, 6, 3, 2, 2, 3, 4, 8, 11, 7, 3, 1.5, 1])  # hours 8..23
    hour_p = hour_p / hour_p.sum()
    for d in range(n_days):
        dow = d % 7
        mult = 1.3 if dow >= 5 else 1.0
        for _ in range(int(per_day * mult)):
            r = rests[rng.choice(10, p=weights)]
            lat, lng = r[0] + rng.normal(0, 0.002), r[1] + rng.normal(0, 0.002)
            hour = 8 + rng.choice(16, p=hour_p)
            rows.append((lat, lng, hour, dow))
    return pd.DataFrame(rows, columns=["rlat", "rlng", "hour", "dow"])
