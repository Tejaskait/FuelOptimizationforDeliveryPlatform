"""Batch dispatch: global min-cost matching of idle riders to pending orders.

cost(r,o) = w.time * (ETA - 0.3*order_wait) + w.pay * payout/10 + w.fair * (rider_earnings - fleet_mean)/10

* time term     -> platform wants fast delivery
* pay term      -> platform wants cheap delivery
* fairness term -> riders who already earned more than average get de-prioritised, so short/cheap
                   jobs and long/rich jobs get spread across the fleet instead of piling on one rider
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import h3_index
from .eta import default_eta
from .geo import ROAD_FACTOR, haversine_km
from .payout import PayoutConfig, payout
from .types import OrderState, RiderState

BIG_M = 1e6


@dataclass
class Weights:
    time: float = 1.0
    pay: float = 0.5
    fair: float = 1.0


@dataclass
class Assignment:
    rider_id: str
    order_id: int
    pickup_km: float
    drop_km: float
    eta_min: float
    payout: float


def assign(
    riders: list[RiderState],
    orders: list[OrderState],
    w: Weights,
    eta_fn=default_eta,
    pay_cfg: PayoutConfig = PayoutConfig(),
    nearest: bool = False,
    hour: int = 12,
    fleet_mean: float | None = None,
    max_ring: int | None = None,
) -> list[Assignment]:
    if not riders or not orders:
        return []
    rl = np.array([[r.lat, r.lng] for r in riders])
    ol = np.array([[o.rlat, o.rlng, o.clat, o.clng] for o in orders])

    d1 = haversine_km(rl[:, [0]], rl[:, [1]], ol[None, :, 0], ol[None, :, 1]) * ROAD_FACTOR  # R x O pickup
    d2 = haversine_km(ol[:, 0], ol[:, 1], ol[:, 2], ol[:, 3]) * ROAD_FACTOR                 # O drop
    total = d1 + d2[None, :]
    eta = np.asarray(eta_fn(total, hour), dtype=float)
    pay = payout(total, pay_cfg)

    earn = np.array([r.earnings for r in riders])[:, None]
    mean = float(earn.mean()) if fleet_mean is None else fleet_mean
    wait = np.array([o.wait_min for o in orders])[None, :]

    if nearest:
        cost = d1.copy()
    else:
        cost = w.time * (eta - 0.3 * wait) + w.pay * pay / 10.0 + w.fair * (earn - mean) / 10.0

    feasible = np.ones_like(cost, dtype=bool)
    if max_ring is not None:  # H3 spatial pre-filter: ignore riders more than K hexes from the restaurant
        rc = [h3_index.to_cell(r.lat, r.lng) for r in riders]
        oc = [h3_index.to_cell(o.rlat, o.rlng) for o in orders]
        for i, a in enumerate(rc):
            for j, b in enumerate(oc):
                feasible[i, j] = h3_index.ring_distance(a, b) <= max_ring
    cost = np.where(feasible, cost, cost + BIG_M)

    ri, oi = linear_sum_assignment(cost)
    return [
        Assignment(riders[i].id, orders[j].id, float(d1[i, j]), float(d2[j]), float(eta[i, j]), float(pay[i, j]))
        for i, j in zip(ri, oi)
        if feasible[i, j]
    ]
