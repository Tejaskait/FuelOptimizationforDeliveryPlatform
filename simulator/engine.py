"""Discrete-time dispatch simulator: same assign() + payout() as production, synthetic city."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.assignment import Weights, assign
from app.core.eta import default_eta
from app.core.metrics import gini
from app.core.payout import PayoutConfig, payout
from app.core.types import OrderState, RiderState

from .synth import CENTER, ground_truth_eta, make_restaurants, random_point


@dataclass
class SimConfig:
    center: tuple = CENTER
    radius_km: float = 5.0
    n_riders: int = 40
    orders_per_min: float = 0.5
    hours: float = 4.0
    start_hour: int = 18
    n_rest: int = 8
    tick_s: int = 30


def run_sim(cfg: SimConfig, w: Weights, nearest=False, eta_fn=default_eta, seed=1, pay_cfg=PayoutConfig()) -> dict:
    rng = np.random.default_rng(seed)
    rests = make_restaurants(cfg.center, cfg.radius_km, cfg.n_rest, rng)
    riders = [RiderState(str(i), *random_point(cfg.center, cfg.radius_km, rng)) for i in range(cfg.n_riders)]
    pending: list[OrderState] = []
    rows, oid = [], 0

    for t in range(int(cfg.hours * 3600 / cfg.tick_s)):
        now = t * cfg.tick_s / 60
        hour = int(cfg.start_hour + now / 60) % 24
        for _ in range(rng.poisson(cfg.orders_per_min * cfg.tick_s / 60)):
            r = rests[rng.integers(len(rests))]
            c = random_point(cfg.center, cfg.radius_km, rng)
            pending.append(OrderState(oid, r[0], r[1], c[0], c[1], created=now))
            oid += 1
        idle = [r for r in riders if r.free_at <= now]
        if not idle or not pending:
            continue
        for o in pending:
            o.wait_min = now - o.created
        mean = float(np.mean([r.earnings for r in riders]))
        res = assign(idle, pending, w, eta_fn, pay_cfg, nearest, hour, fleet_mean=mean)
        by_r = {r.id: r for r in idle}
        by_o = {o.id: o for o in pending}
        served = set()
        for a in res:
            rider, o = by_r[a.rider_id], by_o[a.order_id]
            km = a.pickup_km + a.drop_km
            actual = float(ground_truth_eta(km, hour)) * rng.lognormal(0, 0.15)
            pay = float(payout(km, pay_cfg, late_min=actual - a.eta_min))
            rider.free_at, rider.lat, rider.lng = now + actual, o.clat, o.clng
            rider.earnings += pay
            rider.n += 1
            rows.append((now - o.created + actual, pay, km))
            served.add(o.id)
        pending = [o for o in pending if o.id not in served]

    arr = np.array(rows) if rows else np.zeros((1, 3))
    earn = [r.earnings for r in riders]
    return {
        "avg_time": float(arr[:, 0].mean()), "p90_time": float(np.percentile(arr[:, 0], 90)),
        "avg_pay": float(arr[:, 1].mean()), "avg_km": float(arr[:, 2].mean()),
        "gini": gini(earn), "delivered": len(rows), "backlog": len(pending), "earnings": earn,
    }
