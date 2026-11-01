from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..core.assignment import Weights, assign
from ..core.payout import payout
from ..core.types import OrderState, RiderState
from ..db import utcnow
from ..ml.eta_model import EtaModel
from ..models import Order, Rider

weights = Weights(settings.w_time, settings.w_pay, settings.w_fair)


@lru_cache(maxsize=1)
def get_eta() -> EtaModel:
    return EtaModel.load(Path(settings.model_dir) / "eta.joblib")


def local_hour() -> int:
    return int((utcnow().hour + settings.tz_offset_hours) % 24)


def run_batch(db: Session) -> list[dict]:
    now = utcnow()
    cutoff = now - timedelta(minutes=5)
    riders = [r for r in db.scalars(select(Rider).where(Rider.status == "idle"))
              if r.is_virtual or (r.last_seen and r.last_seen >= cutoff)]
    orders = list(db.scalars(select(Order).where(Order.status == "pending").order_by(Order.created_at)))
    if not riders or not orders:
        return []
    mean = float(np.mean(list(db.scalars(select(Rider.earnings))) or [0.0]))
    rs = [RiderState(r.id, r.lat, r.lng, r.earnings) for r in riders]
    os_ = [OrderState(o.id, o.rlat, o.rlng, o.clat, o.clng, wait_min=(now - o.created_at).total_seconds() / 60) for o in orders]
    res = assign(rs, os_, weights, eta_fn=get_eta(), hour=local_hour(), fleet_mean=mean, max_ring=settings.max_ring)
    rmap, omap, out = {r.id: r for r in riders}, {o.id: o for o in orders}, []
    for a in res:
        r, o = rmap[a.rider_id], omap[a.order_id]
        o.status, o.rider_id, o.assigned_at = "assigned", r.id, now
        o.km, o.eta_min, o.payout = a.pickup_km + a.drop_km, a.eta_min, a.payout
        r.status = "busy"
        out.append(asdict(a))
    db.commit()
    return out


def complete_order(db: Session, o: Order) -> dict:
    now = utcnow()
    actual = (now - o.assigned_at).total_seconds() / 60
    pay = float(payout(o.km, late_min=actual - o.eta_min))
    o.status, o.delivered_at, o.payout = "delivered", now, pay
    r = db.get(Rider, o.rider_id)
    r.earnings += pay
    r.n_orders += 1
    r.status = "idle"
    if r.is_virtual:
        r.lat, r.lng = o.clat, o.clng
    db.commit()
    return {"order_id": o.id, "rider_id": r.id, "minutes": actual, "payout": pay}
