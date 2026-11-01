"""Train the ETA model on REAL phone GPS pings.

    python -m scripts.train_from_pings                 # reads the pings table of DATABASE_URL
    python -m scripts.train_from_pings --csv pings.csv # columns: rider_id,lat,lng,ts

How it works: for each rider, pings are sorted by time and split into trips (a trip ends after a
GAP of >5 min between pings or >=2 min standing still). Each trip gives one training row:
  km      = straight-line start->end distance * ROAD_FACTOR  (same feature the dispatcher computes)
  hour    = local hour the trip started
  minutes = how long it really took
The model therefore learns the real speed of whatever your phones were moving on (walking, bike, bike-taxi...).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.core.geo import ROAD_FACTOR, haversine_km
from app.ml import eta_model

GAP_MIN, STOP_KMH, STOP_MIN, MIN_KM, MIN_MIN = 5.0, 2.0, 2.0, 0.2, 2.0


def trips_from_pings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in df.sort_values("ts").groupby("rider_id"):
        t = pd.to_datetime(g["ts"]).to_numpy("datetime64[s]").astype("int64") / 60.0  # minutes
        lat, lng = g["lat"].to_numpy(), g["lng"].to_numpy()
        start = last_move = None
        still = 0.0
        for i in range(1, len(g)):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                continue
            speed = float(haversine_km(lat[i - 1], lng[i - 1], lat[i], lng[i])) / dt * 60
            moving = speed >= STOP_KMH
            if dt > GAP_MIN:
                still = STOP_MIN  # force a break
            elif moving:
                if start is None:
                    start = i - 1
                last_move, still = i, 0.0
            else:
                still += dt
            if start is not None and still >= STOP_MIN:
                rows.append(_trip(t, lat, lng, start, last_move, rid))
                start = last_move = None
            if still >= STOP_MIN:
                still = 0.0
        if start is not None and last_move is not None:
            rows.append(_trip(t, lat, lng, start, last_move, rid))
    out = pd.DataFrame([r for r in rows if r], columns=["rider_id", "km", "hour", "minutes", "path_km"])
    return out[(out.km >= MIN_KM) & (out.minutes >= MIN_MIN)].reset_index(drop=True)


def _trip(t, lat, lng, a, b, rid):
    if b is None or b <= a:
        return None
    km = float(haversine_km(lat[a], lng[a], lat[b], lng[b])) * ROAD_FACTOR
    path = float(np.sum(haversine_km(lat[a:b], lng[a:b], lat[a + 1:b + 1], lng[a + 1:b + 1])))
    local = (pd.Timestamp(int(t[a] * 60), unit="s") + pd.Timedelta(hours=settings.tz_offset_hours)).hour
    return (rid, km, int(local), float(t[b] - t[a]), path)


def load_pings(csv: str | None) -> pd.DataFrame:
    if csv:
        return pd.read_csv(csv, parse_dates=["ts"])
    from app.db import engine
    return pd.read_sql("SELECT rider_id, lat, lng, ts FROM pings", engine, parse_dates=["ts"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--min-trips", type=int, default=10)
    args = ap.parse_args()
    pings = load_pings(args.csv)
    print(f"{len(pings)} pings from {pings.rider_id.nunique() if len(pings) else 0} rider(s)")
    trips = trips_from_pings(pings) if len(pings) else pd.DataFrame()
    print(f"{len(trips)} usable trips")
    if len(trips) < args.min_trips:
        raise SystemExit(f"Need at least {args.min_trips} trips - keep the trackers running while you actually move around, then rerun.")
    Path("data").mkdir(exist_ok=True)
    trips.to_csv("data/trips.csv", index=False)
    print(trips.describe().round(2).to_string())
    print("ETA model:", eta_model.train(trips, Path(settings.model_dir) / "eta.joblib"))