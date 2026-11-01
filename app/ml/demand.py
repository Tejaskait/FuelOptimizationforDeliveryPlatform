"""Per-hexagon demand forecaster: orders per (H3 cell, hour, weekday). Drives rider repositioning hints."""
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ..core import h3_index

FEATURES = ["lat", "lng", "hour", "dow"]


def aggregate(log: pd.DataFrame) -> pd.DataFrame:
    df = log.copy()
    df["h3"] = [h3_index.to_cell(a, b) for a, b in zip(df.rlat, df.rlng)]
    g = df.groupby(["h3", "hour", "dow"]).size().rename("orders").reset_index()
    ll = g.h3.map(h3.cell_to_latlng)
    g["lat"], g["lng"] = ll.map(lambda p: p[0]), ll.map(lambda p: p[1])
    return g


def train(log: pd.DataFrame, path) -> dict:
    g = aggregate(log)
    Xtr, Xte, ytr, yte = train_test_split(g[FEATURES], g["orders"], test_size=0.2, random_state=0)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xtr, ytr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "cells": sorted(g.h3.unique())}, path)
    return {"mae": float(mean_absolute_error(yte, m.predict(Xte))), "baseline_mae": float(mean_absolute_error(yte, np.full(len(yte), ytr.mean()))), "cells": int(g.h3.nunique())}


def hotspots(path, hour: int, dow: int, top: int = 8) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    blob = joblib.load(p)
    cells = blob["cells"]
    ll = [h3.cell_to_latlng(c) for c in cells]
    X = pd.DataFrame({"lat": [a for a, _ in ll], "lng": [b for _, b in ll], "hour": hour, "dow": dow})
    pred = blob["model"].predict(X)
    order = np.argsort(-pred)[:top]
    return [{"h3": cells[i], "expected_orders": float(pred[i]), "boundary": h3_index.boundary(cells[i])} for i in order]
