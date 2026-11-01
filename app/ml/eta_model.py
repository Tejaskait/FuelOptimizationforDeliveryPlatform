"""Learned ETA (gradient boosting) replacing the naive constant-speed baseline."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ..core.eta import RUSH_HOURS, default_eta


def features(km, hour) -> np.ndarray:
    km = np.asarray(km, dtype=float).ravel()
    hour = np.broadcast_to(np.asarray(hour), km.shape)
    return np.column_stack([km, np.isin(hour, list(RUSH_HOURS)).astype(float)])


class EtaModel:
    def __init__(self, model=None):
        self.model = model

    def __call__(self, km, hour: int):
        km = np.asarray(km, dtype=float)
        if self.model is None:
            return default_eta(km, hour)
        return self.model.predict(features(km, hour)).reshape(km.shape)

    @classmethod
    def load(cls, path) -> "EtaModel":
        p = Path(path)
        return cls(joblib.load(p) if p.exists() else None)


def train(df: pd.DataFrame, path) -> dict:
    """df columns: km (road-distance estimate), hour (0-23), minutes (observed duration)."""
    X, y = features(df["km"], df["hour"].to_numpy()), df["minutes"].to_numpy()
    small = len(df) < 200
    m = GradientBoostingRegressor(n_estimators=60 if small else 150, max_depth=2 if small else 3, random_state=0)
    if len(df) < 30:  # too little data for a held-out split: fit on all, report in-sample error
        m.fit(X, y)
        Xte, yte, held_out = X, y, False
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        m.fit(Xtr, ytr)
        held_out = True
    hours = np.where(Xte[:, 1] > 0, 8, 14)  # any rush / non-rush hour reproduces default_eta's speed switch
    naive = np.array([default_eta(k, int(h)) for k, h in zip(Xte[:, 0], hours)])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, path)
    return {"n_samples": len(df), "held_out_eval": held_out,
            "mae_model": float(mean_absolute_error(yte, m.predict(Xte))), "mae_naive": float(mean_absolute_error(yte, naive))}