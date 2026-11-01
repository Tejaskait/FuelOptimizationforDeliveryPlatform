"""python -m scripts.compare_eta -> does a learned ETA improve delivery time / payout fairness vs naive ETA?"""
from pathlib import Path

from app.config import settings
from app.core.assignment import Weights
from app.core.eta import default_eta
from app.ml.eta_model import EtaModel
from scripts.sweep import mean_run
from simulator.engine import SimConfig

if __name__ == "__main__":
    cfg, w = SimConfig(), Weights(1, 0.5, 1)
    learned = EtaModel.load(Path(settings.model_dir) / "eta.joblib")
    for name, fn in [("naive ETA", default_eta), ("learned ETA", learned)]:
        r = mean_run(cfg, w=w, eta_fn=fn)
        print(f"{name:12s} time={r['avg_time']:.1f}  pay={r['avg_pay']:.1f}  gini={r['gini']:.3f}  delivered={r['delivered']:.0f}")
