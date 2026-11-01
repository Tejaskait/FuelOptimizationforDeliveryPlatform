"""python -m scripts.sweep -> Pareto sweep over (beta, delta); writes reports/pareto.png + sweep.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.config import settings
from app.core.assignment import Weights
from app.ml.eta_model import EtaModel
from simulator.engine import SimConfig, run_sim

SEEDS = [1, 2, 3]
KEYS = ["avg_time", "p90_time", "avg_pay", "gini", "delivered", "backlog"]


def mean_run(cfg, seeds=SEEDS, **kw):
    runs = [run_sim(cfg, seed=s, **kw) for s in seeds]
    return {k: sum(r[k] for r in runs) / len(runs) for k in KEYS}


if __name__ == "__main__":
    cfg = SimConfig()
    eta = EtaModel.load(Path(settings.model_dir) / "eta.joblib")
    rows = [{"name": "nearest", "beta": 0, "delta": 0, **mean_run(cfg, w=Weights(), nearest=True, eta_fn=eta)}]
    for b in [0, 0.5, 1, 2, 3]:
        for d in [0, 0.5, 1, 2, 3]:
            rows.append({"name": f"b{b}_d{d}", "beta": b, "delta": d, **mean_run(cfg, w=Weights(1, b, d), eta_fn=eta)})
    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    s = df[df.name != "nearest"]
    sc = ax.scatter(s.avg_time, s.gini, c=s.avg_pay, cmap="coolwarm", s=70)
    ax.scatter(*df[df.name == "nearest"][["avg_time", "gini"]].values.T, c="k", marker="s", s=90, label="nearest baseline")
    plt.colorbar(sc, label="avg payout per order (Rs)")
    ax.set_xlabel("avg delivery time (min)"); ax.set_ylabel("earnings inequality (Gini)")
    ax.set_title("Speed vs rider fairness vs platform cost"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig("reports/pareto.png", dpi=130, bbox_inches="tight")
    print(df.round(2).to_string(index=False))
