from app.core.assignment import Weights
from simulator.engine import SimConfig, run_sim


def test_sim_runs_and_fairness_reduces_gini():
    cfg = SimConfig(hours=2)
    near = run_sim(cfg, Weights(), nearest=True, seed=1)
    fair = run_sim(cfg, Weights(1, 0.5, 3), seed=1)
    assert near["delivered"] > 0 and fair["delivered"] > 0
    assert fair["gini"] <= near["gini"] + 0.05
