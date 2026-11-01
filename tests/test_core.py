import numpy as np
from app.core import h3_index
from app.core.assignment import Weights, assign
from app.core.metrics import gini
from app.core.payout import PayoutConfig, payout
from app.core.types import OrderState, RiderState


def test_payout_slabs():
    assert payout(2) == 30 and payout(5) == 51 and payout(8) == 83


def test_payout_monotonic_and_late_penalty_capped():
    km = np.linspace(0, 15, 50)
    assert np.all(np.diff(payout(km)) >= 0)
    gross = payout(4)
    assert abs(payout(4, late_min=1000) - gross * (1 - PayoutConfig().max_penalty_frac)) < 1e-9
    assert payout(4, late_min=5) == gross


def test_gini_bounds():
    assert gini([5, 5, 5, 5]) == 0
    assert gini([0, 0, 0, 100]) > 0.7


def test_h3_roundtrip():
    c = h3_index.to_cell(12.9716, 77.5946)
    assert len(h3_index.ring(c, 2)) == 19
    assert h3_index.ring_distance(c, c) == 0


def test_fairness_shifts_assignment():
    # rider A is closest but already earns a lot; rider B is a bit farther and earns nothing
    riders = [RiderState("A", 12.9716, 77.5946, earnings=1000), RiderState("B", 12.9800, 77.5946, earnings=0)]
    orders = [OrderState(1, 12.9720, 77.5946, 12.9900, 77.5946)]
    near = assign(riders, orders, Weights(), nearest=True)
    fair = assign(riders, orders, Weights(1, 0, 3))
    assert near[0].rider_id == "A" and fair[0].rider_id == "B"


def test_assignment_is_one_to_one_and_handles_empty():
    assert assign([], [], Weights()) == []
    rs = [RiderState(str(i), 12.97 + i * 0.001, 77.59) for i in range(3)]
    os_ = [OrderState(i, 12.97, 77.59, 12.98, 77.60) for i in range(5)]
    res = assign(rs, os_, Weights())
    assert len(res) == 3 and len({a.rider_id for a in res}) == 3 and len({a.order_id for a in res}) == 3
