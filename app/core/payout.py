from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PayoutConfig:
    base: float = 20.0          # Rs per order
    k1: float = 5.0             # Rs/km for 0-3 km
    k2: float = 8.0             # Rs/km for 3-6 km
    k3: float = 12.0            # Rs/km beyond 6 km
    grace_min: float = 10.0     # lateness allowed before penalty
    late_per_min: float = 1.0   # Rs deducted per late minute
    max_penalty_frac: float = 0.3


def payout(km, cfg: PayoutConfig = PayoutConfig(), late_min=0.0):
    """Slab-based, vectorised payout with a capped lateness penalty."""
    km = np.asarray(km, dtype=float)
    gross = cfg.base + cfg.k1 * np.minimum(km, 3) + cfg.k2 * np.clip(km - 3, 0, 3) + cfg.k3 * np.maximum(km - 6, 0)
    late = np.maximum(np.asarray(late_min, dtype=float) - cfg.grace_min, 0) * cfg.late_per_min
    return gross - np.minimum(late, cfg.max_penalty_frac * gross)
